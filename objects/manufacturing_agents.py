from ast import Or
from logging.config import valid_ident
from objects.machines import Machine
from objects.buffer import *
from objects.rulesets import RuleSet
import configs.models
from utils.consecutive_performable_tasks import consecutive_performable_tasks
from utils.devisions import div_possible_zero
from configs.dict_pos_types import dict_pos_types
import pandas as pd
import numpy as np
from copy import copy
import random
import csv
import time
from utils import time_tracker, reward_layer


def set_agent_seed(seed:int):
    random.seed(seed)


class ManufacturingAgent:
    instances = []

    def __init__(self, config: dict, env: simpy.Environment, position, ruleset_id=None):

        self.env = env
        self.simulation_environment = None

        t = time.localtime()
        self.timestamp = time.strftime('_%Y-%m-%d_%H-%M', t)

        self.lock = None
        self.count = 0
        self.count_smart = 0
        # Attributes
        self.ruleset = None
        self.ruleset_train = None
        for ruleset in RuleSet.instances:
            if ruleset.id == ruleset_id:
                self.ruleset = ruleset
                  # Reference to the priority ruleset of the agent
                break
        
        self.ranking_criteria = [criteria["measure"] for criteria in self.ruleset.numerical_criteria]

        if not self.ruleset:  # Check if the Agent has a Ruleset selected
            raise Exception(
                "Atleast one Agent has no ruleset defined. Please choose a ruleset or the agent wont do anything!")
    
        self.ruleset_assist = None
        ruleset_assist_id = 4
        for ruleset in RuleSet.instances:
            if ruleset.id == ruleset_assist_id:
                self.ruleset_assist = ruleset
                  # Reference to the priority ruleset of the agent
                break

        self.ranking_criteria_assist = [criteria["measure"] for criteria in self.ruleset_assist.numerical_criteria]
        self.ruleset_temp = None

        self.cell = None

        #for distribution and manufacturing cell 
        self.distribution_simple = True
        self.distribution_opt = False
        self.distribution_smart = False
        
        self.speed = config["AGENT_SPEED"]  # Configured moving speed of the agent: How much distance can be moved within one time points
        self.time_for_item_pick_up = config["TIME_FOR_ITEM_PICK_UP"]
        self.time_for_item_store = config["TIME_FOR_ITEM_STORE"]

        # State
        self.moving = False  # Is the agent currently moving from one position to another?
        self.position = position  # Position object of the agent, None if agent is currently moving
        self.next_position = None  # Destination if agent is currently moving
        self.moving_time = 0  # How long does it take the agent to perform the whole route
        self.moving_start_time = None  # When did the agent start moving
        self.moving_start_position = None  # Where did the agent start moving
        self.remaining_moving_time = 0  # How much moving time of the current route is remaining
        self.moving_end_time = None  # Estimated Time point on which the agent will arrive

        self.waiting = False  # Agent has an active waiting task, only interruptable by the position or after a specific time passed (LONGEST_WAITING_TIME)
        self.has_task = False  # Has the agent an active task it performs? Waiting counts as task...

        self.locked_item = None  # Item locked by this agent. Locked items are not interactable by other agents
        self.picked_up_item = None  # Item the agent is holding, only one at a time

        self.started_tasks = 0  # Amount of started tasks

        # Current tasks
        self.current_task = None  # The current task the agent is performing
        self.current_subtask = None  # Current subtask the agent is performing (Subtasks are part of the current task e.g. "move to position x" as part of "bring item y from z to x")
        self.current_waitingtask = None  # Current waiting task. Agents starts waiting task if its subtask/task cant be performed currently (e.g. wait for processing of item in machine)

        self.__class__.instances.append(self)

        self.env.process(self.initial_event())  # Write initial event in event log when simulation starts
        self.main_proc = self.env.process(self.main_process())  # Initialize first main process of the agent when simulation starts

    def main_process(self):
        """Main process of the agent. Decisions about its behavior are made in this process.
        Will call Tasks after calculating the next task to be done.
        """

        if not self.cell.orders_available():
            return
        self.lock.acquire()

        # Get state of cell and orders inside this cell
        state_calc_start = time.time()
        cell_state = self.cell.get_cell_state(requester=self)  
        time_tracker.time_state_calc += time.time() - state_calc_start

        # For each order in state add the destination if this order would be chosen
        dest_calc_start = time.time()
        cell_state["_destination"] = cell_state.apply(self.add_destinations, axis=1)
        time_tracker.time_destination_calc += time.time() - dest_calc_start   
        self.ranking_criteria = [criteria["measure"] for criteria in self.ruleset.numerical_criteria]

        # Get action depending on agent ruleset
        if self.ruleset.dynamic:
            next_task, next_order, destination, base_state, state_RL, action, action_RL = self.get_smart_action(cell_state)
        elif self.ruleset.dynamic_dispatch:
            next_task, next_order, destination, base_state, state_RL, action, action_RL = self.get_smart_dispatch_rule(cell_state)
        else:
            now = time.time()
            next_task, next_order, destination, base_state = self.get_action(cell_state)
            time_tracker.time_action_calc += time.time() - now
        # Perform next task if there is one
        if next_task:
            self.current_task = next_task
            self.has_task = True
            self.save_event("start_task")
            self.started_tasks += 1

            task_started_at = self.env.now

            # Lock next order
            if next_order:
                next_order.locked_by = self
                self.locked_item = next_order
                self.locked_item.save_event("locked")
                self.announce_arrival(next_order, destination)

            self.lock.release()

            # Perform task
            yield next_task

            self.has_task = False
            self.save_event("end_of_main_process")

            state_calc_start = time.time()
            # Get new state
            new_cell_state = self.cell.get_cell_state(requester=self)
            time_tracker.time_state_calc += time.time() - state_calc_start
            dest_calc_start = time.time()
            new_cell_state["_destination"] = new_cell_state.apply(self.add_destinations, axis=1)
            time_tracker.time_destination_calc += time.time() - dest_calc_start
            new_cell_state = self.state_to_numeric(copy(new_cell_state))
            if self.get_processable_orders(cell_state) > 1:
                if not self.ruleset.dynamic and not self.ruleset.dynamic_dispatch:
                    action = self.get_heuristics_action_index(cell_state, next_order)
                    self.finished_heuristic_action(cell_state, new_cell_state, base_state, next_order, self.env.now - task_started_at, action)

                if self.ruleset.dynamic or self.ruleset.dynamic_dispatch:  # Check rewards            
                    #reward function tbd
                    self.finished_smart_action(cell_state, new_cell_state, base_state, state_RL, next_order, action, action_RL)
 

            # Start new main process
            self.main_proc = self.env.process(self.main_process())

        if self.lock.locked():
            self.lock.release()
    

    def get_action(self, order_state): #get action with neural network but plain dispatch rules
        """Gets an action by using the priority attributes defined in agents ruleset
        :param order_state: Pandas Dataframe, categorical state of the cell
        :return task: simpy process to be performed next
        :return next_order: order to be moved
        :return destination: destination where the order will be brought to""" 
        state_numeric = self.state_to_numeric(copy(order_state))

        order = order_state[(order_state["order"].notnull())]
        
        useable_orders = order[(order["locked"] == 0) & (order["in_m_input"] == 0) & (order["in_m"] == 0) & (order["in_same_cell"] == 1)]

        if useable_orders.empty:
            return None, None, None, None

        useable_with_free_destination = useable_orders[useable_orders["_destination"] != -1]

        if useable_with_free_destination.empty:
            return None, None, None, None

        elif len(useable_with_free_destination) == 1:
            next_order = useable_with_free_destination["order"].iat[0]

        elif self.ruleset.random:  # When Ruleset is random...
            ranking = useable_with_free_destination.sample(frac=1, random_state=self.ruleset.seed).reset_index(
                drop=True)
            next_order = ranking["order"].iat[0]

        else:
            criteria = [criteria["measure"] for criteria in self.ruleset.numerical_criteria]
            ranking = useable_with_free_destination.loc[:, ["order"] + criteria]

            for criterion in self.ruleset.numerical_criteria: #Place where heuristics apply
                weight = criterion["weight"]
                measure = criterion["measure"]
                order = criterion["ranking_order"]

                max_v = ranking[measure].max()
                min_v = ranking[measure].min()

                # Min Max Normalisation
                if order == "ASC":
                    ranking["WS-" + measure] = weight * div_possible_zero((ranking[measure] - min_v), (max_v - min_v))
                else:
                    ranking["WS-" + measure] = weight * (1 - div_possible_zero((ranking[measure] - min_v), (max_v - min_v)))

            order_scores = ranking.filter(regex="WS-")
            ranking.loc[:, "Score"] = order_scores.sum(axis=1)
            ranking.sort_values(by=["Score"], inplace=True)

            next_order = ranking["order"].iat[0]
            

        destination = useable_with_free_destination[useable_with_free_destination["order"] == next_order].reset_index(drop=True).loc[0, "_destination"]

        if destination:
            return self.env.process(self.item_from_to(next_order, next_order.position, destination)), next_order, destination, state_numeric
        else:
            return None, None, None, None

    def get_RL_state(self, order_state, available_destinations): # get flatted state vector with chosen information
        state_due_to = order_state.loc[:, "due_to"] 
        state_priority = order_state.loc[:, "priority"] 
        time_in_cell = order_state.loc[:, "time_in_cell"] 
        state_priority = np.multiply(available_destinations, state_priority)
        state_state_priority = state_priority / 2
        state_due_to_available = np.multiply(available_destinations, state_due_to) # only due_to values for orders with destination -> nothing in machine etc.
        time_in_cell_available = np.multiply(available_destinations, time_in_cell)
        max_time_in_cell = max(abs(i) for i in time_in_cell_available)
        max_due_to = max(abs(i) for i in state_due_to_available)
        
        if max_due_to != 0:
            state_due_to_normalized = [x / max_due_to for x in state_due_to_available] # only due_to values for orders with destination -> nothing in machine etc.
        else:
            state_due_to_normalized = np.zeros(len(state_due_to_available))

        if max_time_in_cell != 0:
            state_time_in_cell_normalized = [x / max_time_in_cell for x in time_in_cell_available]
        else:
            state_time_in_cell_normalized = np.zeros(len(time_in_cell_available))
        
        state_RL = []
        for i in range(len(state_due_to_available)): #(2) look for orders on valid places
            state_RL.append(state_due_to_normalized[i])
            state_RL.append(state_time_in_cell_normalized[i])
            state_RL.append(state_state_priority[i])    

        return state_RL

    def get_available_destinations(self, order_state): #get numerized _destination space
        destination = order_state.loc[:, "_destination"]
        available_destinations = []
        for i in range(len(destination)): #(2) look for orders on valid places
            if destination[i] == -1:
                available_destinations.append(0)
            else:
                available_destinations.append(1)
        return available_destinations

    def get_smart_dispatch_rule(self, order_state):
        """Gets an action by using an dynamic reinforcement learning model defined in agents ruleset
        :param order_state: Pandas Dataframe, categorical state of the cell
        :return task: simpy process to be performed next
        :return next_order: order to be moved
        :return destination: d estination where the order will be brought to"""
        smart_agent = self.ruleset.reinforce_agent
        #smart_agent = configs.models.ReinforceAgent(11, 3)
        #get state with orders on the various slots
        state_numeric = self.state_to_numeric(copy(order_state))
        available_destinations = self.get_available_destinations(state_numeric)

        order = order_state[(order_state["order"].notnull())]
        useable_orders = order[(order["locked"] == 0) & (order["in_m_input"] == 0) & (order["in_m"] == 0) & (order["in_same_cell"] == 1)]
        
        if useable_orders.empty:
            return None, None, None, None, None, None, None
        useable_with_free_destination = useable_orders[useable_orders["_destination"] != -1]
        
        if useable_with_free_destination.empty:
            return None, None, None, None, None, None, None
        elif len(useable_with_free_destination) == 1:
            next_order = useable_with_free_destination["order"].iat[0]
            state_RL, action_RL = None, None
        else:
            state_RL = self.get_RL_state(state_numeric, available_destinations) 
            now = time.time()
            action_RL = smart_agent.get_dispatch_rule(state_RL) #smart agent thinking...
            time_tracker.time_smart_action_calc += time.time() - now

            #ranking = self.pre_ordering(4, useable_with_free_destination)
            possible_dispatch_rules = [3,4,9]
            for ruleset in RuleSet.instances:
                if ruleset.id == possible_dispatch_rules[action_RL]:
                    self.ruleset_temp = ruleset # Reference to the choosen ruleset of the smart agent
                    break
            
            self.ranking_criteria_assist = [criteria["measure"] for criteria in self.ruleset_temp.numerical_criteria]
            criteria = [criteria["measure"] for criteria in self.ruleset_temp.numerical_criteria]

            if self.ruleset_temp.id == 9: #pre-sorting for due_to rule
                for ruleset in RuleSet.instances:
                    if ruleset.id == 4:
                        ruleset_due_to = ruleset # Reference to the choosen ruleset of the smart agent
                        break
                self.ranking_criteria_assist = [criteria["measure"] for criteria in ruleset_due_to.numerical_criteria]
                criteria_temp = [criteria["measure"] for criteria in ruleset_due_to.numerical_criteria]
                ranking = useable_with_free_destination.reindex(columns = (["order"] + criteria_temp + criteria))
                print("ranking 1: ", ranking)
                for criterion in ruleset_due_to.numerical_criteria:
                    weight = criterion["weight"]
                    measure = criterion["measure"]
                    order = criterion["ranking_order"]

                    max_v = ranking[measure].max()
                    min_v = ranking[measure].min()

                    # Min Max Normalisation
                    if order == "ASC":
                        ranking["WS-" + measure] = weight * div_possible_zero((ranking[measure] - min_v), (max_v - min_v))
                    else:
                        ranking["WS-" + measure] = weight * (1 - div_possible_zero((ranking[measure] - min_v), (max_v - min_v)))
                order_scores = ranking.filter(regex="WS-")
                ranking.loc[:, "Score"] = order_scores.sum(axis=1)
                ranking.sort_values(by=["Score"], inplace=True)
                ranking.drop(columns=["Score"])
                print("ranking 2: ", ranking)
           
            if self.ruleset_temp.id != 9:
                ranking = useable_with_free_destination.reindex(columns = (["order"] + criteria))
        
            for criterion in self.ruleset_temp.numerical_criteria:
                weight = criterion["weight"]
                measure = criterion["measure"]
                order = criterion["ranking_order"]

                max_v = ranking[measure].max()
                min_v = ranking[measure].min()

                # Min Max Normalisation
                if order == "ASC":
                    ranking["WS-" + measure] = weight * div_possible_zero((ranking[measure] - min_v), (max_v - min_v))
                else:
                    ranking["WS-" + measure] = weight * (1 - div_possible_zero((ranking[measure] - min_v), (max_v - min_v)))

            order_scores = ranking.filter(regex="WS-")
            ranking.loc[:, "Score"] = order_scores.sum(axis=1)
            ranking.sort_values(by=["Score"], inplace=True)
            print("ranking 3: ", ranking)
            next_order = ranking["order"].iat[0]
        
        action = self.get_heuristics_action_index(order_state, next_order)
        # action_next_order = order_state[order_state["order"]==next_order].index.values
        # action = action_next_order
        destination = useable_with_free_destination[useable_with_free_destination["order"] == next_order].reset_index(drop=True).loc[0, "_destination"]
        # next_task, next_order, destination, base_state_flat, action, dynamic_temp
        if destination:
            return self.env.process(self.item_from_to(next_order, next_order.position, destination)), next_order, destination, state_numeric, state_RL, action, action_RL
        else:
            return None, None, None, None, None, None, None

    def pre_ordering(self, dispatch_rule, useable_with_free_destination):
        for ruleset in RuleSet.instances:
            if ruleset.id == dispatch_rule:
                self.ruleset_temp = ruleset # Reference to the choosen ruleset of the smart agent
                break
            self.ranking_criteria_assist = [criteria["measure"] for criteria in self.ruleset_temp.numerical_criteria]
            criteria = [criteria["measure"] for criteria in self.ruleset_temp.numerical_criteria]            
            #ranking = useable_with_free_destination.loc[:, ["order"] + criteria]
            ranking = useable_with_free_destination.reindex(columns = (["order"] + criteria))

            for criterion in self.ruleset_temp.numerical_criteria:
                weight = criterion["weight"]
                measure = criterion["measure"]
                order = criterion["ranking_order"]

                max_v = ranking[measure].max()
                min_v = ranking[measure].min()

                # Min Max Normalisation
                if order == "ASC":
                    ranking["WS-" + measure] = weight * div_possible_zero((ranking[measure] - min_v), (max_v - min_v))
                else:
                    ranking["WS-" + measure] = weight * (1 - div_possible_zero((ranking[measure] - min_v), (max_v - min_v)))

            order_scores = ranking.filter(regex="WS-")
            ranking.loc[:, "Score"] = order_scores.sum(axis=1)
            ranking.sort_values(by=["Score"], inplace=True)   
        return ranking

    def get_heuristics_action_index(self, order_state, next_order):
        action_next_order = order_state[order_state["order"]==next_order].index.values
        return action_next_order

    def get_smart_action(self, order_state):
        """Gets an action by using an dynamic reinforcement learning model defined in agents ruleset

        :param order_state: Pandas Dataframe, categorical state of the cell
        :return task: simpy process to be performed next
        :return next_order: order to be moved
        :return destination: destination where the order will be brought to"""
        smart_agent = self.ruleset.reinforce_agent  
        state_numeric = self.state_to_numeric(copy(order_state))
        available_destinations = self.get_available_destinations(state_numeric)
        # Get action space
        action_space = range(0, len(state_numeric)) # +1 
        
        state_RL = self.get_RL_state(state_numeric, available_destinations)
        action, smart_action = smart_agent.get_action(state_RL) # action = Q-vector <---------- !!!
        if smart_action: #action by deep RL
            action, action_RL = self.get_valid_smart_action(action, order_state)
            # Get next order and destination or idle mode
            if action < len(action_space):
                # Normal action
                next_order = order_state.at[action, "order"]
                destination = order_state.at[action, "_destination"] 
            else: 
                # Take no action - idle mode
                smart_agent.appendMemory(smart_agent, former_state=state_RL, new_state=state_RL, action = action_RL, reward=0)
                return None, None, None, None, None, None, None # next_task, next_order, destination, base_state_flat, action
        
            penalty = reward_layer.evaluate_choice(state_numeric.loc[action])
            if penalty < 0:
                #action = self.get_RL_action_index(action)
                smart_agent.appendMemory(smart_agent, former_state=state_RL, new_state=state_RL, action= action_RL, reward=penalty)
                return None, None, None, None, None, None, None # next_task, next_order, destination, base_state_flat, action, dynamic_temp
            else:
                return self.env.process(self.item_from_to(next_order, next_order.position, destination)), next_order, destination, state_numeric, state_RL, action, action_RL

        else: #choosen assist_rule for assistance as defined above
            order = order_state[(order_state["order"].notnull())]
            useable_orders = order[(order["locked"] == 0) & (order["in_m_input"] == 0) & (order["in_m"] == 0) & (order["in_same_cell"] == 1)]

            if useable_orders.empty:
                return None, None, None, None, None, None, None
            useable_with_free_destination = useable_orders[useable_orders["_destination"] != -1]

            if useable_with_free_destination.empty:
                return None, None, None, None, None, None, None
            elif len(useable_with_free_destination) == 1:
                next_order = useable_with_free_destination["order"].iat[0]
            elif self.ruleset_assist.random:  # When Ruleset is random...
                ranking = useable_with_free_destination.sample(frac=1, random_state=self.ruleset_assist.seed).reset_index(drop=True)
                next_order = ranking["order"].iat[0]
            else:
                criteria = [criteria["measure"] for criteria in self.ruleset_assist.numerical_criteria]            
                #ranking = useable_with_free_destination.loc[:, ["order"] + criteria]
                ranking = useable_with_free_destination.reindex(columns = (["order"] + criteria))

                for criterion in self.ruleset_assist.numerical_criteria:
                    weight = criterion["weight"]
                    measure = criterion["measure"]
                    order = criterion["ranking_order"]

                    max_v = ranking[measure].max()
                    min_v = ranking[measure].min()

                    # Min Max Normalisation
                    if order == "ASC":
                        ranking["WS-" + measure] = weight * div_possible_zero((ranking[measure] - min_v), (max_v - min_v))
                    else:
                        ranking["WS-" + measure] = weight * (1 - div_possible_zero((ranking[measure] - min_v), (max_v - min_v)))

                order_scores = ranking.filter(regex="WS-")
                ranking.loc[:, "Score"] = order_scores.sum(axis=1)
                ranking.sort_values(by=["Score"], inplace=True)
                next_order = ranking["order"].iat[0]

            #find corresponding action as if choosen by RL algo
            # action_next_order = order_state.loc[order_state["order"] == next_order]
            action_next_order = order_state[order_state["order"]==next_order].index.values
            action = action_next_order
            destination = useable_with_free_destination[useable_with_free_destination["order"] == next_order].reset_index(drop=True).loc[0, "_destination"]
            # next_task, next_order, destination, base_state_flat, action, dynamic_temp
            if destination:
                return self.env.process(self.item_from_to(next_order, next_order.position, destination)), next_order, destination, state_numeric, state_RL, action, action_RL
            else:
                return None, None, None, None, None, None, None
       
    def get_valid_smart_action(self, action, order_state): #Calculate valid action for certain cell_arrangements and order_states
        valid_actions = [] # layout dependent 
        if self.ruleset.id == 10:
            valid_actions = [0,1,4,5,6,10] #(1) positions in whole state vector first column that are valid
        #add more cell layouts here <------
        order = []

        for i in valid_actions: #(2) look for orders on valid positions
            if order_state.loc[i, "_destination"] == -1:
                order.append(0)
            else:
                order.append(1)
        order_without_idle = list(order)
        order.append(1) #idle action

        action = np.multiply(action, order) #no order means no reasonable action
        valid_actions.append(valid_actions[len(valid_actions)-1] + 1) #append idle as last action behind every other action
        if np.sum(order_without_idle) == 0: #no jobs means automated idle mode
            return valid_actions[len(valid_actions)-1], len(valid_actions)-1 #return layout action and DRL action
        else:
            place = np.argmax(action[0])       
            return(valid_actions[place]), place

    def get_RL_action_index(self, action): #find corresponding action
        valid_actions = []
        if self.ruleset.id == 10:
            valid_actions = [0,1,4,5,6,10] #(1) positions in whole state vector first column that are valid
        valid_actions.append(valid_actions[len(valid_actions)-1] + 1)
        action_RL = 0
        i = 0
        for i in range(len(valid_actions)):
            if valid_actions[i] == action:
                action_RL = i
                return action_RL

    def finished_heuristic_action(self, old_state, new_state, old_state_flat, order, time_passed, action): # calc reward and append memory
        # new_state_flat = list(self.state_to_numeric(copy(new_state)).to_numpy().flatten())
        self.count = self.count + 1
        processable_orders = self.get_processable_orders(old_state)
        priority = old_state.loc[action, "priority"].values[0]
        if processable_orders > 1:
            reward = reward_layer.reward_heuristic(old_state, new_state, order, action)
            agent_name = str(self)
            agent_name = agent_name[-14:-1]
            parent = str(self.cell.parent)
            try:
                parent = parent[-14:-1]
            except:
                parent = None
            with open('../result/rewards' + self.timestamp + '_' + agent_name + '_level-' + str(self.cell.level) + '_parent-' + parent + '_rule-' + str(self.ruleset.id) +  '.csv', 'a+', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(list([self.cell.id, self.ruleset.id, agent_name, self.count, reward, priority, processable_orders, action]))
        
        

    def finished_smart_action(self, old_state, new_state, old_state_flat, state_RL, order, action, action_RL): # calc reward and append memory
        """Calculate reward for smart action and inform reinforcement agent about the state changes
        :param old_state: The state at action decision (categorical)
        :param new_state: Current state after finished task (categorical)
        :param old_state_flat: Flat state at action decision (numeric)
        :param order: (Order object) The moved order from finished taskc
        :param time_passed: (float) Time passed between action decision and finished task
        :param action: (int) The chosen action""" 
        smart_agent = self.ruleset.reinforce_agent
        available_destinations = self.get_available_destinations(new_state)
        new_state_RL = self.get_RL_state(new_state, available_destinations)
        # new_state_flat = list(self.state_to_numeric(copy(new_state)).to_numpy().flatten())        
        self.count_smart = self.count_smart + 1
        processable_orders = self.get_processable_orders(old_state)
        priority = old_state.loc[action, "priority"].values[0]
        if processable_orders  > 1: #if more than one order was apparent. 0,1: no AI necessary
            agent_name = str(self)
            agent_name = agent_name[-14:-1]
            parent = str(self.cell.parent)
            try:
                parent = parent[-14:-1]
            except:
                parent = None
            if not self.cell.machines:
                reward = reward_layer.reward_smart_dispatch(old_state, new_state, order, action)                                
            else:
                reward = reward_layer.reward_smart_dispatch(old_state, new_state, order, action)           
            with open('../result/rewards' + self.timestamp + '_' + agent_name + '_level-' + str(self.cell.level) + '_parent-' + parent + '_rule-' + str(self.ruleset.id) +  '.csv', 'a+', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(list([self.cell.id, self.ruleset.id, agent_name, self.count_smart, reward, priority, processable_orders, action, action_RL]))
            smart_agent.appendMemory(smart_agent, state_RL, new_state_RL, action_RL, reward)

    def get_processable_orders(self, old_state):
            # print("orders: ", len(old_state["order"]) - sum(x is not None for x in old_state["order"]), "; ", reward)
            old_cell_state_due_to = old_state.loc[:, "due_to"]

            #get due_to values for all orders that have a destination
            destination = old_state.loc[:, "_destination"]
            available_destinations = []
            for i in range(len(destination)): #(2) look for orders on valid places, if not valid; nan
                if destination[i] == -1:
                    available_destinations.append(np.nan)
                else:
                    available_destinations.append(1)
            due_to_values = np.multiply(old_cell_state_due_to, available_destinations)
            relevant_due_tos = [x for x in due_to_values if np.isnan(x) == False]
            return len(relevant_due_tos)
        

    def add_destinations(self, data):
        """Takes an column of a state and add the destination where the order would go to if chosen by the agent:
        :param data: (pd.Series) Column of state
        :return destination: Calculated destination or -1 if no useable order or no available destination for that order"""

        useable_order = (pd.notnull(data["order"])) and (data["locked"] == 0) and (data["in_m_input"] == 0) and (
                data["in_m"] == 0)

        if useable_order:
            destination = self.calculate_destination(data["order"])

            if destination:
                return destination

        return -1

    def calculate_destination(self, order):
        """Calculate the best next position for an agent to bring an order

        :param order: (Order object) Item to be calculated
        :return destination: Machine or Buffer within the Cell where order would be brought to. None if no destination is available
        """

        destination = None

        if order.current_cell is not self.cell:
            return destination

        next_processing_step = order.next_task
        next_steps = order.remaining_tasks

        #Prio1: if order is fininshed bring it to output or cell storage to prevent machine bloackage
        if order.tasks_finished or next_processing_step not in [task for (task, amount) in self.cell.performable_tasks
                                                                if amount > 0]:

            if self.cell.output_buffer.free_slots():
                destination = self.cell.output_buffer
            elif self.cell.storage.free_slots():
                destination = self.cell.storage
            else:
                return None

        elif self.cell.machines:
            # Order is in machine cell

            # Machines in cell that can perform the next task of the order
            potential_machines = [(
                                  machine, machine.item_in_input, machine.item_in_machine, len(machine.expected_orders),
                                  machine.current_setup) for machine in self.cell.machines if
                                  next_processing_step == machine.performable_task]

            # Machines that have a free input and are already in the right setup for the order
            optimal_machines = [machine for (machine, item_input, item_machine, expected_orders, setup) in
                                potential_machines if
                                item_input is None and expected_orders == 0 and setup == order.type]

            if len(optimal_machines) > 0:
                # Prefer one of the optimal machines
                destination = optimal_machines[0]

            else:

                # Possible machines that have a free input slot
                free_machines = [machine for (machine, item_input, item_machine, expected_orders, setup) in
                                 potential_machines if item_input is None and expected_orders == 0]

                # Prefer one of the free machines
                if len(free_machines) > 0:
                    destination = free_machines[0]

            if destination is None:

                # No machine could be used
                if self.cell.storage.free_slots() and order.position is not self.cell.storage:
                    # Bring order to storage buffer if a free slot is available
                    destination = self.cell.storage

                
                # elif self.cell.output_buffer.free_slots() and order.position is not self.cell.output_buffer:
                    # Alternative: Bring order to cell output to minize the amount of items in this cell
                    # destination = self.cell.output_buffer

        else:
            # Order is in distribution cell

            if self.distribution_simple:
                destination = self.simple_distribution(next_processing_step)
            elif self.distribution_opt:
                destination = self.optimized_distribution(order, next_steps)
            elif self.distribution_smart:
                destination = self.smart_distribution(order, next_steps)

            if not destination:
                if self.cell.storage.free_slots() and order.position is not self.cell.storage:
                    destination = self.cell.storage

        if destination == order.position:
            return None
        return destination

    def simple_distribution(self, next_processing_step):
        """Item is in distribution cell: Get next position for the item. Always choose random between an useable child that has a free input slot.
                :param next_processing_step: The next processing steps that should be performed on this item
                :return destination: Calculated by simple distribution"""

        # Input of cell has free slots
        possibilities = [(cell, dict(cell.performable_tasks)) for cell in self.cell.childs if cell.input_buffer.free_slots()]
        # Next task can be processed by this cell/tree branch
        possibilities = [cell.input_buffer for (cell, tasks) in possibilities if tasks[next_processing_step] > 0]

        if len(possibilities) > 1:
            return random.choice(possibilities)
        elif possibilities:
            return possibilities[0]
        else:
            return None

    def optimized_distribution(self, item, next_steps):
        """Item is in distribution cell: Get next position for the item. Always prefer those child cell that have an free
        input slot and can perform the maximal amount of consecutive processing steps for this item. Not useful for
        full utilization and setups with high storage, input buffer capacities.
        :param item: (Order object) The item for which the destination should be calculated
        :param next_steps: list of all next processing steps that should be performed on this item
        :return destination: Calculated by optimized distribution"""

        destination = None
        possibilities = [(cell, cell.check_best_path(item, include_all=False), cell.performable_tasks) for cell in self.cell.childs]
        
        # Check all Child cells and sort by least amount of
        # manufacturing cells needed to completely process this order

        best_possibilities = sorted(
            [(cell, shortest_path, cell.input_buffer.free_slots()) for (cell, shortest_path, performable_tasks) in
             possibilities if shortest_path], key=lambda tup: tup[1])

        free_best_destinations = [cell.input_buffer for (cell, shortest_path, free_slots) in best_possibilities if
                                  free_slots]

        if free_best_destinations:
            destination = free_best_destinations[0]

        else:
            # Prefer the one that can perform the most continuous tasks and has a free Input Slot.
            result = [(cell, consecutive_performable_tasks(next_steps, performable_tasks)) for
                      (cell, shortest_path, performable_tasks) in possibilities]

            result = sorted([(cell, amount) for (cell, amount) in result if amount > 0], key=lambda tup: tup[1],
                            reverse=True)

            if result:
                for cell, amount in result:
                    if not cell.input_buffer.full:
                        best_cell = cell
                        destination = best_cell.input_buffer
                        break

        return destination
    
    def smart_distribution(self, item, next_steps):

        return None 
    

    def item_from_to(self, item, from_pos, to_pos):
        """TASK: Pick up an item and store it at another position within his cell. Performs all needed Subtasks

        :param item: (Order object) The item that should be moved
        :param from_pos: (Buffer or Machine object) Position where the item should be picked up
        :param to_pos: (Buffer or Machine object) Position where the item should be stored in the end
        """
        if from_pos == to_pos:
            raise Exception("The item is already at its target destination")

        # Move to position where the item is
        if self.position != from_pos:
            self.current_subtask = self.env.process(self.moving_proc(from_pos))
            yield self.current_subtask

        # Pick up item
        if not self.picked_up_item:
            self.current_subtask = self.env.process(self.pick_up(item))
            yield self.current_subtask

        # Move to target destination
        self.current_subtask = self.env.process(self.moving_proc(to_pos))
        yield self.current_subtask

        # Store item at destination
        self.current_subtask = self.env.process(self.store_item())
        yield self.current_subtask

        self.unlock_item()

        self.current_task = None

        # Inform agents in own cell and new cell if item was put in an interface
        self.cell.inform_agents()
        if item.current_cell is not self.cell and item.current_cell:
            item.current_cell.inform_agents()

    def announce_arrival(self, order, destination):
        """Announce the arrival of an order to the target destination. The announcement will contain the agent, item and time

        :param order: (Order object) Item to be announced to the destination
        :param destination: (Buffer or Machine object) Destination where the item will arrive"""
        # Calculate arrival time
        arr_time = self.env.now + self.time_for_distance(order.position) + self.time_for_distance(destination, start_position=order.position) + self.time_for_item_pick_up + self.time_for_item_store

        # Send announcement
        destination.expected_orders.append((order, arr_time, self))

        # Inform cell if destination is an interface buffer
        if isinstance(destination, InterfaceBuffer):
            if destination.upper_cell == self.cell:
                destination.lower_cell.inform_incoming_order(self, order, arr_time, destination)

            elif destination.upper_cell is not None:
                destination.upper_cell.inform_incoming_order(self, order, arr_time, destination)

    def moving_proc(self, destination):
        """SUBTASK: Agent is moving to its target position.
        :param destination: (Machine or Buffer object) The target position"""

        if isinstance(destination, Machine) and self.picked_up_item:
            if self.picked_up_item.next_task != destination.performable_task:
                raise Exception("Warning: An agent tried to move an item to an machine that can not perform the next processing step of the item!")

        if not self.moving and self.position != destination:
            self.start_moving(destination)
        else:
            return

        # Perform moving (Wait remaining moving time and change status afterwards)
        yield self.env.timeout(self.remaining_moving_time)

        self.end_moving()
        self.current_subtask = None

    def pick_up(self, item):
        """SUBTASK: Pick up item from position if no other item is picked up by this agent
        :param item: (Order object) The item to pick up"""

        # Check for errors
        if isinstance(self.position, Machine):
            if self.position.item_in_output != item:
                raise Exception("Agent can not pick up the next item, because it is currently not in the machine output!")
        else:
            if not item in self.position.items_in_storage:
                raise Exception("Agent can not pick up the next item, because it is not in the storage slots!")
            elif item.current_cell is not self.cell:
                raise Exception("Agent can not pick up the next item, because it is not in the same cell as the agent!")

        if self.picked_up_item is None:
            self.save_event("pick_up_start")

            # Perform picking up
            yield self.env.timeout(self.time_for_item_pick_up)

            # State changes after pick up
            self.end_pick_up(item)

        # Inform other agents within this cell
        self.cell.inform_agents()
        self.current_subtask = None

    def store_item(self):
        """SUBTASK: Put down item at a machine or buffer and inform position."""
        if self.picked_up_item:
            item = self.picked_up_item
        else:
            raise Exception("Agent can not store an item because no item was picked up before!")

        full = False
        lock = False

        if isinstance(self.position, Buffer):
            if self.position.full:
                full = True
        else:
            if self.position.item_in_input or self.position.input_lock:
                full = True
            else:
                self.position.input_lock = True
                lock = True

        if full:
            self.current_waitingtask = self.env.process(self.wait_for_free_slot())
            yield self.current_waitingtask

            self.current_subtask = self.env.process(self.store_item())
            yield self.current_subtask
            return

        self.save_event("store_item_start")

        yield self.env.timeout(self.time_for_item_store)

        if lock:
            self.position.input_lock = False

        self.position.item_stored(item, self.cell)

        self.picked_up_item = None
        item.picked_up_by = None
        item.save_event("put_down")

        self.current_subtask = None
        self.save_event("store_item_end")

    def wait_for_free_slot(self):
        """SUBTASK: Endless loop. Wait for an item slot at current position to be free again.
        Interruption removes waiting agent from position. Loop can only be interrupted by simpy interruption"""
        try:
            self.position.waiting_agents.append(self)
            self.waiting = True
            self.save_event("wait_for_slot_start")

            while True:
                yield self.env.timeout(100000)

        except simpy.Interrupt as interruption:
            self.current_waitingtask = None
            self.waiting = False
            self.position.waiting_agents.remove(self)
            self.save_event("wait_for_slot_end")

    def unlock_item(self):
        """Release the locked item of the agent"""
        item = self.locked_item
        item.locked_by = None
        item.save_event("unlocked")
        self.locked_item = None

    def time_for_distance(self, destination, start_position=None):
        """Calculate the needed time for a given route

        :param destination: (Buffer or Machine object) Destination where the agend is heading to
        :param start_position: (Buffer or Machine object) Optional: Start position where the agent starts moving from. If None the agents position will be used
        :return time needed to move from start position to destination"""

        if not destination:
            raise Exception("Time for distance: Can not calculate the distance to destination None")

        def get_time(start_pos, end_pos):
            if start_pos == end_pos:
                return 0
            for start, end, length in self.cell.distances:
                if start == start_pos and end == end_pos:
                    return length / self.speed

        if not start_position:
            start_position = self.position

        if destination == start_position:
            return 0
        else:
            return get_time(start_position, destination)

    def state_change_in_cell(self):
        """The state of the agents cell has changed. If it has no current main process, start a new one"""
        if not self.main_proc.is_alive:
            self.main_proc = self.env.process(self.main_process())

    def start_moving(self, destination):
        """State changes for an agent that starts to move to an new destination.
        Sets the position of picked up items to None.
        :param destination: (Buffer or Machine object) The target position the agent is heading to"""

        self.moving = True
        self.moving_start_position = self.position
        self.moving_start_time = self.env.now
        self.next_position = destination
        self.moving_time = self.time_for_distance(destination)
        self.remaining_moving_time = self.moving_time
        self.moving_end_time = self.moving_start_time + self.moving_time

        if self.picked_up_item:
            self.picked_up_item.position = None
            self.picked_up_item.save_event("transportation_start")

        self.position = None

        self.save_event("moving_start", next_position=self.next_position, travel_time=self.remaining_moving_time)

    def end_moving(self):
        """State changes for an agent that reached its destination. Sets position for picked up items."""
        self.moving = False
        self.remaining_moving_time = 0
        self.moving_time = 0
        self.moving_end_time = None
        self.position = self.next_position
        self.next_position = None

        if self.picked_up_item:
            self.picked_up_item.position = self.position
            self.picked_up_item.save_event("transportation_end")

        self.save_event("moving_end")

    def end_pick_up(self, item):
        """State changes: Finished picking up an item
        :param item: (Order object) The picked up order"""
        self.picked_up_item = item

        item.picked_up_by = self
        item.position = None

        self.position.item_picked_up(item)

        item.save_event("picked_up")
        self.save_event("pick_up_end")

    def occupancy(self, attributes: list, requester=None):
        """State calculation for the agent. Gets agent attributes and picked up orders

        :param attributes: List of strings. Each element is an attribute that should be calculated and returned
        :param requester: (Agent object) Manufacturing agent that requests the state
        :return tuple of orders picked up and attributes of this agent. (list of dict, dict)"""

        if requester == self:
            pos_type = "Agent - Self"
        else:
            pos_type = "Agent"

        def agent_position():
            return self.position

        def moving():
            return int(self.moving)

        def remaining_moving_time():
            if self.moving:
                return self.moving_end_time - self.env.now
            else:
                return 0

        def next_position():
            if self.moving:
                return self.next_position
            else:
                return -1

        def has_task():
            return int(self.has_task)

        def locked_item():
            if self.locked_item:
                return self.locked_item
            else:
                return -1

        attr = {}
        for attribute in attributes:
            attr[attribute] = locals()[attribute]()

        if self.picked_up_item:
            return [{"order": self.picked_up_item, "pos": self, "pos_type": pos_type}], attr
        else:
            return [{"order": None, "pos": self, "pos_type": pos_type}], attr

    def save_event(self, event_type: str, next_position=None, travel_time=None):
        """Save an event to the event log database. Includes the current state of the object.

        :param event_type: (str) The title of the triggered event
        :param next_position: (Buffer object or Machine object) Only when a new moving process is started: The destination where the agent is going
        :param travel_time: (float) Only when a new moving process is started: The time needed to get to the next position"""

        if self.simulation_environment.train_model:
            return

        db = self.simulation_environment.db_con
        cursor = self.simulation_environment.db_cu

        time = self.env.now

        if next_position:
            nxt_pos = id(next_position)
        else:
            nxt_pos = None

        if self.position:
            pos = id(self.position)
        else:
            pos = None

        if self.picked_up_item:
            pui = id(self.picked_up_item)
        else:
            pui = None

        if self.locked_item:
            locki = id(self.locked_item)
        else:
            locki = None

        cursor.execute("INSERT INTO agent_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (id(self), time, event_type, nxt_pos, travel_time, self.moving, self.waiting, self.has_task, pos,
                        pui, locki))
        db.commit()

    def initial_event(self):
        """Add initial events to event log database. Necessary to calculate measures and results"""
        self.save_event("Initial")
        yield self.env.timeout(0)

    def end_event(self):
        """Add end events to event log database. Necessary to calculate measures and results"""
        self.save_event("End_of_Time")

    def state_to_numeric(self, order_state):
        """Util: Converts an categorical state into an numerical state. Fill all Nan values.
        :param order_state: Pandas Dataframe containing the categorical state
        :return numerical_order_state: Converted State"""
        
        # Add index column
        order_state.loc[:, "slot_id"] = order_state.index
        slot_ids = order_state.pop("slot_id")
        order_state.insert(0, "slot_id", slot_ids)
        
        a = order_state.loc[:, "pos_type"]
        
        # Get ids for all positions within the cell
        pos_in_cell = order_state["pos"].unique()
        pos_ids = np.arange(1, len(pos_in_cell) + 1)
        pos_ids = dict(zip(pos_in_cell, pos_ids))

        # Get ids for all position types
        pos_type_ids = dict_pos_types

        # Get all orders currently within this cell
        orders_in_cell = order_state[order_state["order"].notnull()]["order"].to_dict()
        orders_in_cell = {orders_in_cell[key]: key for key in orders_in_cell}

        # Map categorical values to ids
        cols = ["pos", "agent_position", "next_position", "_destination"]
        cols = [column for column in cols if column in order_state.columns.values.tolist()]
        order_state[cols] = order_state[cols].replace(pos_ids)
        
        cols = ["pos_type"]
        order_state[cols] = order_state[cols].replace(pos_type_ids)
        
        if "locked_item" in order_state.columns.values.tolist():
            order_state["locked_item"].fillna(-2)
            cols = ["locked_item"]
            order_state[cols] = order_state[cols].replace(orders_in_cell)
        
        order_state = order_state.fillna(0)
        now = time.time()
        order_state.loc[order_state["order"] != 0, "order"] = 1
        time_tracker.time_prob_2 += time.time() - now
        return order_state
