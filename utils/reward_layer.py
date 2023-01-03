from cmath import exp
from pickletools import read_int4
import pandas as pd
import numpy as np
import math
import time

def evaluate_choice(choice):
    """Take an chosen action and check if this action is valid or should not be performed to save the system stability.
    Return an high penalty if action is unvalid. Those actions will not be performed by the agent.
    :param choice: (pd.Series) Chosen row from numeric state
    :return penalty amount"""

    # Penalty criteria for forbidden choices
    criteria = [
        choice["order"] == 0,
        choice["locked"] == 2,
        choice["picked_up"] == 1,
        choice["in_m_input"] == 1,
        choice["in_m"] == 1,
        choice["in_same_cell"] == 0,
        choice["_destination"] == -2
    ]
    
    penalty = -200 * sum(criteria)
    return penalty

def reward_action(old_state, new_state, order, action):
    """Calculate the reward for an agent for the last action he performed
    :return The reward amount"""

    # base reward
    reward = 0

    old_pos_type = old_state[old_state["order"] == order]["pos_type"].iloc[0]

    # Penalty for blocked Input
    if not old_pos_type == "Input":
        input_full = not old_state[old_state["pos_type"] == "Input"]["free_slots"].iloc[0]
    else:
        input_full = False

    if new_state[new_state["order"] == order].empty:
        # A 2 was placed at the main output of the environment

        order_in_machine = False
        order_in_storage = False
        order_in_interface = False
        order_in_empty_interface = False
        storage_full_afterwards = False
        order_in_defective_machine = False
        machine_wrong_setup = False
        order_in_output = True
        next_task_in_cell = False
        order_completed = True

    else:
        # Normal reward calculation. Item is still in state
        new_pos_type = new_state[new_state["order"] == order]["pos_type"].iloc[0]
        new_pos = new_state[new_state["order"] == order]["pos"].iloc[0]

        # Reward for item put down in machine
        order_in_machine = new_pos_type == "Machine-Input"

        # Reward for item put down in storage
        order_in_storage = new_pos_type == "Storage"

        # Reward for item put down in interface buffer
        order_in_interface = new_pos_type == "Interface-In"

        # Reward if Interface was previously empty
        if order_in_interface:
            order_in_empty_interface = old_state[old_state["pos"] == new_pos]["order"].dropna().empty
        else:
            order_in_empty_interface = False

        # Small Penalty if storage is full afterwards
        if order_in_storage:
            storage_full_afterwards = not new_state[new_state["order"] == order]["free_slots"].iloc[0]
        else:
            storage_full_afterwards = False

        # Penalty for item put down in defect machine or machine in wrong setup
        if order_in_machine:
            order_in_defective_machine = new_state[new_state["pos"] == new_pos]["failure"].iloc[0]
            machine_wrong_setup = order.type.type_id != new_state[new_state["pos"] == new_pos]["current_setup"].iloc[0]
        else:
            order_in_defective_machine = False
            machine_wrong_setup = False

        # Reward for item put down in Cell output
        order_in_output = new_pos_type == "Output"

        # Penalty for item in output if next task could have been processed within this cell, reward for finished item without next tasks
        if order_in_output:
            next_task = new_state[new_state["order"] == order]["next_task"].iloc[0]
            if "machine_type" in new_state.columns:
                next_task_in_cell = not new_state[new_state["machine_type"] == next_task].empty
            else:
                next_task_in_cell = False
            order_completed = new_state[new_state["order"] == order]["tasks_finished"].iloc[0]
        else:
            next_task_in_cell = False
            order_completed = False

    # Reward/Penalty amount
    reward_settings = [(input_full, -50),
                       (order_in_machine, 100),
                       (order_in_storage, 20),
                       (order_in_interface, 50),
                       (order_in_empty_interface, 40),
                       (storage_full_afterwards, -10),
                       (order_in_defective_machine, -20),
                       (machine_wrong_setup, -10),
                       (order_in_output, 100),
                       (next_task_in_cell, -50),
                       (order_completed, 50)]

    reward += sum([value for condition, value in reward_settings if condition])
    
    return reward

def reward_smart_dispatch(old_state, new_state, order, action):
    """Calculate the reward for an agent for the last action he performed
    :return The reward amount"""
    reward_due_to, reward_basic, reward_throughput_time_local = 0, 0, 0
    reward_due_to = calc_reward_due_to(old_state, action)
    reward_priority = calc_reward_priority(old_state, action) # 0 or 1; 1 if order has high priority
    reward_throughput_time_local = calc_reward_throughput_time_local(old_state, action)
    return reward_due_to + reward_priority + reward_throughput_time_local
    # if reward_due_to > 0:
    #     return reward_due_to * math.exp(reward_priority/3) + reward_basic
    # else:  
    #     return reward_due_to * math.exp(-reward_priority/3) + reward_basic
    

def reward_heuristic(old_state, new_state, order, action):
    reward_due_to, reward_basic = 0, 0
    reward_due_to = calc_reward_due_to(old_state, action) # -300 or 400; if order had lower due_to in average
    reward_priority = calc_reward_priority(old_state, action) # 0 or 1; 1 if order has high priority
    reward_throughput_time_local = calc_reward_throughput_time_local(old_state, action)
    # reward_basic = calc_reward_basic(old_state, new_state, order)
    return reward_due_to + reward_priority + reward_throughput_time_local
    # if reward_due_to > 0:
    #     return reward_due_to * math.exp(reward_priority/2) + reward_basic
    # else:  
    #     return reward_due_to * math.exp(-reward_priority/2) + reward_basic


def calc_reward_due_to(old_state, action):
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
    min_due_to, max_due_to = np.min(due_to_values), np.max(due_to_values)
    try:
        due_to = old_state.loc[action, "due_to"].values[0]
    except AttributeError:
        due_to = old_state.loc[action, "due_to"]
    reward_due_to = (2*(max_due_to-due_to)/(max_due_to-min_due_to) - 1)**5 * 150
    return reward_due_to

def calc_reward_due_to_old(old_state, action):
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

    try:
        due_to = old_state.loc[action, "due_to"].values[0]
    except AttributeError:
        due_to = old_state.loc[action, "due_to"]

    relevant_due_tos = [x for x in due_to_values if np.isnan(x) == False]

    try:
        if len(relevant_due_tos)>1:
            if due_to <= np.mean(relevant_due_tos):
                reward_due_to = 400
            else:
                reward_due_to = -400
        else: 
            reward_due_to = 0
    except:
        reward_due_to = 0
    return reward_due_to

def calc_reward_priority(old_state, action): #get priority indicators for all orders that have a destination; 0 = normal priority; 1 = high priority
    old_cell_priorities = old_state.loc[:, "priority"]
    reward_priority, reward_priority_2 = 0, 0
    if old_cell_priorities[action].values[0] == 0:
        reward_priority = 0
    elif old_cell_priorities[action].values[0] == 1:
        reward_priority = 200
    elif old_cell_priorities[action].values[0] == 2:
        reward_priority = 700

    count_prio_2 = 0
    for i in range(len(old_cell_priorities)):
        try:
            if old_cell_priorities[i].values[0] == 2:
                count_prio_2 += 1    
        except AttributeError:
            pass
        
    if count_prio_2 >= 1 and old_cell_priorities[action].values[0] != 2:
        reward_priority_2 = -400

    return reward_priority + reward_priority_2

def calc_reward_throughput_time_local(old_state, action):
    time_in_cell = old_state.loc[:, "time_in_cell"]
    #get due_to values for all orders that have a destination
    destination = old_state.loc[:, "_destination"]
    available_destinations = []
    for i in range(len(destination)): #(2) look for orders on valid places, if not valid; nan
        if destination[i] == -1:
            available_destinations.append(np.nan)
        else:
            available_destinations.append(1)
    time_in_cell_available = np.multiply(time_in_cell, available_destinations) #Time in cell of orders that can be processed by an agent

    time_in_cell_min, time_in_cell_max = np.min(time_in_cell_available), np.max(time_in_cell_available)
    try:
        time_in_cell_order = old_state.loc[action, "time_in_cell"].values[0]
    except AttributeError:
        time_in_cell_order = old_state.loc[action, "time_in_cell"]

    reward_throughput_time = (1 - 2*(time_in_cell_max-time_in_cell_order)/(time_in_cell_max-time_in_cell_min))**5 * 100 #Highest time in cell to awarded, lowest to be punished
    return reward_throughput_time



def calc_reward_basic(old_state, new_state, order):
    old_pos_type = old_state[old_state["order"] == order]["pos_type"].iloc[0]

    # Penalty for blocked Input
    if not old_pos_type == "Input":
        input_full = not old_state[old_state["pos_type"] == "Input"]["free_slots"].iloc[0]
    else:
        input_full = False

    if new_state[new_state["order"] == order].empty:
        # A 2 was placed at the main output of the environment
        order_in_machine = False
        order_in_storage = False
        order_in_interface = False
        order_in_empty_interface = False
        storage_full_afterwards = False
        order_in_defective_machine = False
        machine_wrong_setup = False
        order_in_output = True
        next_task_in_cell = False
        order_completed = True

    else:
        # Normal reward calculation. Item is still in state
        new_pos_type = new_state[new_state["order"] == order]["pos_type"].iloc[0]
        new_pos = new_state[new_state["order"] == order]["pos"].iloc[0]

        # Reward for item put down in machine
        order_in_machine = new_pos_type == "Machine-Input"

        # Reward for item put down in storage
        order_in_storage = new_pos_type == "Storage"

        # Reward for item put down in interface buffer
        order_in_interface = new_pos_type == "Interface-In"

        # Reward if Interface was previously empty
        if order_in_interface:
            order_in_empty_interface = old_state[old_state["pos"] == new_pos]["order"].dropna().empty
        else:
            order_in_empty_interface = False

        # Small Penalty if storage is full afterwards
        if order_in_storage:
            storage_full_afterwards = not new_state[new_state["order"] == order]["free_slots"].iloc[0]
        else:
            storage_full_afterwards = False

        # Penalty for item put down in defect machine or machine in wrong setup
        if order_in_machine:
            order_in_defective_machine = new_state[new_state["pos"] == new_pos]["failure"].iloc[0]
            machine_wrong_setup = order.type.type_id != new_state[new_state["pos"] == new_pos]["current_setup"].iloc[0]
        else:
            order_in_defective_machine = False
            machine_wrong_setup = False

        # Reward for item put down in Cell output
        order_in_output = new_pos_type == "Output"

        # Penalty for item in output if next task could have been processed within this cell, reward for finished item without next tasks
        if order_in_output:
            next_task = new_state[new_state["order"] == order]["next_task"].iloc[0]
            if "machine_type" in new_state.columns:
                next_task_in_cell = not new_state[new_state["machine_type"] == next_task].empty
            else:
                next_task_in_cell = False
            order_completed = new_state[new_state["order"] == order]["tasks_finished"].iloc[0]
        else:
            next_task_in_cell = False
            order_completed = False

    # Reward/Penalty amount
    reward_settings = [(input_full, -50),
                       (order_in_machine, 100),
                       (order_in_storage, 20),
                       (order_in_interface, 50),
                       (order_in_empty_interface, 40),
                       (storage_full_afterwards, -10),
                       (order_in_defective_machine, -20),
                       (machine_wrong_setup, -10),
                       (order_in_output, 200),
                       (next_task_in_cell, -50),
                       (order_completed, 50)]

    return sum([value for condition, value in reward_settings if condition])    