# -*- coding: utf-8 -*-
import json
from objects.processing_steps import load_processing_steps, ProcessingStep
import simpy
import time
import math
import random
from copy import copy
import numpy as np
from objects.machines import Machine


class Order:
    instances = []
    finished_instances = []

    def __init__(self, env: simpy.Environment, sim_env, start, due_to,
                 type, complexity=1, priority=1):
        self.env = env
        self.simulation_environment = sim_env

        # Attributes
        self.type = type  # Type of order. New Types can be defined in Order_types.json.
        self.composition = self.type.composition  # Material composition of the order. Defined by order type.
        self.work_schedule = copy(self.type.work_schedule)  # The whole processing steps to be performed on this item to be completed
        self.start = start  # Time when the order arrived/will arrive
        self.starting_position = sim_env.main_cell.input_buffer  # The position where the item will spawn once it started
        self.due_to = due_to  # Due to date of the order
        self.complexity = complexity  # Numerical value, modifier for processing time within machines
        self.priority = priority # Numerical value, modifier for priority of orders
        # State
        self.started = False  # Order has arrived
        self.overdue = False  # Order is overdue
        self.tasks_finished = False  # All tasks of the workschedule are finished
        self.processing = False  # Is currently beeing processed in a machine
        self.wait_for_repair = False  # Is in a machine waiting for the machine to be fixed
        self.completed = False  # Order is completed (Tasks finished and put down in main output of the environment)
        self.completed_at = None  # Time when the order was completed
        self.remaining_tasks = copy(self.work_schedule)  # Remaining tasks to be performed for the order to be completed
        self.next_task = self.remaining_tasks[0]  # The next task that has to be performed on this item
        self.position = None  # Current position within the environment
        self.current_cell = None  # Cell the item is currently in
        self.in_cell_since = None  # Time when the item entered its current cell over the interface buffer
        self.picked_up_by = None  # The agent which picked up the item
        self.blocked_by = None  # Other order that might block the further processing of this order
        self.locked_by = None  # Locked by Agent X. A locked Order can´t be part of other agent tasks

        self.__class__.instances.append(self)
        self.result = None

        self.env.process(self.set_order_overdue())

    def order_finished(self):
        """Event: Order is at output buffer of the main cell.
        Remove it from the simulation environment if it is finished"""

        if self.position == self.simulation_environment.main_cell.output_buffer and self.tasks_finished:
            self.position.items_in_storage.remove(self)
            self.current_cell = None
            self.position = None
            self.completed = True
            self.completed_at = self.env.now
            self.__class__.finished_instances.append(self)
            if (len(self.__class__.finished_instances) % 10) == 0:
                print("Order finished! Nr ", len(self.__class__.finished_instances))

    def processing_step_finished(self):
        """Event: One processing step of the order was finished. Get next and check if order tasks are completed."""
        if len(self.remaining_tasks) == 1:
            # Tasks of this order where finished
            del self.remaining_tasks[0]
            self.next_task = ProcessingStep.dummy_processing_step
            self.tasks_finished = True
        else:
            # Get next task
            del self.remaining_tasks[0]
            self.next_task = self.remaining_tasks[0]

    def order_arrival(self):
        """A new order arrives in the simulation environment. Check wether the start position is full or not.
        If full put this order on a waiting queue"""

        if len(self.starting_position.items_in_storage) < self.starting_position.storage_capacity:
            # Order arrives
            self.position = self.starting_position
            self.started = True
            self.position.items_in_storage.append(self)

            if len(self.position.items_in_storage) == self.position.storage_capacity:
                self.position.full = True

            # print(round(self.env.now, 2), "Arrival of new item")

            self.simulation_environment.main_cell.new_order_in_cell(self)
            self.simulation_environment.main_cell.inform_agents()

            self.save_event("order_arrival")
            self.position.save_event("order_arrival")
        else:
            # Start position full
            self.starting_position.items_waiting.append((self, self.env.now))
            self.save_event("incoming_order")

    def set_order_overdue(self):
        """Set order overdue if it wasn´t finished in time. Started when order arrives in the simulation"""
        yield self.env.timeout(self.due_to - self.start)
        if not self.completed:
            self.overdue = True
            self.save_event("over_due")

    def machine_failure(self, new):
        """State change: Machine where the order is a has an failure or an exiting one was repaired
        :param new: (bool) Has the failure just started or was it repaired?"""

        if new:
            self.wait_for_repair = True
            self.save_event("machine_failure")
        else:
            self.wait_for_repair = False

    def save_event(self, event_type: str):
        """Save an event to the event log database. Includes the current state of the object.

        :param event_type: (str) The title of the triggered event"""

        if self.simulation_environment.train_model:
            return

        db = self.simulation_environment.db_con
        cursor = self.simulation_environment.db_cu

        time = self.env.now

        if self.blocked_by:
            blocked = True
        else:
            blocked = False

        if self.picked_up_by:
            picked_up = True
            picked_by = id(self.picked_up_by)
            transportation = self.picked_up_by.moving
        else:
            picked_up = False
            picked_by = None
            transportation = False

        if self.current_cell:
            cell = id(self.current_cell)
        else:
            cell = None

        if self.position:
            pos = id(self.position)
            pos_type = type(self.position).__name__
        else:
            pos = None
            pos_type = None

        if self.locked_by:
            lock_by = id(self.locked_by)
        else:
            lock_by = None

        tasks_remaining = len(self.remaining_tasks)

        cursor.execute("INSERT INTO item_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (id(self), time, event_type, self.started, self.overdue, blocked, self.tasks_finished,
                        self.completed, picked_up, transportation, self.processing, self.wait_for_repair, tasks_remaining,
                        cell, pos, str(pos_type), picked_by, lock_by))
        db.commit()

    def end_event(self):
        """Add end events to event log database if order was
        not completed. Necessary to calculate measures and results"""
        if not self.completed:
            self.save_event("End_of_Time")


class OrderType:
    instances = []

    def __init__(self, type_config: dict):
        self.__class__.instances.append(self)
        self.instance = len(self.__class__.instances)
        self.name = type_config['title'].encode()
        self.type_id = type_config['id']
        self.frequency_factor = type_config['frequency_factor']  # Factor to change the chance of appearence
        self.duration_factor = type_config['duration_factor']  # Factor to change the base order duration
        self.composition = type_config['composition']  # Material composition of an item of this type
        self.work_schedule = type_config['work_schedule']  # Tasks to be done to finish an order of this type
        for processing_step in ProcessingStep.instances:
            self.work_schedule = [processing_step if x == processing_step.id else x for x in self.work_schedule]

    def __eq__(self, other):
        if other:
            return self.instance == other.instance
        else:
            return False

    def __lt__(self, other):
        return self.instance < other.instance


def load_order_types():
    """
    Create instances for order types from order types config json
    """
    load_processing_steps()
    order_types = json.load(open("../configs/order_types.json", encoding="UTF-8"))

    for o_type in order_types['order_types']:
        OrderType(o_type)


def order_arrivals(env: simpy.Environment, sim_env, config: dict):
    """
    Create incoming order events for the simulation environment
    :param env: SimPy environment
    :param sim_env: Object of class simulation environment
    :param config: Configuration with Parameter like number of orders, order length
    """
    last_arrival = 0
    max_orders = config['NUMBER_OF_ORDERS']
    seed = config["SEED_INCOMING_ORDERS"]

    list_of_orders = get_orders_from_seed(max_orders, seed, config)
    sorted_list = list_of_orders[np.argsort(list_of_orders[:], order=["start"])]
    #sorted_list = list_of_orders[np.argsort(list_of_orders[:], order=["start", "due_to"])]

    for order in sorted_list:
        yield env.timeout(order['start'] - last_arrival)
        new_order = Order(env, sim_env, env.now, order['due_to'], order['type'], complexity=order['complexity'], priority=order['priority'])
        new_order.order_arrival()
        last_arrival = env.now

def get_orders_from_seed(amount: int, seed: int, config: dict):
    """Create a list of random order attributes from seed
a
    :param amount: (int) The amount of orders to generate
    :param seed: (int) Seed for order arrivals and random attributes
    :param config: (dict) Main configuration dictionary to get setting for generation of orders
    :return order_records: (numpy records) All generated orders with attributes
    """
    np.random.seed(seed)

    possible_types = OrderType.instances
    frequency_factors = [order_type.frequency_factor for order_type in possible_types]
    factors_sum = sum(frequency_factors)
    frequency_factors = [factor/factors_sum for factor in frequency_factors]
    # Create attributes
    start_times_1 = np.random.uniform(low=0, high=config['SIMULATION_RANGE'], size=amount)
    start_times = np.arange(0, config['SIMULATION_RANGE'], config['SIMULATION_RANGE']/config['NUMBER_OF_ORDERS'])

    types = np.random.choice(possible_types, amount, p=frequency_factors,  replace=True)
   
    types_test_base = [possible_types[0], possible_types[1], possible_types[2], possible_types[0], possible_types[1],  possible_types[3]]
    types_test = []
    i = 0
    while i < (math.floor(amount/len(types_test_base))):
        types_test = types_test + types_test_base
        i += 1
    m = len(types_test)
    while m < amount:
        add_order = [possible_types[random.randint(0, len(possible_types)-1)]]
        types_test = types_test + add_order
        m += 1
    types = types_test

    # maximum = count = 0
    # current = ''
    # for c in types:
    #     if c == current:
    #         count += 1
    #     else:
    #         count = 1
    #         current = c
    #     maximum = max(count,maximum)
    # print("maximum: ", maximum)

    duration_factors = np.asarray([order_type.duration_factor for order_type in types])
    base_lengths_1 = np.random.randint(low=config['ORDER_MINIMAL_LENGTH'], high=config['ORDER_MAXIMAL_LENGTH'], size=amount)

    # Calculate order prioritities
    base_lengths = np.full(amount, 1)
    for base_length in range(len(base_lengths)):
        lenght = random.randint(0,99)
        if lenght < 20:
            base_lengths[base_length] = 0
        elif lenght >= 20 and lenght < 50:
            base_lengths[base_length] = 40
        else:
            base_lengths[base_length] = 70

    complexities_1 = np.random.normal(loc=1, scale=config['SPREAD_ORDER_COMPLEXITY'], size=amount)
    complexities = np.full(amount, 1)

    # Check if random complexities are greater than 0
    for complexity in complexities:
        comp_value = complexity
        while comp_value <= 0:
            comp_value = np.random.normal(loc=1, scale=config['SPREAD_ORDER_COMPLEXITY'], size=1)
        complexities[complexities == complexity] = comp_value
    
    # Calculate order prioritities
    priorities = np.full(amount, 1)
    
    for priority in range(len(priorities)):
        prio = random.randint(0,99)
        if prio < 10:
            priorities[priority] = 2
        elif prio >= 10 and prio < 25:
            priorities[priority] = 1
        else:
            priorities[priority] = 0

    # Calculate order due_tue dates
    due_tues = start_times + base_lengths * duration_factors
    
    order_records = np.rec.fromarrays((start_times, due_tues, complexities, priorities, types), names=('start', 'due_to', 'complexity','priority', 'type'))

    return order_records


def get_order_attributes(order, requester, attributes: list, now):
    """Gets order attributes for states

    :param order
    :param requester: (Agent object) Manufacturing agent that requests the state
    :param attributes: List of strings. Each element is an attribute that should be calculated and returned
    :return dict of order attributes defined in attributes argument"""

    def start():
        return now - order.start

    def due_to():
        return order.due_to - now

    def complexity():
        return order.complexity

    def priority():
        return order.priority

    def type():
        return order.type.type_id

    def time_in_cell():
        return now - order.in_cell_since

    def locked():
        if not order.locked_by:
            return 0
        elif order.locked_by == requester:
            return 1
        else:
            return 2

    def picked_up():
        if not order.picked_up_by:
            return 0
        elif order.picked_up_by == requester:
            return 1
        else:
            return 2

    def processing():
        return int(order.processing)

    def tasks_finished():
        return int(order.tasks_finished)

    def remaining_tasks():
        return len(order.remaining_tasks)

    def next_task():
        return order.next_task.id

    def distance():
        if order.position:
            return requester.time_for_distance(order.position)
        else:
            return -1

    def in_m():
        if isinstance(order.position, Machine):
            if order == order.position.item_in_machine:
                return 1
            else:
                return 0
        else:
            return 0

    def in_m_input():
        if isinstance(order.position, Machine):
            if order == order.position.item_in_input:
                return 1
            else:
                return 0
        else:
            return 0

    def in_same_cell():
        if order.current_cell == requester.cell:
            return 1
        else:
            return 0

    attr = {}

    if order:
        for attribute in attributes:
            attr[attribute] = locals()[attribute]()
    return attr
