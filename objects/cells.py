from objects.manufacturing_agents import ManufacturingAgent, set_agent_seed
from objects.buffer import Buffer, InterfaceBuffer, QueueBuffer
from objects.processing_steps import ProcessingStep
from objects.order import get_order_attributes
from objects.machines import Machine
import csv
import simpy
import pandas as pd
from configs import state_attributes

from utils import time_tracker
import time


class Cell:
    instances = []

    def __init__(self, env: simpy.Environment, agents: list, storage: QueueBuffer, input_buffer: InterfaceBuffer,
                 output_buffer: InterfaceBuffer, level, cell_id, cell_type):
        self.env = env
        self.simulation_environment = None

        # Attributes
        self.id = cell_id
        self.type = cell_type
        self.parent = None  # Parent cell of this cell, None if cell is main cell
        self.level = level  # Hierarchy level of this cell, counting bottom up
        self.height = None  # Physical distance between top and bottom of the cell, used for distance calculations
        self.width = None  # Physical distance between left and right side of the cell, used for distance calculations
        self.distances = []  # List containing all possible paths within the cell with shortest length. Agent always use the shortest path to its destination
        self.agents = agents  # Agents within the cell
        for agent in agents:
            agent.cell = self
        self.input_buffer = input_buffer  # Input buffer of the cell, Interface with parent cell.
        self.output_buffer = output_buffer  # Output buffer of the cell, Interface with parent cell.
        self.storage = storage  # Storage buffer of the cell. Only one Storage per cell.
        self.possible_positions = [input_buffer, output_buffer, storage]  # All possible positions for agents within this cell
        self.cell_capacity = sum([pos.storage_capacity for pos in self.possible_positions]) + len(self.machines) * 3 + len(self.agents)  # Maximum amount of items that can be stored within the cell
        self.performable_tasks = []  # Amount of machines in this cell or its childs for each processing step. Used to determine if orders can be completly processed in this tree branch

        # State
        self.orders_in_cell = []  # Items currently located within this cell
        self.expected_orders = []  # Announced Orders, that will be available within this cell within next time (Order, Time, Position, Agent)

        self.__class__.instances.append(self)
        self.result = None

    def orders_available(self):
        """Check if there are non locked items available within this cell
        :return orders_available: Boolean"""
        non_locked = [order for order in self.orders_in_cell if not order.locked_by or order.processing]

        if len(non_locked) > 0:
            return True
        return False

    def inform_incoming_order(self, agent, item, time, position):
        """Cell gets announcement that a new item will arrive in this cell
        :param agent: (Manufacturing agent object) Informing agent
        :param item: (Order object) item that will be arrive
        :param time: (float) time when the item will approximately arrive
        :param position: (Interfacebuffer object) Position where the item will arrive"""

        self.expected_orders.append((item, time, position, agent))

    def cancel_incoming_order(self, order_cancel):
        """Cancel an existing announcement for an arriving item
        :param order_cancel: (Order object) Item to remove from expected arriving items"""

        if self.expected_orders:
            for item in self.expected_orders:
                order, time, position, agent = item
                if order == order_cancel:
                    self.expected_orders.remove(item)
                    return

    def all_tasks_included(self, order, all_tasks=True, alternative_tasks=None):
        """Test if all tasks within the orders work schedule can be performed by this cell.
        Alternative list of tasks is possible.

        :param order: (Order object) order to check
        :param all_tasks: (boolean) Should the complete work schedule be checked or only the remaining?
        :param alternative_tasks: list of tasks, alternative work schedule to check
        :return all_tasks_included: boolean"""

        if alternative_tasks:
            performable_tasks = alternative_tasks
        else:
            performable_tasks = self.performable_tasks

        if all_tasks:
            work_schedule = order.work_schedule
        else:
            work_schedule = order.remaining_tasks

        for task in work_schedule:
            task_possible = False
            for (perform_task, machines) in performable_tasks:
                if task == perform_task and machines > 0:
                    task_possible = True

            if not task_possible:
                return 0
        return 1

    def occupancy(self, requester: ManufacturingAgent, criteria: dict):
        """Get cell state by merging the states of all objects within the cell.

        :param requester: (Agent object) state requesting agents
        :param criteria: dictionary containing the criterias to be included and the context
        :return cell_state: categorical cell state"""

        # Get states and occupancy from all objects within the cell
        buffer = [self.input_buffer.occupancy("Input", criteria["buffer"], self)] + [self.output_buffer.occupancy("Output", criteria["buffer"], self)]

        storage = [self.storage.occupancy("Storage", criteria["buffer"], self)]

        agents = [agent.occupancy(criteria["agent"], requester=requester) for agent in self.agents]

        machines = [machine.occupancy(criteria["machine"]) for machine in self.machines]

        interfaces_in = [interface.occupancy("Interface-In", criteria["buffer"]) for interface in self.interfaces_in]
        interfaces_out = [interface.occupancy("Interface-Out", criteria["buffer"]) for interface in self.interfaces_out]

        # Merge states
        result = buffer + storage + agents + machines + interfaces_in + interfaces_out

        return [{**item, **pos_attr} for sublist, pos_attr in result for item in sublist]

    def get_cell_state(self, requester: ManufacturingAgent):
        """Get cell state. Calculate the needed criteria included in the state. Add order attributes.
        :param requester: (Agent object) state requesting agent
        :return cell_state: complete categorical cell state containing state and order attributes"""

        if requester.ruleset.dynamic:
            criteria = state_attributes.smart_state
            ranking_criteria = []
        elif requester.ruleset.dynamic_dispatch:
            criteria = state_attributes.smart_state
            ranking_criteria = []
        else:
            criteria = state_attributes.normal_state
            ranking_criteria = requester.ranking_criteria
        
        # Get occupancy of all available slots within this cell
        now = time.time()
        occupancy_states = pd.DataFrame(self.occupancy(requester, criteria))
        time_tracker.time_occupancy_calc += time.time() - now

        # Add attributes for each order within this cell
        now = time.time()
        occupancy_states = self.add_order_attributes(occupancy_states, requester, criteria["order"] + list(set(ranking_criteria) - set(criteria["order"])))
        time_tracker.time_order_attr_calc += time.time() - now
        return occupancy_states

    def add_order_attributes(self, occupancy, requester: ManufacturingAgent, attributes: list):
        """Get order attributes for occupancy state and join them

        :param occupancy: Dataframe containing occupancies and position attributes
        :param requester: (Agent object) state requesting agent
        :param attributes: list of strings. Each string represents an attribute to be added to the state
        :return merged_state: Dataframe"""

        current_time = self.env.now
        occupancy["attributes"] = occupancy["order"].apply(get_order_attributes, args=(requester, attributes, current_time))
        occupancy = occupancy.join(pd.DataFrame(occupancy.pop("attributes").values.tolist()))
        return occupancy

    def inform_agents(self):
        """Inform all agents that the cell states have changed. Idling agent will check for new tasks"""
        for agent in self.agents:
            agent.state_change_in_cell()

    def new_order_in_cell(self, order):
        """Arrival of a new item in this cell. Change attributes.
        :param order: (Order object) The arrived item"""
        order.current_cell = self
        order.in_cell_since = self.env.now
        self.cancel_incoming_order(order)
        self.orders_in_cell.append(order)

    def remove_order_in_cell(self, order):
        """Removal of an item from this cell. Change attributes.
        :param order: (Order object) The item, that has leaved the cell"""
        order.in_cell_since = None
        self.orders_in_cell.remove(order)


class ManufacturingCell(Cell):

    def __init__(self, machines: list, *args):
        self.machines = machines  # list of all machines within the cell
        self.interfaces_in = []
        self.interfaces_out = []

        super().__init__(*args)

        self.possible_positions += machines

    def init_performable_tasks(self):
        """Initialize performable tasks of cell:
        Which tasks can be performed within this cell and how many machines are there for each?"""
        result = []

        # Check how many machines there are for each task
        for task in ProcessingStep.instances:
            machine_counter = len([machine for machine in self.machines if machine.performable_task == task])
            result.append((task, machine_counter))

        self.performable_tasks = result


class DistributionCell(Cell):

    def __init__(self, childs: list, *args):
        self.childs = childs  # Child cells
        self.machines = []
        self.interfaces_in = [child.input_buffer for child in childs]  # Interfaces to cells of lower hierarchy level
        self.interfaces_out = [child.output_buffer for child in childs]  # Interfaces to cells of lower hierarchy level

        super().__init__(*args)

        self.possible_positions += self.interfaces_in
        self.possible_positions += self.interfaces_out
        self.cell_capacity += sum([inpt.storage_capacity for inpt in self.interfaces_in]) + sum([outpt.storage_capacity for outpt in self.interfaces_out])

    def init_performable_tasks(self):
        """Initialize self.PERFORMABLE_TASKS:
        Which tasks can be performed within this cell and how many machines are there for each?
        Iterate through complete tree branch"""
        child_tasks = []

        for child in self.childs:
            if len(child.performable_tasks) == 0:
                child.init_performable_tasks()
            child_tasks.append(child.performable_tasks)

        self.performable_tasks = combine_performable_tasks(child_tasks)


def combine_performable_tasks(task_list):
    """Util function to flatten multidimensional lists into one flat list
    with the amount of appearences within the list

    :param task_list: (list) Multidimensional list of all child cells performable tasks.
    :return flatten_list: List containing the amount of machines of each type within cells child cells.
    """

    result = []
    flatten_list = []

    for child_cell in task_list:
        flatten_list += child_cell

    for task in ProcessingStep.instances:
        number_of_machines = 0
        for list_element in flatten_list:
            task_type, machines = list_element
            if task_type == task:
                number_of_machines += machines
        result.append((task, number_of_machines))

    return result