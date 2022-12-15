import simpy


class Buffer:
    instances = []

    def __init__(self, env: simpy.Environment, size: int):
        self.env = env
        self.simulation_environment = None
        self.cell = None
        self.coordinates = None  # Used only for initialization of distances within its cell

        # Attributes
        self.storage_capacity = size

        # State
        self.items_in_storage = []  # Currently stored items
        self.full = False
        self.items_waiting = []  # Queue of waiting order to be started (Only if buffer is main input interface and full)
        self.waiting_agents = []  # Agents waiting for a free slot at the position
        self.expected_orders = []  # Announced order to arrive soon: List of tuples (order, time, agent)

        self.__class__.instances.append(self)
        self.result = None

        self.env.process(self.initial_event())

    def free_slots(self):
        """Check if the buffer has free storage slots. Include announced items as already blocked slots.
        :return free_slots (boolean)"""

        return self.storage_capacity > len(self.items_in_storage) - len(
            [order for order in self.items_in_storage if order.locked_by]) + len(
            self.expected_orders)

    def item_picked_up(self, item):
        """State change: An item was picked up from buffer
        :param item (Order object): picked up item"""

        self.items_in_storage.remove(item)

        if self.items_waiting:
            self.items_waiting = sorted(self.items_waiting, key=lambda tup: tup[1])
            self.items_waiting[0][0].order_arrival()
            # print("len(self.items_waiting): ", len(self.items_waiting))
            del self.items_waiting[0]
        else:
            self.full = False
            if len(self.waiting_agents) > 0:
                self.waiting_agents[0].current_waitingtask.interrupt("New space free")

        self.save_event("item_picked_up", item)

    def item_stored(self, item, cell):
        """State change: An item was stored. Perform switch between two cell if buffer is an interface.
        :param item: (Order object) stored item
        :param cell: (Cell object) current cell of the order"""

        # Remove announcement
        self.expected_orders.remove(
            [(order, time, agent) for order, time, agent in self.expected_orders if order == item][0])

        # Add item
        self.items_in_storage.append(item)

        # Check if buffer full
        if len(self.items_in_storage) >= self.storage_capacity:
            self.full = True

        # If Interface: Handing over item to new cell
        if isinstance(self, InterfaceBuffer):
            item.save_event("cell_change")
            cell.remove_order_in_cell(item)

            if self.upper_cell == cell:
                next_cell = self.lower_cell
                next_cell.new_order_in_cell(item)

            elif self.upper_cell is not None:
                next_cell = self.upper_cell
                item.current_cell = next_cell
                next_cell.new_order_in_cell(item)

            # Item is at main interface out
            elif not self.upper_cell:
                item.order_finished()
                cell.inform_agents()

        self.save_event("item_stored", item)

    def occupancy(self, pos_type: str, attributes: list, cell=None):
        """State calculation for the buffer. Gets buffer attributes and items in this buffer

        :param pos_type: (string) The type of buffer (Input, Output, Storage, Interface...)
        :param attributes: List of strings. Each element is an attribute that should be calculated and returned
        :param cell: (cell object) The cell which requests the state. Has to be used if buffer is a interface buffer
        :return tuple of orders within the buffer and buffer attributes. (list of dict, dict)"""

        def interface_outgoing():
            if self.lower_cell == cell:
                # Input/Output of Cell
                if self == cell.input_buffer:
                    return 0
                else:
                    return 1
            elif self.upper_cell == cell:
                if self == self.lower_cell.input_buffer:
                    return 1
                else:
                    return 0

        def interface_ingoing():
            if self.lower_cell == cell:
                # Input/Output of Cell
                if self == cell.input_buffer:
                    return 1
                else:
                    return 0
            elif self.upper_cell == cell:
                if self == self.lower_cell.input_buffer:
                    return 0
                else:
                    return 1

        def free_slots():
            return self.free_slots()

        attr = {}

        # Get values for all attributes in attribute list
        if isinstance(self, InterfaceBuffer):
            for attribute in attributes:
                attr[attribute] = locals()[attribute]()

        return ([{"order": item, "pos": self, "pos_type": pos_type} for item in self.items_in_storage] \
                + [{"order": None, "pos": self, "pos_type": pos_type}] * (self.storage_capacity - len(self.items_in_storage)), attr)

    def save_event(self, event_type: str, item=None):
        """Save an event to the event log database. Includes the current state of the buffer.

        :param event_type (str): The title of the triggered event
        :param item (Order object): Context of the event: Stored item, picked up item, ..."""

        if self.simulation_environment.train_model:
            return

        db = self.simulation_environment.db_con
        cursor = self.simulation_environment.db_cu

        time = self.env.now

        if item:
            item = id(item)

        cursor.execute("INSERT INTO buffer_events VALUES(?,?,?,?,?,?)",
                       (id(self), time, event_type, item, self.full, len(self.items_in_storage)))
        db.commit()

    def end_event(self):
        """Add end events to event log database. Necessary to calculate measures and results"""
        self.save_event("End_of_Time")

    def initial_event(self):
        """Add initial events to event log database. Necessary to calculate measures and results"""
        self.save_event("Initial")
        yield self.env.timeout(0)


class QueueBuffer(Buffer):

    def __init__(self, env: simpy.Environment, size: int):
        super().__init__(env, size)


class InterfaceBuffer(Buffer):

    def __init__(self, env: simpy.Environment, size: int, lower_cell=None, upper_cell=None):
        self.lower_cell = lower_cell
        self.upper_cell = upper_cell
        super().__init__(env, size)

