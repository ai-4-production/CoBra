from objects.processing_steps import ProcessingStep
import objects.materials
import simpy
import numpy as np


class Machine:
    instances = []

    def __init__(self, config: dict, env: simpy.Environment, task_id):
        self.env = env
        self.simulation_environment = None
        self.cell = None
        self.coordinates = None  # Used only for initialization of distances within its cell

        # Attributes
        for task in ProcessingStep.instances:
            if task.id == int(task_id):
                self.performable_task = task  # Processing step the machine can perform
                break

        self.error_rate = config["MACHINE_FAILURE_RATE"]  # Average machine failure in 1000 processing time periods
        self.failure_min_length = config["FAILURE_MINIMAL_LENGTH"]  # Minimal length of a machine failure
        self.failure_max_length = config["FAILURE_MAXIMAL_LENGTH"]  # Maximal length of a machine failure
        self.base_setup_time = config["MACHINE_SETUP_TIME"]  # Time to change the machines setup
        self.loading_time = config["MACHINE_TIME_LOAD_ITEM"]  # Time to load an item from input into machine
        self.releasing_time = config["MACHINE_TIME_RELEASE_ITEM"]  # Time to release an item from machine to output

        # State
        self.waiting_agents = []  # Agents waiting at machine for a free input slot
        self.wait_for_item_proc = None  # Own process if machine is waiting for an new item to arrive at machine
        self.wait_for_output_proc = None  # Own process if output is blocked and item in machine can not be released
        self.setup_proc = None  # Own process if machine is in setup
        self.wait_for_setup_and_load = False  # True if machine is waiting for setup and/or loading an item

        self.expected_orders = []  # Announced orders to arrive at machine (order, time, agent)
        self.next_expected_order = None  # Next order that will arrive at machine
        self.current_setup = None  # Current setup of the machine (OrderType)
        self.previous_item = None  # The previous processed item

        self.item_in_input = None  # Input slot of the machine
        self.item_in_machine = None  # Slot within the machine. Item can be processed here.
        self.item_in_output = None  # Output slot of the machine
        self.input_lock = False  # Input locked if an agent is currently storing an item in machine input

        self.idle = True  # Machine does not perform any task
        self.load_item = False  # Machine is loading an item from input into the machine

        self.manufacturing = False  # Machine is currently processing an item within the machine
        self.manufacturing_start_time = None  # Processing start time
        self.manufacturing_time = 0  # Time the whole processing takes
        self.remaining_manufacturing_time = 0  # Remaining time the machine has to process the current item
        self.manufacturing_end_time = None  # Expected end time of the current processing

        self.setup = False  # Machine is currently in setup
        self.setup_start_time = None  # Time when setup was started
        self.remaining_setup_time = 0  # Remaining time until setup will be finished
        self.setup_finished_at = None  # Time when setup will be finished

        self.failure = False  # Machine has a failure currently and is in repair
        self.failure_time = None  # Length of the current repair
        self.failure_fixed_in = 0  # Remaining time until failure was fixed
        self.failure_fixed_at = 0  # Time when failure will be fixed

        self.__class__.instances.append(self)
        self.result = None

        # Start processes
        self.env.process(self.initial_event())
        self.main_proc = self.env.process(self.main_process())

    def main_process(self):
        """Main control flow for this machine. It will start setup, load/unload items and start producing. If there
        is no order to be produced it will start a waiting task which has to be interrupted in order to continue."""

        try:
            if self.expected_orders or self.item_in_input:
                if self.item_in_input:
                    self.next_expected_order = self.item_in_input
                else:
                    self.expected_orders.sort(key=lambda x:x[1])
                    self.next_expected_order = self.expected_orders[0][0]

                # Setup has to be finished and an order loaded to start producing
                self.wait_for_setup_and_load = True
                laden = self.env.process(self.get_item())
                self.setup_proc = self.env.process(self.setup_process(self.next_expected_order))
                yield laden & self.setup_proc

                self.next_expected_order = None
                self.wait_for_setup_and_load = False

                # Start producing
                yield self.env.process(self.starter())

                # Releasing finished item
                yield self.env.process(self.release_item_to_output())

                # Start another main process
                self.main_proc = self.env.process(self.main_process())
            else:
                # There is no item in machine input or expected to arrive soon
                self.wait_for_item_proc = self.env.process(self.wait_for_item())
                yield self.wait_for_item_proc

                self.main_proc = self.env.process(self.main_process())

        except simpy.Interrupt as interruption:
            # Stop main process
            self.main_proc = None

            try:
                laden.interrupt("The expected order arrival was cancel")
            except Exception:
                pass

            self.next_expected_order = None
            self.wait_for_setup_and_load = False

            # Start another main process
            self.main_proc = self.env.process(self.main_process())

    def get_item(self):
        """Load item from input into the machine. Start waiting if there is no item to be loaded."""
        try:
            if not self.item_in_input:
                self.wait_for_item_proc = self.env.process(self.wait_for_item())
                yield self.wait_for_item_proc

            if self.item_in_input is not None and self.item_in_machine is None and self.item_in_input.next_task == self.performable_task:
                # State changes - Start loading item from input into the machine
                self.idle = False
                self.load_item = True
                self.save_event("load_item_start")

                # Perform loading
                yield self.env.timeout(self.loading_time)

                # State changes after loading
                self.item_in_machine = self.item_in_input
                self.item_in_input = None
                self.load_item = False
                if not self.setup:
                    self.idle = True

                self.save_event("load_item_end")

                # Inform agent that wait for an free input space
                if len(self.waiting_agents) > 0:
                    self.waiting_agents[0].current_waitingtask.interrupt("New free slot in Machine Input")

                # Inform agents in cell that state has changed
                self.cell.inform_agents()
            else:
                raise Exception("Can not load item from machine input! "
                                "This may most likely be caused by an item that can not be processed by this machine.")

        except simpy.Interrupt as interruption:

            if self.load_item:
                self.load_item = False
                if not self.setup:
                    self.idle = True

            if self.wait_for_item_proc:
                self.wait_for_item_proc.interrupt()

    def setup_process(self, next_item):
        """Planned setup if next item has a different type than the previous one

        :param next_item: (Order object) Item that should be processed next
        """

        # Can not perform setup during processing or failure
        if self.manufacturing or self.failure:
            return

        # Calculate needed setup time
        setup_time, new_task = self.calculate_setup_time(next_item=next_item)

        if setup_time == 0:
            # No setup needed
            return
        else:
            # Setup needed
            self.start_setup(setup_time, new_task)

            # Perform setup
            yield self.env.timeout(self.remaining_setup_time)

            # Setup finished
            self.end_setup(new_task)

    def calculate_setup_time(self, next_item):
        """Calculate Time needed for setup.

        :param next_item: (Order object) Item to be processed next.
        :return Setup time, Next setup type: 0, None if no setup needed because
        the machine is already in the needed setup type"""
        current_type = self.current_setup

        if next_item.type == current_type:
            return 0, None
        else:
            return self.base_setup_time, next_item.type

    def release_item_to_output(self):
        """Release finished Item to output slot of this machine"""

        if self.manufacturing or not self.item_in_machine:
            # There is no item to be released
            return

        if self.item_in_output:
            # Output already used. Start waiting.
            self.item_in_machine.blocked_by = self.item_in_output
            self.wait_for_output_proc = self.env.process(self.wait_for_free_output())
            yield self.wait_for_output_proc
            self.item_in_machine.blocked_by = None

        # Start releasing
        self.idle = False
        self.save_event("release_item_start")
        yield self.env.timeout(self.releasing_time)

        # Releasing finished
        released_item = self.item_in_machine
        self.item_in_machine = None
        self.item_in_output = released_item

        if not self.setup and not self.load_item:
            self.idle = True

        self.save_event("release_item_end")

        # Inform agents in cell about the state change
        self.cell.inform_agents()

    def starter(self):
        """Start a new manufacturing process.
        Used to determine wether a manufacturing process is new or the continuation of a previous one"""
        manufacturing_process = self.env.process(self.process_manufacturing(new=True))
        yield manufacturing_process

    def process_manufacturing(self, new: bool):
        """
        Perform the main manufacturing process of the machine
        :param new: (boolean) Is the item in machine an new item or an partly processed one?
        """
        if self.setup:
            print("Machine Error: Machine is in Setup!")
            return
        if self.item_in_machine is None or self.item_in_machine.next_task is not self.performable_task:
            print("Machine Error: There is not an useful item in the machine!")
            return
        elif self.item_in_machine.type is not self.current_setup:
            print("Machine Error: Machine is in wrong setup!", self, self.current_setup, self.env.now, self.item_in_machine)
            return

        def start_new():
            # State changes for new manufacturing process
            self.manufacturing_time = self.calculate_processing_time(self.item_in_machine, self.performable_task)
            self.remaining_manufacturing_time = self.manufacturing_time
            self.idle = False
            self.manufacturing = True
            self.manufacturing_start_time = self.env.now
            self.manufacturing_end_time = self.manufacturing_start_time + self.manufacturing_time
            self.item_in_machine.processing = True
            self.item_in_machine.save_event("processing_start")
            self.save_event("production_start", est_time=self.manufacturing_time)

        def continue_process():
            # State changes for continuation of an former manufacturing process
            self.idle = False
            self.manufacturing = True
            self.manufacturing_start_time = self.env.now
            self.manufacturing_end_time = self.manufacturing_start_time + self.remaining_manufacturing_time
            self.item_in_machine.processing = True
            self.item_in_machine.save_event("processing_continue")
            self.save_event("failure_end")

        if new:
            start_new()
        else:
            continue_process()

        # Calculate machine failures
        if self.error_rate > 0:
            errors = np.random.uniform(low=0, high=1000, size=self.error_rate)
            errors.sort()
            first_error = errors[0]
        else:
            first_error = float('inf')

        if first_error < self.remaining_manufacturing_time:
            # Wait until next error and trigger and failure event
            yield self.env.timeout(first_error)
            yield self.env.process(self.failure_event())

        else:
            # Perform the remaining manufacturing time
            yield self.env.timeout(self.remaining_manufacturing_time)

            # State changes after manufacturing
            self.end_manufacturing()

    def calculate_processing_time(self, item, task):
        """
        Calculate how long an item will take to be processed by this machine

        :param item: Which item should be processed?
        :param task: What task should be performed?
        :return: Manufacturing time needed for this item
        """
        material_attributes = []

        materials = objects.materials.loaded_materials

        for composition_element in item.composition:
            for material in materials['materials']:
                if material['title'] == composition_element:
                    material_complexity = material['complexity']
                    material_hardness = material['hardness']
                    break
            material_attributes.append((item.composition[composition_element], material_complexity, material_hardness))

        manufacturing_time = (task.base_duration * item.complexity * sum([amount*(complexity+(hardness/2)) for amount, complexity, hardness in material_attributes]))/10

        return manufacturing_time

    def failure_event(self):
        """Machine has an unplanned Failure Event. Calculate length of repair and continue production afterwards"""

        self.failure_start()

        yield self.env.timeout(self.failure_fixed_in)

        self.failure_end()

        yield self.env.process(self.process_manufacturing(new=False))

    def wait_for_free_output(self):
        """Infinite waiting loop. It has to be interrupted by an simpy interruption to be stopped.
        Loop will be interrupted if the machine output slot is free again."""

        try:
            self.item_in_machine.blocked_by = self.item_in_output
            while 1:
                yield self.env.timeout(1000)

        except simpy.Interrupt as interruption:
            self.item_in_machine.blocked_by = None
            self.wait_for_output_proc = None

    def wait_for_item(self):
        """Infinite waiting loop. It has to be interrupted by an simpy interruption to be stopped.
        Loop will be interrupted if an item was announced to arrive at the machine soon."""

        try:
            while 1:
                yield self.env.timeout(1000)
        except simpy.Interrupt as interruption:
            self.wait_for_item_proc = None

    def start_setup(self, setup_time, new_task):
        """State changes: Start a new machine setup

        :param setup_time: (float) Time needed to change the setup
        :param new_task: (Order type object) The Setup to change into"""
        self.idle = False
        self.setup = True
        self.setup_start_time = self.env.now
        self.remaining_setup_time = setup_time
        self.setup_finished_at = self.setup_start_time + self.remaining_setup_time
        self.save_event("setup_start", next_setup_type=new_task)

    def end_setup(self, new_task):
        """State changes: End a machine setup

        :param new_task: (Order type object) The Setup changed to"""
        self.setup = False

        if not self.load_item:
            self.idle = True

        self.remaining_setup_time = 0
        self.setup_finished_at = None
        self.setup_start_time = None
        self.current_setup = new_task
        self.save_event("setup_end")

    def failure_start(self):
        """State changes for an appearing machine failure"""
        self.failure = True
        self.failure_time = self.env.now
        self.failure_fixed_in = np.random.uniform(low=self.failure_min_length, high=self.failure_max_length)
        self.failure_fixed_at = self.failure_fixed_in + self.failure_time
        self.manufacturing = False
        self.remaining_manufacturing_time = self.manufacturing_time - (self.env.now - self.manufacturing_start_time)
        self.manufacturing_end_time = self.failure_fixed_at + self.remaining_manufacturing_time
        self.save_event("failure_start", est_time=self.failure_fixed_in)

        self.item_in_machine.machine_failure(True)

    def failure_end(self):
        """State changes for an fixed machine failure"""
        self.failure = False
        self.failure_time = None
        self.failure_fixed_in = 0
        self.failure_fixed_at = None
        self.item_in_machine.machine_failure(False)

    def end_manufacturing(self):
        """State changes: Finish processing an item"""
        self.manufacturing = False
        self.idle = True
        self.manufacturing_start_time = None
        self.remaining_manufacturing_time = 0
        self.previous_item = self.item_in_machine
        self.item_in_machine.processing = False
        self.item_in_machine.processing_step_finished()

        if self.item_in_output:
            self.item_in_machine.blocked_by = self.item_in_output

        self.item_in_machine.save_event("processing_finished")
        self.save_event("production_end")

    def get_remaining_time(self):
        """Get remaining manufacturing time at the current moment
        :return remaining_time (float)"""
        return self.remaining_manufacturing_time - (self.env.now - self.manufacturing_start_time)

    def get_remaining_repair_time(self):
        """Get remaining repair time at the current moment
        :return remaining_time (float)"""
        return self.failure_fixed_in - (self.env.now - self.failure_time)

    def item_picked_up(self, item):
        """State changes: Agent has picked up an item from machine output
        :param item: (Order object) Item that has been picked up from machine output"""

        if self.item_in_output is not item:
            raise Exception("Agent tried to pick up an item from machine that wasn´t in the output slot!")

        self.item_in_output = None

        if self.wait_for_output_proc:
            self.wait_for_output_proc.interrupt("Output free again")
            self.save_event("item_picked_up")

    def item_stored(self, item, cell):
        """State changes: Agent has stored an item in machine input
        :param item: (Order object) Item that has been stored in machine input
        :param cell: (Cell Object): Current Cell of the object, not used in this function"""

        # Remove announcement
        self.expected_orders.remove(
            [(order, time, agent) for order, time, agent in self.expected_orders if order == item][0])

        if self.item_in_input:
            raise Exception("Agent stored an item in machine input that was already occupied")

        self.item_in_input = item

        if self.wait_for_item_proc:
            self.wait_for_item_proc.interrupt("Order arrived")

        self.save_event("item_stored")

    def cancel_expected_order(self, order):
        """Cancel an item that was expected to arrive at the machine soon
        :param order: (Order object) The item to cancel"""

        if order != self.next_expected_order:
            return

        if self.main_proc and self.wait_for_setup_and_load:
            if self.setup_proc:
                if self.setup_proc.is_alive:
                    # Wait until setup was performed
                    yield self.setup_proc

            # Interrupt the main process when the machine is in idle
            self.main_proc.interrupt("Expected Order canceled")

    def occupancy(self, attributes: list, requester=None):
        """State calculation for the machine. Gets machine attributes and orders at this machine

        :param attributes: List of strings. Each element is an attribute that should be calculated and returned
        :param requester: (Agent object) Manufacturing agent that requests the state
        :return tuple of orders within the machine slots and machine attributes. (list of dict, dict)"""

        def machine_type():
            return self.performable_task.id

        def current_setup():
            if self.current_setup:
                return self.current_setup.type_id
            else:
                return -1

        def in_setup():
            return int(self.setup)

        def next_setup():
            if self.setup:
                return self.next_expected_order.type.type_id
            else:
                return current_setup()

        def remaining_setup_time():
            if self.setup:
                return self.setup_finished_at - self.env.now
            else:
                return 0

        def manufacturing():
            return int(self.manufacturing)

        def failure():
            return int(self.failure)

        def remaining_man_time():
            if self.failure:
                return self.remaining_manufacturing_time
            elif self.manufacturing:
                return self.manufacturing_end_time - self.env.now
            else:
                return 0

        def failure_fixed_in():
            if self.failure:
                return self.failure_fixed_at - self.env.now
            else:
                return 0

        attr = {}
        for attribute in attributes:
            attr[attribute] = locals()[attribute]()

        return ([{"order": self.item_in_input, "pos": self, "pos_type": "Machine-Input"},
                {"order": self.item_in_machine, "pos": self, "pos_type": "Machine-Internal"},
                {"order": self.item_in_output, "pos": self, "pos_type": "Machine-Output"}], attr)

    def save_event(self, event_type: str, est_time=None, next_setup_type=None):
        """Save an event to the event log database. Includes the current state of the machine.

        :param event_type: (str) The title of the triggered event
        :param est_time: (float) Only at new processing process: The estimated manufacturing time if no failure event occurs
        :param next_setup_type: (OrderType object) Only if new setup is started: The setup type to be changed into"""

        if self.simulation_environment.train_model:
            return

        db = self.simulation_environment.db_con
        cursor = self.simulation_environment.db_cu

        time = self.env.now

        if next_setup_type:
            nst = id(next_setup_type)
        else:
            nst = None

        if self.current_setup:
            cst = id(self.current_setup)
        else:
            cst = None

        if self.item_in_input:
            iii = id(self.item_in_input)
        else:
            iii = None

        if self.item_in_machine:
            iim = id(self.item_in_machine)
        else:
            iim = None

        if self.item_in_output:
            iio = id(self.item_in_output)
        else:
            iio = None

        cursor.execute("INSERT INTO machine_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (id(self), time, event_type, est_time, nst, cst, self.load_item, self.manufacturing, self.setup, self.idle, self.failure, iii, iim, iio))

        db.commit()

    def initial_event(self):
        """Add initial events to event log database. Necessary to calculate measures and results"""
        self.save_event("Initial")
        yield self.env.timeout(0)

    def end_event(self):
        """Add end events to event log database. Necessary to calculate measures and results"""
        self.save_event("End_of_Time")

