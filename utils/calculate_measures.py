import statistics
import pandas as pd
import numpy as np
from objects.order import Order, OrderType
from utils.devisions import div_possible_zero


def machine_measures(sim_env, obj, measures=[]):
    """Calculate performance measures for a single machine in an simulation environment
    :param sim_env: (Simulation environment object) The simulation run and environment to use
    :param obj: (Machine object) The machine to evaluate
    :param measures: list of strings, representing the measures to be calculated
    :return dict of results (measure_name: value)"""

    db_con = sim_env.db_con
    simulation_length = sim_env.simulation_time_range
    result = {}
    event_counts = event_count_single_object("machine", obj, db_con)

    def setup_events():
        setup_events = event_counts[(event_counts["event"] == "setup_start")]["#events"]

        if setup_events.empty:
            return 0
        else:
            return setup_events.values[0].item()

    def setup_time():
        return boolean_times_single_object("machine", obj, "setup", db_con)

    def idle_time():
        return boolean_times_single_object("machine", obj, "idle", db_con)

    def processing_time():
        return boolean_times_single_object("machine", obj, "manufacturing", db_con)

    def processed_quantity():
        processed = event_counts[(event_counts["event"] == "production_start")]["#events"]

        if processed.empty:
            return 0
        else:
            return processed.values[0].item()

    def finished_quantity():
        finished = event_counts[(event_counts["event"] == "production_end")]["#events"]

        if finished.empty:
            return 0
        else:
            return finished.values[0].item()

    def time_to_repair():
        if failure_events():
            return boolean_times_single_object("machine", obj, "repair", db_con)
        else:
            return 0

    def failure_events():
        failures = event_counts[event_counts["event"] == "failure_start"]
        if not failures.empty:
            return failures["#events"].values[0].item()
        else:
            return 0

    def avg_time_between_failure():
        if failure_events():
            return (simulation_length - time_to_repair())/failure_events()
        else:
            return 0

    def avg_processing_time_between_failure():
        if failure_events():
            return processing_time()/failure_events()
        else:
            return 0

    def avg_time_to_repair():
        if failure_events():
            starts = event_times_single_object("machine", obj, "failure_start", db_con)
            ends = event_times_single_object("machine", obj, "failure_end", db_con)
            df = pd.merge(starts, ends, left_index=True, right_index=True)
            df["time_to_repair"] = df["time_y"] - df["time_x"]
            return df["time_to_repair"].mean()
        else:
            return 0

    def availability():
        return ((simulation_length - time_to_repair())/simulation_length) * 100

    for measure in measures:
        result[measure] = round(locals()[measure](),2)

    return result


def order_measures(sim_env, obj, measures=[]):
    """Calculate performance measures for an single order in an simulation environment
    :param sim_env: (Simulation environment object) The simulation run and environment to use
    :param obj: (Order object) The order/item to evaluate
    :param measures: list of strings, representing the measures to be calculated
    :return dict of results (measure_name: value)"""

    db_con = sim_env.db_con
    simulation_length = sim_env.simulation_time_range
    result = {}
    event_counts = event_count_single_object("item", obj, db_con)

    def completion_time():
        if obj.completed_at:
            return (obj.completed_at - obj.start).item()
        else:
            return 0

    def tardiness():
        if obj.completed_at and obj.overdue:
            return (obj.completed_at - obj.due_to).item()
        else:
            return 0

    def lateness():
        if obj.completed_at:
            return (obj.completed_at - obj.due_to).item()
        else:
            return 0

    def transportation_time():
        return boolean_times_single_object("item", obj, "transportation", db_con)

    def avg_transportation_time():
        transport_time = transportation_time()
        if transport_time == 0:
            return 0
        else:
            return transport_time/event_counts[(event_counts["event"] == "transportation_start")]["#events"].values[0].item()

    def time_at_pos():
        return time_by_dimension("item", obj, "position", db_con)

    def time_at_pos_type():
        return time_by_dimension("item", obj, "position_type", db_con)

    def time_at_machines():
        df = time_at_pos_type()
        result = df[df["position_type"] == "Machine"]
        if not result.empty:
            return result["length"].iloc[0].item()
        else:
            return 0

    def time_in_interface_buffer():
        df = time_at_pos_type()
        result = df[df["position_type"] == "InterfaceBuffer"]
        if not result.empty:
            return result["length"].iloc[0].item()
        else:
            return 0

    def time_in_queue_buffer():
        df = time_at_pos_type()
        result = df[df["position_type"] == "QueueBuffer"]
        if not result.empty:
            return result["length"].iloc[0].item()
        else:
            return 0

    def production_time():
        return (boolean_times_single_object("item", obj, "processing", db_con) - wait_for_repair_time())

    def wait_for_repair_time():
        result = boolean_times_single_object("item", obj, "wait_for_repair", db_con)
        if isinstance(result, int):
            return result
        else:
            return result.item()

    for measure in measures:
        result[measure] = round(locals()[measure](), 2)

    return result


def agent_measures(sim_env, obj, measures=[]):
    """Calculate performance measures for an single agent in an simulation environment
    :param sim_env: (Simulation environment object) The simulation run and environment to use
    :param obj: (Manufacuturing agent object) The agent to evaluate
    :param measures: list of strings, representing the measures to be calculated
    :return dict of results (measure_name: value)"""

    db_con = sim_env.db_con
    simulation_length = sim_env.simulation_time_range
    result = {}
    event_counts = event_count_single_object("agent", obj, db_con)

    def moving_time():
        return boolean_times_single_object("agent", obj, "moving", db_con)

    def transportation_time():
        df = pd.read_sql_query(
            "SELECT time, moving, picked_up_item FROM agent_events WHERE agent={object} and picked_up_item NOT NULL".format(object=id(obj)), db_con)
        df = remove_events_without_changes(df, "moving")
        df["length"] = df["time"].shift(periods=-1, axis=0) - df["time"]
        result = df.groupby(["moving"], as_index=False)["length"].sum()

        if result.empty:
            return 0

        return result[result["moving"] == 1]["length"].values[0].item()

    def waiting_time():
        result = boolean_times_single_object("agent", obj, "waiting", db_con)
        if isinstance(result, int):
            return result
        else:
            return result.item()

    def idle_time():
        return simulation_length - task_time()

    def task_time():
        return boolean_times_single_object("agent", obj, "task", db_con)

    def started_tasks():
        started = event_counts[(event_counts["event"] == "start_task")]["#events"]

        if started.empty:
            return 0
        else:
            return started.values[0].item()

    def avg_task_length():
        return div_possible_zero(task_time(), started_tasks())

    def time_at_pos():
        return time_by_dimension("agent", obj, "position", db_con)

    def utilization():
        return div_possible_zero(task_time(), simulation_length)

    for measure in measures:
        result[measure] = round(locals()[measure](), 2)

    return result


def buffer_measures(sim_env, obj, measures=[]):
    """Calculate performance measures for a single buffer in an simulation environment
    :param sim_env: (Simulation environment object) The simulation run and environment to use
    :param obj: (Buffer object) The buffer to evaluate
    :param measures: list of strings, representing the measures to be calculated
    :return dict of results (measure_name: value)"""

    db_con = sim_env.db_con
    simulation_length = sim_env.simulation_time_range
    result = {}
    event_counts = event_count_single_object("buffer", obj, db_con)
    capacity = obj.storage_capacity

    def time_full():
        return round(boolean_times_single_object("buffer", obj, "full", db_con), 2)

    def overfill_rate():
        return round((time_full()/simulation_length)*100, 2)

    def avg_items_in_storage():
        df = time_by_dimension("buffer", obj, "items_in_storage", db_con)
        df["factor"] = df["length"] * df["items_in_storage"]
        return round(df["factor"].sum()/simulation_length, 2)

    def avg_time_in_storage():
        df = time_by_dimension("buffer", obj, "event_item", db_con)
        return round(df["length"].mean(), 2)

    for measure in measures:
        result[measure] = locals()[measure]()

    return result


def cell_measures(sim_env, obj, measures=[]):
    """Calculate performance measures for a single cell in an simulation environment
    :param sim_env: (Simulation environment object) The simulation run and environment to use
    :param obj: (Order object) The order/item to evaluate
    :param measures: list of strings, representing the measures to be calculated
    :return dict of results (measure_name: value)"""

    db_con = sim_env.db_con
    simulation_length = sim_env.simulation_time_range
    orders = pd.read_sql_query("SELECT DISTINCT item as item FROM item_events WHERE cell={}".format(id(obj)), db_con)["item"]
    result = {}

    def mean_time_in_cell():
        results = []

        for order_id in orders:
            df = time_by_dimension("item", order_id, "cell", db_con, object_as_id=True)
            results.append(df[df["cell"] == id(obj)]["length"].iloc[0])

        if len(results) == 0:
            return 0

        return statistics.mean(results)

    def mean_items_in_cell():
        return (len(orders) * mean_time_in_cell())/simulation_length

    def capacity():
        return obj.cell_capacity

    def storage_utilization():
        return (mean_items_in_cell()/capacity())*100

    for measure in measures:
        result[measure] = round(locals()[measure](), 2)

    return result


def simulation_measures(sim_env, measures=[]):
    """Calculate performance measures for a single simulation environment and run
    :param sim_env: (Simulation environment object) The simulation run and environment to evaluate
    :param measures: list of strings, representing the measures to be calculated
    :return dict of results (measure_name: value)"""

    db_con = sim_env.db_con
    simulation_length = sim_env.simulation_time_range
    orders = [order for order in Order.instances if order.simulation_environment == sim_env]
    orders_completed = [order for order in orders if order.completed]
    result = {}

    def arrived_orders():
        return len(orders)

    def processed_quantity(alt_list=None):
        if alt_list:
            return len(alt_list)
        else:
            return len(orders_completed)

    def processed_in_time(alt_list=None):
        if alt_list:
            return len([order for order in alt_list if order.completed_at <= order.due_to])
        else:
            return len([order for order in orders_completed if order.completed_at <= order.due_to])

    def processed_in_time_rate(alt_list=None):
        return round((processed_in_time(alt_list)/processed_quantity(alt_list))*100, 2)

    def in_time_rate_by_order_type():
        order_types = OrderType.instances
        result = []

        for o_type in order_types:
            alt_list = [order for order in orders_completed if order.type == o_type]
            result.append((o_type.name.decode("UTF-8"), processed_in_time_rate(alt_list)))
        return result

    def processed_by_order_type():
        order_types = OrderType.instances
        result = []

        for o_type in order_types:
            alt_list = [order for order in orders_completed if order.type == o_type]
            result.append((o_type.name.decode("UTF-8"), len(alt_list)))

        return result

    def mean_tardiness():
        results = []
        for order in orders_completed:
            results.append(order_measures(sim_env, order, measures=["tardiness"])["tardiness"])
        results = [0 if v is None else v for v in results]
        return round(statistics.mean(results), 2)

    def mean_lateness():
        results = []
        for order in orders_completed:
            results.append(order_measures(sim_env, order, measures=["lateness"])["lateness"])
        results = [0 if v is None else v for v in results]
        return round(statistics.mean(results), 2)

    for measure in measures:
        result[measure] = locals()[measure]()

    return result


def boolean_times_single_object(focus: str, object, measure: str, db_con):
    """Calculate the absolute amount of time per boolean value in event_log for a specific object
    :param focus: (str) The type of object e.g. "agent", "machine", "item"...
    :param object: The object to evaluate
    :param measure: (str) Measure to evaluate. Has to be a column within the event log of the object
    :param db_con: (SQLite3 Database connection) Connection to the event log database
    :return The absolute time where the measure for this object was true (e.g. agent moving,...)
    """

    # Get data and remove rows where the measure don´t change
    df = pd.read_sql_query("SELECT time, {measure} FROM {focus}_events WHERE {focus}={object}".format(measure=measure, focus=focus, object=id(object)), db_con)
    df = remove_events_without_changes(df, measure)

    # Get length of each period
    df["length"] = df["time"].shift(periods=-1, axis=0) - df["time"]
    result = df.groupby([measure], as_index=False)["length"].sum()

    # Get time where the measure is true
    if result[result[measure] == 1].empty:
        result = 0
    else:
        result = result[result[measure] == 1]["length"].values[0]

    return result


def time_by_dimension(focus: str, object, dimension: str, db_con, object_as_id=False):
    """Calculate the time of each value of a dimension for a single object. E.g the time per position for an agent
    :param focus: (str) The type of object e.g. "agent", "machine", "item"...
    :param object: The object to evaluate
    :param dimension: (str) Dimension to evaluate. Has to be a column within the event log of the object
    :param db_con: (SQLite3 Database connection) Connection to the event log database
    :return (DataFrame) The absolute time per value of an dimension for the object
    """
    if object_as_id:
        object_id = object
    else:
        object_id = id(object)

    # Get data and remove rows where the dimension did not change
    df = pd.read_sql_query(
        "SELECT time, {dimension} FROM {focus}_events WHERE {focus}={object}".format(dimension=dimension, focus=focus,
                                                                                   object=object_id), db_con)
    df = remove_events_without_changes(df, dimension)

    # Calculate the time for each value
    df["length"] = df["time"].shift(periods=-1, axis=0) - df["time"]
    result = df.groupby([dimension], as_index=False)["length"].sum()

    return result


def event_count_single_object(focus: str, object, db_con):
    """Calculate the amount of times an event of an single object was triggered.
    :param focus: (str) The type of object e.g. "agent", "machine", "item"...
    :param object: The object to evaluate
    :param db_con: (SQLite3 Database connection) Connection to the event log database
    :return (DataFrame) The absolute amount of appearances of each event for the object"""

    result = pd.read_sql_query(
            "SELECT event, COUNT(time) as '#events' FROM {focus}_events WHERE {focus}={object} GROUP BY event".format(focus=focus,
                                                                                                object=id(object)), db_con)
    return result


def event_times_single_object(focus: str, object, event: str, db_con):
    """Get the amount of triggered events for an single event for an single object.
    :param focus: (str) The type of object e.g. "agent", "machine", "item"...
    :param object: The object to evaluate
    :param event: (str) The name of the event
    :param db_con: (SQLite3 Database connection) Connection to the event log database
    :return (DataFrame) The absolute amount of appearances of the event for the object"""

    result = pd.read_sql_query(
            "SELECT time FROM {focus}_events WHERE {focus}={object} AND event='{event}'".format(focus=focus, object=id(object), event=event), db_con)
    return result


def remove_events_without_changes(df: pd.DataFrame, column: str):
    """Take a event log and remove all rows where a specific column has not changed
    :param df: Dataframe event log to be shortend
    :param column: (str) The name of the column
    :return shortened Dataframe"""

    if df.empty:
        return df

    last_row = df.iloc[-1]
    df["to_drop"] = df[column].shift(periods=1, axis=0) == df[column]
    df = df.drop(df[df["to_drop"]].index)
    del df["to_drop"]
    df = df.append(last_row)

    return df