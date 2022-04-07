import sqlite3
import os
import time
import shutil
import pandas as pd
from objects import cells
from objects.order import Order


def set_up_db(sim_env):
    """Create a new database for a simulation run
    :param sim_env (Simulation environment object) The simulation environment of the run
    :return connectors to the database"""

    if sim_env.train_model:
        return None, None

    if sim_env.db_in_memory:
        db_con = sqlite3.connect(":memory:")
    else:
        try:
            os.remove("data/current_run.db")
        except:
            pass
        db_con = sqlite3.connect("data/current_run.db")

    db_cu = db_con.cursor()

    db_cu.execute("""CREATE TABLE machine_events (
                    machine INTEGER NOT NULL,
                    time FLOAT NOT NULL,
                    event TEXT NOT NULL,
                    est_time FLOAT,
                    next_setup_type INTEGER,
                    current_setup_type INTEGER,
                    load_item INTEGER NOT NULL CHECK(load_item IN (0,1)),
                    manufacturing INTEGER NOT NULL CHECK(manufacturing IN (0,1)),
                    setup INTEGER NOT NULL CHECK(setup IN (0,1)),
                    idle INTEGER NOT NULL CHECK(idle IN (0,1)),
                    repair INTEGER NOT NULL CHECK(repair IN (0,1)),
                    item_in_input INTEGER,
                    item_in_machine INTEGER,
                    item_in_output INTEGER
    )""")

    db_cu.execute("""CREATE TABLE agent_events (
                    agent INTEGER NOT NULL,
                    time FLOAT NOT NULL,
                    event TEXT NOT NULL,
                    next_position INTEGER,
                    travel_time FLOAT,
                    moving INTEGER NOT NULL CHECK(moving IN (0,1)),
                    waiting INTEGER NOT NULL CHECK(waiting IN (0,1)),
                    task INTEGER NOT NULL CHECK(task IN (0,1)),
                    position INTEGER,
                    picked_up_item INTEGER,
                    locked_item INTEGER
    )""")

    db_cu.execute("""CREATE TABLE item_events (
                            item INTEGER NOT NULL,
                            time FLOAT NOT NULL,
                            event TEXT NOT NULL,
                            started INTEGER NOT NULL CHECK(started IN (0,1)),
                            over_due INTEGER NOT NULL CHECK(over_due IN (0,1)),
                            blocked INTEGER NOT NULL CHECK(blocked IN (0,1)),
                            tasks_finished INTEGER NOT NULL CHECK(tasks_finished IN (0,1)),
                            completed INTEGER NOT NULL CHECK(completed IN (0,1)),
                            picked_up INTEGER NOT NULL CHECK(picked_up IN (0,1)),
                            transportation INTEGER NOT NULL CHECK(transportation IN (0,1)),
                            processing INTEGER NOT NULL CHECK(processing IN (0,1)),
                            wait_for_repair INTEGER NOT NULL CHECK(wait_for_repair IN (0,1)),
                            tasks_remaining INTEGER,
                            cell INTEGER,
                            position INTEGER,
                            position_type TEXT,
                            picked_up_by INTEGER,
                            locked_by INTEGER
                            )""")

    db_cu.execute("""CREATE TABLE buffer_events (
                            buffer INTEGER NOT NULL,
                            time FLOAT NOT NULL,
                            event TEXT NOT NULL,
                            event_item INTEGER,
                            full INTEGER NOT NULL CHECK(full IN (0,1)),
                            items_in_storage INTEGER
                            )""")

    db_con.commit()
    return db_con, db_cu


def save_as_csv(sim_env, run):
    """Save an event log database as csv file for further exploration
    :param sim_env: (Simulation Environment object) The environment of the simulation run
    :param run: (int) The number of the performed run"""

    if sim_env.train_model:
        return

    print("\nSave database tables as CSV-files for further exploration")
    start_time = time.time()
    sim_env.db_cu.execute("SELECT name FROM sqlite_master WHERE type='table'")
    data = sim_env.db_cu.fetchall()
    directory = "../data/{}".format("sim_run_"+str(run))

    if not os.path.exists(directory):
        os.makedirs(directory)

    for table in data:
        pd.read_sql_query("SELECT * from {table}".format(table=table[0]), sim_env.db_con).to_csv("{directory}/{table}.csv".format(directory=directory, table=table[0]))

    print("Saving finished in %d seconds!" % (time.time() - start_time))


def clear_files():
    """Delete all files from former simulation runs"""
    for root, dirs, files in os.walk('data'):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def close_connection(sim_env):
    """Close the connection to the database of an simulation run
    :param sim_env: (Simulation Environment object) The environment of the simulation run"""
    if not sim_env.train_model:
        sim_env.db_con.close()


def add_final_events():
    """Add final events to the event log at the end of a simulation run"""

    for buffer in cells.InterfaceBuffer.instances:
        buffer.end_event()
    for buffer in cells.QueueBuffer.instances:
        buffer.end_event()
    for order in Order.instances:
        order.end_event()
    for agent in cells.ManufacturingAgent.instances:
        agent.end_event()
    for machine in cells.Machine.instances:
        machine.end_event()