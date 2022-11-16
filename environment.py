from objects.manufacturing_agents import ManufacturingAgent
from objects.order import load_order_types, order_arrivals, Order
import time
from objects.rulesets import load_rulesets
from utils import calculate_measures, database, check_config, time_tracker
from utils.init_simulation_env import *
from utils.save_results import SimulationResults
from utils.progress_func import show_progress_func
import configs.config as configuration_file
from objects.materials import load_materials
import numpy as np
import json


class SimulationEnvironment:
    instances = []

    def __init__(self, env: simpy.Environment, config: dict, main_cell: cells.DistributionCell, train=False):
        self.env = env

        # Attributes and Settings
        self.config_file = config
        self.simulation_time_range = config.get("SIMULATION_RANGE", 1000)
        self.seed_machine_interruptions = config.get("SEED_MACHINE_INTERRUPTIONS", 464638465)
        self.seed_incoming_orders = config.get("SEED_INCOMING_ORDERS", 37346463)
        self.number_of_orders = config.get("NUMBER_OF_ORDERS", 0)
        self.min_order_length = config.get("ORDER_MINIMAL_LENGTH")
        self.max_order_length = config.get("ORDER_MAXIMAL_LENGTH")
        self.order_complexity_spread = config.get("SPREAD_ORDER_COMPLEXITY", 0)
        self.db_in_memory = config.get("DB_IN_MEMORY")
        self.train_model = train

        self.main_cell = main_cell
        self.cells = []

        self.result = None

        self.db_con, self.db_cu = database.set_up_db(self)
        self.__class__.instances.append(self)


def set_up_sim_env(config: dict, env: simpy.Environment, setup, train):
    """Setting up an new simulation environment

    :param config: (dict) The simulation config with simulation attributes like seeds
    :param env: (simpy environment) An simpy environment
    :param setup: (Pandas dataframe) The cell setup. Either created by setup process or loaded from file
    :param train: (boolean) Determines if the simulation is used to train an reinforcement learning algorithm

    :return sim_env: A new created simulation environment object"""

    # Generate objects from setup
    cells = generator_from_setup(setup, config, env)

    # Create new simulation environment and set this to all cells
    main_cell = cells[np.isnan(cells["Parent"])]["cell_obj"].item()

    sim_env = SimulationEnvironment(env, config, main_cell, train)

    set_env_in_cells(sim_env, cells["cell_obj"])

    return sim_env


def simulation(runs=1, show_progress=False, save_log=True,
               change_interruptions=True, change_incoming_orders=True, train=False):
    """Main function of the simulation: Create project setup and run simulation on it"""

    config = configuration_file.configuration
    eval_measures = configuration_file.evaluation_measures

    # Check configuration files before simulating
    check_config.check_configuration_file(config)
    check_config.check_state_attributes()

    # Load order types and rulesets from file
    load_order_types()
    load_rulesets(train)
    load_materials()

    # Set seed on agents
    cells.set_agent_seed(12345)

    # Clear the event log database
    database.clear_files()

    # Create random seeds if desired
    if change_interruptions:
        np.random.seed(seed=config["SEED_GENERATOR"]["SEED_GEN_M_INTERRUPTIONS"])
        interruption_seeds = np.random.randint(99999999, size=runs)
    else:
        interruption_seeds = np.full([runs, ], config["SEED_MACHINE_INTERRUPTIONS"])

    if change_incoming_orders:
        np.random.seed(config["SEED_GENERATOR"]["SEED_GEN_INC_ORDERS"])
        order_seeds = np.random.randint(99999999, size=runs)
    else:
        order_seeds = np.full([runs, ], config["SEED_INCOMING_ORDERS"])

    # Switch between new setup and loading an existing one
    if yes_no_question("Do you want to load an existing cell setup? [Y/N]\n"):
        configuration = load_setup_from_config(config)
    else:
        configuration = new_cell_setup()

    # Run the set amount of simulations
    for sim_count in range(runs):
        config["SEED_MACHINE_INTERUPTIONS"] = interruption_seeds[sim_count].item()
        config["SEED_INCOMING_ORDERS"] = order_seeds[sim_count].item()
        env = simpy.Environment()

        simulation_environment = set_up_sim_env(config, env, configuration, train)

        print('----------------------------------------------------------------------------')
        # Reset Timer
        time_tracker.reset_timer()
        start_time = time.time()

        env.process(order_arrivals(env, simulation_environment, config))

        if show_progress:
            env.process(show_progress_func(env, simulation_environment))
        # print("cells.ManufacturingAgent.instances: ", cells.ManufacturingAgent.instances)
        # print(len(cells.ManufacturingAgent.instances))
        time.sleep(15)
        env.run(until=config["SIMULATION_RANGE"])

        print('\nSimulation %d finished in %d seconds!' % (sim_count + 1, time.time() - start_time))
        print("Time Tracker:\nTime for state calculations: %d seconds \nTime for destination calculations: %d seconds" % (time_tracker.time_state_calc, time_tracker.time_destination_calc))
        print("State Calculations:\nTime for occupancy: %d seconds \nTime for order attributes: %d seconds" % (time_tracker.time_occupancy_calc, time_tracker.time_order_attr_calc))
        print("Time for action finding: Normal actions %d seconds, Smart actions: %d seconds" % (time_tracker.time_action_calc, time_tracker.time_smart_action_calc))

        if not simulation_environment.train_model:
            database.add_final_events()

            sim_run_evaluation(simulation_environment, eval_measures)

            if save_log:
                database.save_as_csv(simulation_environment, sim_count + 1)
            database.close_connection(simulation_environment)

        release_objects()

    if not simulation_environment.train_model:

        schema = json.loads("""
                            {"simulation_runs":[],
                            "orders":[]}
                            """)

        for run in SimulationResults.instances:
            schema["simulation_runs"].append(run.results)
            schema["orders"].append(run.order_results)

        with open('../result/last_runs.json', 'w') as f:
            json.dump(schema, f, indent=4, ensure_ascii=False)


def sim_run_evaluation(sim_env, eval_measures):
    """Evalute the performance of a simulation run
    :param sim_env: Simulation environment object to evaluate
    :param eval_measures: (dict) dictionary containing the measures to be calculated"""

    print("\nCalculate the chosen measures for the finished simulation run!")
    start_time = time.time()

    functionList = {"machine": calculate_measures.machine_measures,
                    "buffer": calculate_measures.buffer_measures,
                    "agent": calculate_measures.agent_measures,
                    "cell": calculate_measures.cell_measures,
                    "order": calculate_measures.order_measures,
                    "simulation": calculate_measures.simulation_measures
                    }

    objectList = {  "machine": cells.Machine.instances,
                    "buffer": cells.Buffer.instances,
                    "agent": cells.ManufacturingAgent.instances,
                    "cell": cells.Cell.instances,
                    "order": Order.instances
                    }
    for focus in eval_measures.keys():
        measures = [key for key, value in eval_measures[focus].items() if value == True]
        if focus == "simulation":
            parameters = {'sim_env': sim_env, 'measures': measures}
            sim_env.result = functionList[focus](**parameters)
        else:
            objects = objectList[focus]
            for obj_to_check in objects:
                parameters = {'sim_env': sim_env, 'obj': obj_to_check, 'measures': measures}
                obj_to_check.result = functionList[focus](**parameters)

    SimulationResults(sim_env)

    print("\nCalculation finished in %d seconds!" % (time.time() - start_time))


def release_objects():
    """Release all created objects"""
    SimulationEnvironment.instances.clear()
    cells.Cell.instances.clear()
    cells.Buffer.instances.clear()
    cells.ManufacturingAgent.instances.clear()
    cells.Machine.instances.clear()
    Order.instances.clear()
    Order.finished_instances.clear()

