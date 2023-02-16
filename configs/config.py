"""Configuration file for SimPy Simulation, environment and performance measures"""

configuration = {
    # "SETUP_FILE": "Layout_FAZI.txt",
    # "SETUP_FILE": "Layout_Scenario_Paper_AI.txt",
    "SETUP_FILE": "Layout_Scenario_Paper_AI_operation.txt",
    # "SETUP_FILE": "Layout_Scenario_Paper_EDD.txt",
    # "SETUP_FILE": "Layout_Scenario_Cell_0_train.txt",
    "SIMULATION_RANGE": 7200,
    "SEED_MACHINE_INTERRUPTIONS": 29378374,
    "MACHINE_FAILURE_RATE": 4,
    "FAILURE_MINIMAL_LENGTH": 10,
    "FAILURE_MAXIMAL_LENGTH": 20,

    "MACHINE_SETUP_TIME": 1,
    "MACHINE_TIME_LOAD_ITEM": 1,
    "MACHINE_TIME_RELEASE_ITEM": 1,

    "SEED_INCOMING_ORDERS": 2578695,
    "NUMBER_OF_ORDERS": 3000,
    "ORDER_MINIMAL_LENGTH": 0,
    "ORDER_MAXIMAL_LENGTH": 100,
    "SPREAD_ORDER_COMPLEXITY": 0.1,
    # "SPREAD_ORDER_Priority": 1,

    "AGENT_SPEED": 15,
    "TIME_FOR_ITEM_PICK_UP": 0.1,
    "TIME_FOR_ITEM_STORE": 0.1,

    "DISTRIBUTION_SIMPLE": True,

    "DB_IN_MEMORY": True,

    "DISTANCES": {
        "BASE_HEIGHT": 1,
        "BASE_WIDTH": 1,
        "DISTANCE_BETWEEN_CELLS": 1,
        "SAFE_DISTANCE": 0.1
    },

    "SEED_GENERATOR": {
        "SEED_GEN_M_INTERRUPTIONS": 2928337,
        #"SEED_GEN_INC_ORDERS": 122424
        "SEED_GEN_INC_ORDERS": 484837
    }
}

evaluation_measures = {
    "machine": {
        "setup_events": True,
        "setup_time": True,
        "idle_time": True,
        "processing_time": True,
        "processed_quantity": True,
        "finished_quantity": True,
        "time_to_repair": True,
        "failure_events": True,
        "avg_time_between_failure": True,
        "avg_processing_time_between_failure": True,
        "avg_time_to_repair": True,
        "availability": True
    },

    "order": {
        "completion_time": True,
        "tardiness": True,
        "lateness": True,
        "transportation_time": True,
        "avg_transportation_time": True,
        "time_at_machines": True,
        "time_in_interface_buffer": True,
        "time_in_queue_buffer": True,
        "production_time": True,
        "wait_for_repair_time": True
    },
    
    "agent": {
        "moving_time": True,
        "transportation_time": True,
        "waiting_time": True,
        "idle_time": True,
        "task_time": True,
        "started_tasks": True,
        "avg_task_length": True,
        "utilization": True
    },

    "buffer": {
        "time_full": True,
        "overfill_rate": True,
        "avg_items_in_storage": True,
        "avg_time_in_storage": True
    },

    "cell": {
        "mean_time_in_cell": True,
        "mean_items_in_cell": True,
        "capacity": True,
        "storage_utilization": True
    },

    "simulation": {
        "arrived_orders": True,
        "processed_quantity": True,
        "processed_in_time": True,
        "processed_in_time_rate": True,
        "in_time_rate_by_order_type": True,
        "processed_by_order_type": True,
        "mean_tardiness": True,
        "mean_lateness": True
    }
}