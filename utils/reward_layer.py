import pandas as pd


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
        choice["_destination"] == -1
    ]

    penalty = -1000 * sum(criteria)

    return penalty


def reward_action(old_state, new_state, order):
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
        # A finished order was placed at the main output of the environment

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