"""This file defines what attributes should be included in the state for either normal or smart agents.
A list of useable values is including within the documentation. There are certain attributes that have to be included."""

normal_state = {"order": ["type", "tasks_finished", "next_task", "locked", "picked_up", "in_same_cell", "in_m_input", "in_m", "processing"],
                "buffer": ["free_slots"],
                "machine": ["failure"],
                "agent": []}

smart_state = {"order": ["type", "tasks_finished", "next_task", "locked", "picked_up", "in_same_cell", "in_m_input", "in_m", "processing"],
                "buffer": ["free_slots"],
                "machine": ["current_setup", "failure"],
                "agent": []}


smart_state_full = {"order": [
                        "start", "due_to", "complexity", "type",
                        "time_in_cell", "locked", "picked_up", "processing", "tasks_finished", "remaining_tasks",
                        "next_task", "distance", "in_m", "in_m_input", "in_same_cell"],

               "buffer": ["interface_ingoing", "interface_outgoing", "free_slots"],

               "machine": ["machine_type", "current_setup", "in_setup", "next_setup", "remainingsetup_time", "manufacturing", "failure", "remaining_man_time", "failure_fixed_in"],

               "agent": ["agent_position", "moving", "remaining_moving_time", "next_position", "has_task", "locked_item"]}

