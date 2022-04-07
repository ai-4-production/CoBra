"""Variables that track passed time for each simulation run"""

time_state_calc = 0
time_destination_calc = 0

time_occupancy_calc = 0
time_order_attr_calc = 0
time_pos_attr_calc = 0

time_action_calc = 0
time_smart_action_calc = 0


def reset_timer():
    """Reset all the timer variables for a new run"""
    global time_state_calc
    global time_destination_calc

    global time_occupancy_calc
    global time_order_attr_calc
    global time_pos_attr_calc

    global time_action_calc
    global time_smart_action_calc

    time_state_calc = 0
    time_destination_calc = 0

    time_occupancy_calc = 0
    time_order_attr_calc = 0
    time_pos_attr_calc = 0

    time_action_calc = 0
    time_smart_action_calc = 0
