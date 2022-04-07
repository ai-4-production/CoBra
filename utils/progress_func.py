import simpy


def show_progress_func(env, sim_env):
    """Print out the current progress and occupancy while simulating
    :param env: (simpy environment) The simpy environment of the run
    :param sim_env: (simulation environment object) The simulation environment of the run"""

    periods = 10
    period_length = sim_env.simulation_time_range/periods
    counter = 1

    def show_occupancy():
        """Print out the current cell occupancy"""
        print("\nCurrent orders per cell:")
        for cell in sim_env.cells:
            order_amount = len(cell.orders_in_cell)
            capacity = cell.cell_capacity
            bar = '█' * order_amount
            bar = bar or '▏'
            label = "Cell {id} ({type})".format(id=cell.id, type=cell.type)
            print(f'{label.rjust(15)} ▏ {order_amount:#2d} / {capacity:#2d} {bar}')
        print()

    while counter <= periods:
        yield env.timeout(period_length)
        print("Finished", (100/periods)*counter, "% of the simulation!")
        show_occupancy()
        counter += 1


