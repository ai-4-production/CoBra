from tkinter import Y
import environment

# Start a new simulation
sim_results = environment.simulation(show_progress=True, save_log=False, runs=1, change_interruptions=True, change_incoming_orders=True, train=False)