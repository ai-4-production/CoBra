import os
import json
import numpy as np
import csv
from matplotlib import pyplot as plt

path = os.getcwd() + '/result/P_2/Benchmarks/Deep_RL_2.json'

with open(path) as numbers:
    data = json.load(numbers)

lengths_low_load = [21.4, 22.0,26.4,28.1]
load_constant = 40
product_types = ["Produkt A","Produkt B","Produkt C","Produkt D"]
product_revenues = {"Produkt A": 200, "Produkt B": 200, "Produkt C": 300, "Produkt D": 300}
rush_service_fee = 50
priorities = [0,1]
urgencies = [0,1]
penalty_tardiness = 2
penalty_prioritized = 1.5
penalty_rush = 2.0

total_revenue = 0
for order in range(len(data["orders"][0]["orders"])):
    type = data["orders"][0]["orders"][order]["type"]
    priority = data["orders"][0]["orders"][order]["priority"]
    urgency = data["orders"][0]["orders"][order]["urgency"]
    length = lengths_low_load[product_types.index(type)] + load_constant
    if urgency == 1:
        length += 45
        total_revenue += rush_service_fee
    
    order_revenue = product_revenues[type]
    total_revenue += order_revenue

    throughput_time = data["orders"][0]["orders"][order]["item_results"]["transportation_time"]+data["orders"][0]["orders"][order]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][order]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][order]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][order]["item_results"]["wait_for_repair_time"]
    tardiness = data["orders"][0]["orders"][order]["start"]  + throughput_time - length - data["orders"][0]["orders"][order]["due_to"] 
    if tardiness > 0:
        total_revenue -= penalty_tardiness * tardiness
        if priority == 1:
            total_revenue -= penalty_tardiness * penalty_prioritized * tardiness
        if urgency == 1:
            total_revenue -= penalty_tardiness * penalty_rush * tardiness

print("Total revenue: ", total_revenue)