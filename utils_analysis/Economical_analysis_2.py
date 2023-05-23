import os
import json
import numpy as np
import csv
from matplotlib import pyplot as plt

# Defined costs and fees
product_types = ["Produkt A","Produkt B","Produkt C","Produkt D"]
product_revenues = {"Produkt A": 200, "Produkt B": 300, "Produkt C": 220, "Produkt D": 320}
product_costs = {"Produkt A": 120, "Produkt B": 170, "Produkt C": 130, "Produkt D": 180}
standard_processing_fee = 20
rush_processing_fee = 30
priority_processing_fee = 40
combined_processing_fee = 50

# Initialize counters
standard_revenue, rush_revenue, priority_revenue, combined_revenue = 0, 0, 0, 0
standard_cost, rush_cost, priority_cost, combined_cost = 0, 0, 0, 0
standard_delay_fee, rush_delay_fee, priority_delay_fee, combined_delay_fee = 0, 0, 0, 0

path = os.getcwd() + '/result/P_2/Benchmarks/Due_to_1.json'

with open(path) as numbers:
    data = json.load(numbers)

lengths_low_load = [21.4, 22.0, 26.4, 28.1]
load_constant = 40
rush_service_fee = 50
priorities = [0, 1]
urgencies = [0, 1]
penalty_tardiness = 2
penalty_prioritized = 1.5
penalty_rush = 2.0

total_revenue = 0
total_cost = 0

for order in range(len(data["orders"][0]["orders"])):
    type = data["orders"][0]["orders"][order]["type"]
    priority = data["orders"][0]["orders"][order]["priority"]
    urgency = data["orders"][0]["orders"][order]["urgency"]
    length = lengths_low_load[product_types.index(type)] + load_constant

    order_revenue = product_revenues[type]
    order_cost = product_costs[type]

    if urgency == 1 and priority == 1:
        length += 45
        total_revenue += rush_service_fee
        total_cost += combined_processing_fee
        combined_revenue += order_revenue + rush_service_fee
        combined_cost += order_cost + combined_processing_fee
    elif urgency == 1:
        length += 45
        total_revenue += rush_service_fee
        total_cost += rush_processing_fee
        rush_revenue += order_revenue + rush_service_fee
        rush_cost += order_cost + rush_processing_fee
    elif priority == 1:
        total_cost += priority_processing_fee
        priority_revenue += order_revenue
        priority_cost += order_cost + priority_processing_fee
    else:
        total_cost += standard_processing_fee
        standard_revenue += order_revenue
        standard_cost += order_cost + standard_processing_fee

    total_revenue += order_revenue
    total_cost += order_cost
    throughput_time = data["orders"][0]["orders"][order]["item_results"]["transportation_time"]+data["orders"][0]["orders"][order]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][order]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][order]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][order]["item_results"]["wait_for_repair_time"]
    tardiness = data["orders"][0]["orders"][order]["start"]  + throughput_time - length - data["orders"][0]["orders"][order]["due_to"] 
    if tardiness > 0:
        penalty = penalty_tardiness * tardiness
        total_revenue -= penalty
        if priority == 1 and urgency == 1:
            total_revenue -= penalty_prioritized * penalty
            combined_delay_fee += penalty_rush * penalty
        elif priority == 1:
            total_revenue -= penalty_prioritized * penalty
            priority_delay_fee += penalty_prioritized * penalty
        elif urgency == 1:
            total_revenue -= penalty_rush * penalty
            rush_delay_fee += penalty_rush * penalty
        else:
            standard_delay_fee += penalty

print("Total revenue: ", total_revenue)
print("Total cost: ", total_cost)
print("Profit: ", total_revenue - total_cost)

# Analysis per order type
print("Standard Orders: Revenue -", standard_revenue, "Cost -", standard_cost, "Delay Fees -", standard_delay_fee, "Profit -", standard_revenue - standard_cost - standard_delay_fee)
print("Rush Orders: Revenue -", rush_revenue, "Cost -", rush_cost, "Delay Fees -", rush_delay_fee, "Profit -", rush_revenue - rush_cost - rush_delay_fee)
print("Priority Orders: Revenue -", priority_revenue, "Cost -", priority_cost, "Delay Fees -", priority_delay_fee, "Profit -", priority_revenue - priority_cost - priority_delay_fee)
print("Combined Rush/Priority Orders: Revenue -", combined_revenue, "Cost -", combined_cost, "Delay Fees -", combined_delay_fee, "Profit -", combined_revenue - combined_cost - combined_delay_fee)