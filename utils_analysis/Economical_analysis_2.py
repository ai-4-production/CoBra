import os
import json
import numpy as np
import csv
from matplotlib import pyplot as plt

# Defined costs and fees
product_types = ["Produkt A","Produkt B","Produkt C","Produkt D"]
product_revenues = {"Produkt A": 100, "Produkt B": 150, "Produkt C": 110, "Produkt D": 160}
product_costs = {"Produkt A": 60, "Produkt B": 85, "Produkt C": 65, "Produkt D": 90}
standard_processing_fee = 10
priority_processing_fee = 30
rush_service_fee = 30
combined_processing_fee = 50

# Initialize counters
standard_revenue, rush_revenue, priority_revenue, combined_revenue = 0, 0, 0, 0
add_on_rev = 0
standard_cost, rush_cost, priority_cost, combined_cost = 0, 0, 0, 0
standard_delay_fee, rush_delay_fee, priority_delay_fee, combined_delay_fee, all_fee = 0, 0, 0, 0, 0

path = os.getcwd() + '/result/P_3/FiFO_local_Run_4.json'

with open(path) as numbers:
    data = json.load(numbers)

lengths_low_load = [21.4, 22.0, 26.4, 28.1]
load_constant = 40
priorities = [0, 1]
urgencies = [0, 1]
penalty_tardiness = 0.2
penalty_prioritized = 2
penalty_rush = 0.5

total_revenue = 0
total_cost = 0

for order in range(len(data["orders"][0]["orders"])):
    completion_time = data["orders"][0]["orders"][order]["item_results"]["completion_time"]
    if completion_time != 0:
        type = data["orders"][0]["orders"][order]["type"]
        priority = data["orders"][0]["orders"][order]["priority"]
        urgency = data["orders"][0]["orders"][order]["urgency"]
        length = lengths_low_load[product_types.index(type)] + load_constant

        order_revenue = product_revenues[type]
        order_cost = product_costs[type]

        if urgency == 1 and priority == 1:
            length += 15
            total_revenue += combined_processing_fee
            combined_revenue += order_revenue + combined_processing_fee
            add_on_rev += combined_processing_fee
            combined_cost += order_cost
        elif urgency == 1 and priority == 0:
            length += 15
            total_revenue += rush_service_fee
            rush_revenue += order_revenue + rush_service_fee
            add_on_rev += rush_service_fee
            rush_cost += order_cost 
        elif priority == 1 and urgency == 0:
            length += 15
            total_revenue += priority_processing_fee
            priority_revenue += order_revenue + priority_processing_fee
            add_on_rev += priority_processing_fee
            priority_cost += order_cost
        else:
            length += 45
            standard_revenue += order_revenue
            standard_cost += order_cost

        total_revenue += order_revenue
        total_cost += order_cost
        throughput_time = data["orders"][0]["orders"][order]["item_results"]["transportation_time"]+data["orders"][0]["orders"][order]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][order]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][order]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][order]["item_results"]["wait_for_repair_time"]
        tardiness = data["orders"][0]["orders"][order]["start"]  + throughput_time - length - data["orders"][0]["orders"][order]["due_to"] 
        if tardiness > 0: 
            # total_revenue -= penalty
            if priority == 1 and urgency == 1:
                # total_revenue -= penalty_prioritized * penalty
                combined_delay_fee += penalty_rush * tardiness
                combined_delay_fee += penalty_prioritized * tardiness
                all_fee += penalty_rush * tardiness + penalty_prioritized * tardiness
            elif priority == 1 and urgency == 0:
                # total_revenue -= penalty_prioritized * penalty
                priority_delay_fee += penalty_prioritized * tardiness
                all_fee += penalty_prioritized * tardiness
            elif urgency == 1 and priority == 0:
                # total_revenue -= penalty_rush * penalty
                rush_delay_fee += penalty_rush * tardiness
                all_fee += penalty_rush * tardiness
            else:
                standard_delay_fee += penalty_tardiness * tardiness
                all_fee += penalty_tardiness * tardiness

print("Total revenue: ", round(total_revenue, 1))
print("Add-on revenue: ", round(add_on_rev, 1))
print("Total cost: ", round(total_cost, 1))
print("Profit: ", round(total_revenue - total_cost - all_fee, 1))
print("Standard Orders - Revenue:", round(standard_revenue, 1))
print("Cost:", round(standard_cost, 1))
print("Delay Fees:", round(standard_delay_fee, 1))
print("Profit:", round(standard_revenue - standard_cost - standard_delay_fee, 1))
print("Rush Orders - Revenue:", round(rush_revenue, 1))
print("Cost:", round(rush_cost, 1))
print("Delay Fees:", round(rush_delay_fee, 1))
print("Profit:", round(rush_revenue - rush_cost - rush_delay_fee, 1))
print("Priority Orders - Revenue:", round(priority_revenue, 1))
print("Cost:", round(priority_cost, 1))
print("Delay Fees:", round(priority_delay_fee, 1))
print("Profit:", round(priority_revenue - priority_cost - priority_delay_fee, 1))
print("Combined Rush/Priority Orders - Revenue:", round(combined_revenue, 1))
print("Cost:", round(combined_cost, 1))
print("Delay Fees:", round(combined_delay_fee, 1))
print("Profit:", round(combined_revenue - combined_cost - combined_delay_fee, 1))