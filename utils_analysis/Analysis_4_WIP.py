import os
import json
import time
import csv
import numpy as np
path = os.getcwd() + '/result/Scenario_1_2800/Run_3/last_runs_02-15-2023_18-17-04_RL.json'
# path = os.getcwd() + '/result/last_runs_01-30-2023_10-41.json'

numbers = open(path)
data = json.load(numbers)

# count files in directory that needs to be analyzed
json_file_count = 0 
for roots,dirs, files in os.walk(os.getcwd() + '/Benchmark/Data'):
    for file in files:
        if os.path.splitext(file)[1] == '.json':
            json_file_count += 1   

prio_data = []
length = 0
priorities = [0,1,2]
groups = [0, 1]
lenghts = [26.64, 21.55,13.81,18.93]
product_types = ["Produkt A","Produkt B","Produkt C","Produkt D"]
wip = []
wip_count, prio_0, prio_1, prio_2 = 0,0,0,0
completion_time = 0
for i in range(len(data["orders"][0]["orders"])):
    if data["orders"][0]["orders"][i]["item_results"]["completion_time"] != 0:
        wip_count, prio_0, prio_1, prio_2 = 0,0,0,0
        for j in range(len(data["orders"][0]["orders"])):
            priority = data["orders"][0]["orders"][j]["priority"]
            lateness = data["orders"][0]["orders"][j]["item_results"]["lateness"]
            completion_time = data["orders"][0]["orders"][j]["item_results"]["completion_time"]
            start = data["orders"][0]["orders"][j]["start"] 
            throughput_time = data["orders"][0]["orders"][j]["item_results"]["transportation_time"]+data["orders"][0]["orders"][j]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][j]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][j]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][j]["item_results"]["wait_for_repair_time"]
            order_proc_end = start + completion_time
            order_proc_start = order_proc_end - throughput_time
            if i >= order_proc_start and i <= order_proc_end:
                wip_count += 1
                if priority == 0:
                    prio_0 += 1
                elif priority == 1:
                    prio_1 += 1
                elif priority == 2:
                    prio_2 += 1
            
        wip.append([wip_count, prio_0, prio_1, prio_2])
        with open('WIP' + '.csv', 'a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([i, wip_count, prio_0, prio_1, prio_2])
print("Total mean WIP: ", np.mean(wip, axis = 0), " , len: ",len(wip))
