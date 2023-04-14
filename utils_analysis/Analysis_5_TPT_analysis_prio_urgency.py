import os
import json
import time
import numpy as np
import csv
from matplotlib import pyplot as plt

os.chdir("..")
path = os.getcwd() + '/result/last_runs/last_runs_04-05-2023_17-21-42.json'

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
proc_in_time = 0
priorities = [0,1]
urgencies = [0,1]
lenghts_low_load = [21.4, 22.0,26.4,28.1]
load_constant = 40
product_types = ["Produkt A","Produkt B","Produkt C","Produkt D"]
throughput_times = []
throughput_times_avg = []
throughput_time_avg_k = 0
moving_average = 100
for m in priorities:
    for u in urgencies:
        prio_data = []
        throughput_times_avg = None
        throughput_times_avg = []
        proc_in_time = 0
        priority, tardiness, lateness, completion_time, throughput_time, time_to_EDD = 0,0,0,0,0,0
        for i in range(len(data["orders"][0]["orders"])):
            proc_in_time = 0
            type = data["orders"][0]["orders"][i]["type"]
            if type == "Produkt A":
                length = lenghts_low_load[0] + load_constant
            elif type == "Produkt B":
                length = lenghts_low_load[1] + load_constant
            elif type == "Produkt C":
                length = lenghts_low_load[2] + load_constant
            elif type == "Produkt D":
                length = lenghts_low_load[3] + load_constant
            if u == 1:
                length = length + 45
            
            priority = data["orders"][0]["orders"][i]["priority"]
            lateness = data["orders"][0]["orders"][i]["item_results"]["lateness"]
            completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
            throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
            tardiness = data["orders"][0]["orders"][i]["start"]  + throughput_time - length - data["orders"][0]["orders"][i]["due_to"] 
            lateness = tardiness
            if tardiness < 0:
                tardiness = 0
                proc_in_time = 1
            
            if completion_time != 0:
                if data["orders"][0]["orders"][i]["priority"] == m and data["orders"][0]["orders"][i]["urgency"] == u:
                    prio_data.append([priority, tardiness,lateness, completion_time,throughput_time, time_to_EDD, proc_in_time])
                    throughput_times.append(throughput_time)
                    with open('tpt_priority_' + str(m) + '_' + str(u) + '.csv', 'a+', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([i, round(throughput_time,1), round(tardiness,1), type])

        for k in range(len(throughput_times)-100):
            throughput_time_avg_k = 0
            for j in range(moving_average):
                try:
                    throughput_time_avg_k += np.average(throughput_times[k+j])
                except:
                    pass
            throughput_times_avg.append(throughput_time_avg_k/moving_average)
        print(len(throughput_times_avg))
        throughput_times_avg