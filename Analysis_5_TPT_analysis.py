import os
import json
import time
import numpy as np
import csv
from matplotlib import pyplot as plt
path = os.getcwd() + '/result/last_runs_01-30-2023_10-41.json'

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
priorities = [0,1,2]
lenghts_low_load = [26.64, 21.55,13.81,18.93]
load_constant = 40
product_types = ["Produkt A","Produkt B","Produkt C","Produkt D"]
throughput_times = []
throughput_times_avg = []
throughput_time_avg_k = 0
moving_average = 100
for m in priorities:
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
            if data["orders"][0]["orders"][i]["priority"] == m:
                prio_data.append([priority, tardiness,lateness, completion_time,throughput_time, time_to_EDD, proc_in_time])
                throughput_times.append(throughput_time)
                with open('priority_' + str(m) + '.csv', 'a+', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([i, throughput_time, tardiness])
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
    

    tardiness_1, lateness_1, completion_time_1, throughput_time_1,time_to_EDD_1,proc_in_time_1 = 0,0,0,0,0,0
    #print low/mid/high priority orders in one graph
    for n in range(len(prio_data)):
        tardiness_1 += prio_data[n][1] 
        lateness_1 += prio_data[n][2]
        completion_time_1 += prio_data[n][3]
        throughput_time_1 += prio_data[n][4]
        time_to_EDD_1 += prio_data[n][5]
        proc_in_time_1 += prio_data[n][6]
    tardiness_average_0 = tardiness_1/len(prio_data)
    lateness_average_0 = lateness_1/len(prio_data)
    completion_time_average_0 = completion_time_1/len(prio_data)
    throughput_time_average_0 = throughput_time_1/len(prio_data)
    time_to_EDD_average_0 = time_to_EDD_1/len(prio_data)
    proc_in_time_average_0 = proc_in_time_1/len(prio_data)


prio_data = []
length = 0
tardiness_1, lateness_1, completion_time_1, throughput_time_1, time_to_EDD_1 = 0,0,0,0,0
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
    priority = data["orders"][0]["orders"][i]["priority"]
    completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
    time_to_EDD = data["orders"][0]["orders"][i]["due_to"] - data["orders"][0]["orders"][i]["start"]
    throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
    with open('priority_total' + '.csv', 'a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([k, np.average(throughput_times[k])])
    tardiness = data["orders"][0]["orders"][i]["start"]  + throughput_time - length - data["orders"][0]["orders"][i]["due_to"] 
    lateness = tardiness
    if tardiness < 0:
            tardiness = 0
            proc_in_time = 1
    if completion_time != 0:
        prio_data.append([priority, tardiness,lateness, completion_time,throughput_time,time_to_EDD, proc_in_time])

tpt_times = []
#print low/mid/high priority orders in one graph
for n in range(len(prio_data)):
    tardiness_1 += prio_data[n][1]
    lateness_1 += prio_data[n][2]
    completion_time_1 += prio_data[n][3]
    throughput_time_1 += prio_data[n][4]
    tpt_times.append(prio_data[n][4])
    time_to_EDD_1 += prio_data[n][5]
    proc_in_time_1 += prio_data[n][6]

tardiness_average_0 = tardiness_1/len(prio_data)
lateness_average_0 = lateness_1/len(prio_data)
completion_time_average_0 = completion_time_1/len(prio_data)
throughput_time_average_0 = throughput_time_1/len(prio_data)
time_to_EDD_average_0 = time_to_EDD_1/len(prio_data)
proc_in_time_average_0 = proc_in_time_1/len(prio_data)
