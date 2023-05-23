import os
import json
import time
import numpy as np
import csv
from matplotlib import pyplot as plt
# os.chdir("..")
path = os.getcwd() + '/result/P_3/last_runs_05-23-2023_10-08-19.json'

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
for m in product_types:
    prio_data = []
    throughput_times = []
    proc_in_time = 0
    priority, tardiness, lateness, completion_time, throughput_time, time_to_EDD = 0,0,0,0,0,0
    for i in range(len(data["orders"][0]["orders"])):
        proc_in_time = 0
        type = data["orders"][0]["orders"][i]["type"]
        
        completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
        throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
        
        if completion_time != 0:
            if type == m:
                prio_data.append([throughput_time])
                throughput_times.append(throughput_time)
            
            
    print(m, round(np.mean(throughput_times),1))
        

    