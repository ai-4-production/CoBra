import os
import json
import time
import numpy as np
path = os.getcwd() + '/result/Scenario_1_1900/1900_last_runs_02-08-2023_13-50-48.json'

numbers = open(path)
data = json.load(numbers)

# count files in directory that needs to be analyzed
json_file_count = 0 
for roots,dirs, files in os.walk(os.getcwd() + '/Benchmark/Data'):
    for file in files:
        if os.path.splitext(file)[1] == '.json':
            json_file_count += 1   

prio_data = []
priorities = [0,1,2]

for m in priorities:
    prio_data = []
    priority, tardiness, lateness, completion_time, throughput_time, time_to_EDD = 0,0,0,0,0,0
    for i in range(len(data["orders"][0]["orders"])):
        priority = data["orders"][0]["orders"][i]["priority"]
        time_to_EDD = data["orders"][0]["orders"][i]["due_to"] - data["orders"][0]["orders"][i]["start"]
        tardiness = data["orders"][0]["orders"][i]["item_results"]["tardiness"]
        lateness = data["orders"][0]["orders"][i]["item_results"]["lateness"]
        completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
        throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["production_time"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
        
        if completion_time != 0:
            if data["orders"][0]["orders"][i]["priority"] == m:
                prio_data.append([priority, tardiness,lateness, completion_time,throughput_time, time_to_EDD])

        
    
    tardiness_1, lateness_1, completion_time_1, throughput_time_1,time_to_EDD_1 = 0,0,0,0,0
    #print low/mid/high priority orders in one graph
    for n in range(len(prio_data)):
        tardiness_1 += prio_data[n][1]
        lateness_1 += prio_data[n][2]
        completion_time_1 += prio_data[n][3]
        throughput_time_1 += prio_data[n][4]
        time_to_EDD_1 += prio_data[n][5]
    tardiness_average_0 = tardiness_1/len(prio_data)
    lateness_average_0 = lateness_1/len(prio_data)
    completion_time_average_0 = completion_time_1/len(prio_data)
    throughput_time_average_0 = throughput_time_1/len(prio_data)
    time_to_EDD_average_0 = time_to_EDD_1/len(prio_data)

    print("Priority: ", m, " , orders: ", len(prio_data))
    print("tardiness_average: ", tardiness_average_0)
    print("lateness_average: ", lateness_average_0)
    print("completion_time_average: ", completion_time_average_0)
    print("throughput_time_average: ", throughput_time_average_0)
    print("time_to_EDD_average: ", time_to_EDD_average_0)
    print("________________________________________________________")


prio_data = []
tardiness_1, lateness_1, completion_time_1, throughput_time_1, time_to_EDD_1 = 0,0,0,0,0
for i in range(len(data["orders"][0]["orders"])):
        priority = data["orders"][0]["orders"][i]["priority"]
        tardiness = data["orders"][0]["orders"][i]["item_results"]["tardiness"]
        lateness = data["orders"][0]["orders"][i]["item_results"]["lateness"]
        completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
        time_to_EDD = data["orders"][0]["orders"][i]["due_to"] - data["orders"][0]["orders"][i]["start"]
        throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["production_time"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
        
        if completion_time != 0:
            prio_data.append([priority, tardiness,lateness, completion_time,throughput_time,time_to_EDD])
            
#print low/mid/high priority orders in one graph
for n in range(len(prio_data)):
    tardiness_1 += prio_data[n][1]
    lateness_1 += prio_data[n][2]
    completion_time_1 += prio_data[n][3]
    throughput_time_1 += prio_data[n][4]
    time_to_EDD_1 += prio_data[n][5]
tardiness_average_0 = tardiness_1/len(prio_data)
lateness_average_0 = lateness_1/len(prio_data)
completion_time_average_0 = completion_time_1/len(prio_data)
throughput_time_average_0 = throughput_time_1/len(prio_data)
time_to_EDD_average_0 = time_to_EDD_1/len(prio_data)

print("All orders: " , len(prio_data))
print("tardiness_average: ", tardiness_average_0)
print("lateness_average: ", lateness_average_0)
print("completion_time_average: ", completion_time_average_0)
print("throughput_time_average: ", throughput_time_average_0)
print("time_to_EDD_average_0: ", time_to_EDD_average_0)
print("________________________________________________________")