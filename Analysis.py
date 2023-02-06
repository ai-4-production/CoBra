import os
import json
import time
import numpy as np
# path = os.getcwd() + '/result/last_runs_02-02-2023_16-15-29_RL_2000.json'
path = os.getcwd() + '/result/last_runs_02-02-2023_16-36-18_FiFo_local_2000.json'

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
    print(prio_data)
    tardiness, lateness, completion_time, throughput_time = 0 ,0 ,0 ,0
    for i in range(len(data["orders"][0]["orders"])):
        priority = data["orders"][0]["orders"][i]["priority"]
        tardiness = data["orders"][0]["orders"][i]["item_results"]["tardiness"]
        lateness = data["orders"][0]["orders"][i]["item_results"]["lateness"]
        completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
        throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["production_time"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
        
        if completion_time != 0:
            if priority == m:
                prio_data.append([priority, tardiness,lateness, completion_time,throughput_time])
            # if priority == 2:
            #     print(i , ", ", throughput_time)
    #print low/mid/high priority orders in one graph
    tardiness_sum, lateness_sum, completion_time_sum, throughput_time_sum = 0 ,0 ,0 ,0
    for n in range(len(prio_data)):
        tardiness_sum += prio_data[n][1]
        lateness_sum += prio_data[n][2]
        completion_time_sum += prio_data[n][3]
        throughput_time_sum += prio_data[n][4]
    tardiness_average_0 = tardiness_sum/len(prio_data)
    lateness_average_0 = lateness_sum/len(prio_data)
    completion_time_average_0 = completion_time_sum/len(prio_data)
    throughput_time_average_0 = throughput_time_sum/len(prio_data)

    print("Priority: ", m, " , orders: ", len(prio_data))
    print("tardiness_average: ", tardiness_average_0)
    print("lateness_average: ", lateness_average_0)
    # print("completion_time_average: ", completion_time_average_0)
    print("throughput_time_average: ", throughput_time_average_0)

length_completed_orders = 0
for i in range(len(data["orders"][0]["orders"])):
    if data["orders"][0]["orders"][i]["item_results"]["completion_time"] != 0:
        length_completed_orders += 1
print("total_completed_orders: ", length_completed_orders)
print("________________________________________________________")

# tardiness, lateness, completion_time, throughput_time = 0 ,0 ,0 ,0
# prio_data = []
# for i in range(len(data["orders"][0]["orders"])):
#     priority = data["orders"][0]["orders"][i]["priority"]
#     tardiness = data["orders"][0]["orders"][i]["item_results"]["tardiness"]
#     lateness = data["orders"][0]["orders"][i]["item_results"]["lateness"]
#     completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
#     throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["production_time"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
#     prio_data.append([priority, tardiness,lateness, completion_time,throughput_time])

# for n in range(len(prio_data)):
#     tardiness += prio_data[n][1]
#     lateness += prio_data[n][2]
#     completion_time += prio_data[n][3]
#     throughput_time += prio_data[n][4]
# tardiness_average_0 = tardiness/len(prio_data)
# lateness_average_0 = lateness/len(prio_data)
# completion_time_average_0 = completion_time/len(prio_data)
# throughput_time_average_0 = throughput_time/len(prio_data)

# print("Priority: ", m, " , orders: ", len(prio_data))
# print("tardiness_average: ", tardiness_average_0)
# print("lateness_average: ", lateness_average_0)
# # print("completion_time_average: ", completion_time_average_0)
# print("throughput_time_average: ", throughput_time_average_0)