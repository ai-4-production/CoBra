import os
import json
import time
import numpy as np
path = os.getcwd() + '/result/Scenario_1_2800/last_runs_02-11-2023_16-10-02_EDD.json'

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
lenghts = [26.64, 21.55,13.81,18.93]
product_types = ["Produkt A","Produkt B","Produkt C","Produkt D"]

for m in priorities:
    prio_data = []
    priority, tardiness, lateness, completion_time, throughput_time, time_to_EDD = 0,0,0,0,0,0
    for i in range(len(data["orders"][0]["orders"])):
        type = data["orders"][0]["orders"][i]["type"]
        # if type == "Produkt A":
        #     length = 56.64
        # elif type == "Produkt B":
        #     length = 51.55
        # elif type == "Produkt C":
        #     length = 53.81
        # elif type == "Produkt D":
        #     length = 58.93
        if type == "Produkt A":
            length = 26.64
        elif type == "Produkt B":
            length = 21.55
        elif type == "Produkt C":
            length = 13.81
        elif type == "Produkt D":
            length = 18.93
        priority = data["orders"][0]["orders"][i]["priority"]
        lateness = data["orders"][0]["orders"][i]["item_results"]["lateness"]
        completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
        throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
        tardiness = data["orders"][0]["orders"][i]["start"]  + throughput_time - length - data["orders"][0]["orders"][i]["due_to"] 
        lateness = tardiness
        if tardiness < 0:
            tardiness = 0
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
length = 0
tardiness_1, lateness_1, completion_time_1, throughput_time_1, time_to_EDD_1 = 0,0,0,0,0
priority, tardiness, lateness, completion_time, throughput_time, time_to_EDD = 0,0,0,0,0,0

for i in range(len(data["orders"][0]["orders"])):
    type = data["orders"][0]["orders"][i]["type"]
    if type == "Produkt A":
        length = 26.64
    elif type == "Produkt B":
        length = 21.55
    elif type == "Produkt C":
        length = 13.81
    elif type == "Produkt D":
        length = 18.93
    priority = data["orders"][0]["orders"][i]["priority"]
    completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
    time_to_EDD = data["orders"][0]["orders"][i]["due_to"] - data["orders"][0]["orders"][i]["start"]
    throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
    tardiness = data["orders"][0]["orders"][i]["start"]  + throughput_time - length - data["orders"][0]["orders"][i]["due_to"] 
    lateness = tardiness
    if tardiness < 0:
            tardiness = 0
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


# for p in product_types:
#     prio_data = []
#     priority, tardiness, lateness, completion_time, throughput_time, time_to_EDD = 0,0,0,0,0,0
#     for i in range(len(data["orders"][0]["orders"])):
#         type = data["orders"][0]["orders"][i]["type"]
#         if type == "Produkt A":
#             length = 20
#         elif type == "Produkt B":
#             length = 15
#         elif type == "Produkt C":
#             length = 16
#         elif type == "Produkt D":
#             length = 12
#         priority = data["orders"][0]["orders"][i]["priority"]
#         completion_time = data["orders"][0]["orders"][i]["item_results"]["completion_time"]
#         throughput_time = data["orders"][0]["orders"][i]["item_results"]["transportation_time"]+data["orders"][0]["orders"][i]["item_results"]["time_at_machines"]+data["orders"][0]["orders"][i]["item_results"]["time_in_interface_buffer"]+data["orders"][0]["orders"][i]["item_results"]["time_in_queue_buffer"]+data["orders"][0]["orders"][i]["item_results"]["wait_for_repair_time"]
#         tardiness = data["orders"][0]["orders"][i]["start"]  + throughput_time - length - data["orders"][0]["orders"][i]["due_to"] 
#         lateness = tardiness
#         if tardiness < 0:
#             tardiness = 0

#         if completion_time != 0:
#             if data["orders"][0]["orders"][i]["type"] == p:
#                 prio_data.append([priority, tardiness,lateness, completion_time,throughput_time, time_to_EDD])

        
    
#     tardiness_1, lateness_1, completion_time_1, throughput_time_1,time_to_EDD_1 = 0,0,0,0,0
#     #print low/mid/high priority orders in one graph
#     for n in range(len(prio_data)):
#         tardiness_1 += prio_data[n][1]
#         lateness_1 += prio_data[n][2]
#         completion_time_1 += prio_data[n][3]
#         throughput_time_1 += prio_data[n][4]
#         time_to_EDD_1 += prio_data[n][5]
#     tardiness_average_0 = tardiness_1/len(prio_data)
#     lateness_average_0 = lateness_1/len(prio_data)
#     completion_time_average_0 = completion_time_1/len(prio_data)
#     throughput_time_average_0 = throughput_time_1/len(prio_data)
#     time_to_EDD_average_0 = time_to_EDD_1/len(prio_data)

#     print("Type: ", p, " , orders: ", len(prio_data))
#     print("tardiness_average: ", tardiness_average_0)
#     print("lateness_average: ", lateness_average_0)
#     print("completion_time_average: ", completion_time_average_0)
#     print("throughput_time_average: ", throughput_time_average_0)
#     print("time_to_EDD_average: ", time_to_EDD_average_0)
#     print("________________________________________________________")