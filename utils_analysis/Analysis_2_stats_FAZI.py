import os
import json
import time
import numpy as np
# path = os.getcwd() + '/result/FAZI_transfer/last_runs_03-01-2023_11-09-58_RL.json'
path = os.getcwd() + '/result/FAZI_transfer/last_runs_03-01-2023_11-13-36_FiFo_Local.json'
#path = os.getcwd() + '/result/Scenario_1_2800/last_runs_02-11-2023_16-10-33_HP.json'
#path = os.getcwd() + '/result/last_runs_02-15-2023_18-17-04.json'

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
load_constant = 20
product_types = ["Produkt A","Produkt B","Produkt C","Produkt D"]

for m in priorities:
    prio_data = []
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
        start = data["orders"][0]["orders"][i]["start"]
        tardiness = data["orders"][0]["orders"][i]["start"]  + throughput_time - length - data["orders"][0]["orders"][i]["due_to"] 
        lateness = tardiness
        if tardiness < 0:
            tardiness = 0
            proc_in_time = 1
        if completion_time != 0 and (start + throughput_time) < 300:
            if data["orders"][0]["orders"][i]["priority"] == m:
                prio_data.append([priority, tardiness,lateness, completion_time,throughput_time, time_to_EDD, proc_in_time])

        
    
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

    print(m) 
    print(len(prio_data))
    print(tardiness_average_0)
    print(lateness_average_0)
    print(completion_time_average_0)
    print(throughput_time_average_0)
    print(proc_in_time_average_0)
    print(proc_in_time_1)
    print("_____________")

prio_data_total = []
length = 0
tardiness_1, lateness_1, completion_time_1, throughput_time_1, time_to_EDD_1 = 0,0,0,0,0
start, priority, tardiness, lateness, completion_time, throughput_time, time_to_EDD = 0,0,0,0,0,0,0

for i in range(len(data["orders"][0]["orders"])):
    proc_in_time = 0
    type = data["orders"][0]["orders"][i]["type"]
    if type == "Produkt A" or type == "Produkt D":
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
    tardiness = data["orders"][0]["orders"][i]["start"]  + throughput_time - length - data["orders"][0]["orders"][i]["due_to"] 
    start = data["orders"][0]["orders"][i]["start"]
    lateness = tardiness
    if tardiness < 0:
            tardiness = 0
            proc_in_time = 1
    if completion_time != 0 and (start + throughput_time) < 300:
        prio_data_total.append([priority, tardiness,lateness, completion_time,throughput_time,time_to_EDD, proc_in_time])

tpt_times = []
#print low/mid/high priority orders in one graph
for n in range(len(prio_data_total)):
    tardiness_1 += prio_data_total[n][1]
    lateness_1 += prio_data_total[n][2]
    completion_time_1 += prio_data_total[n][3]
    throughput_time_1 += prio_data_total[n][4]
    #tpt_times.append(prio_data_total[n][4])
    time_to_EDD_1 += prio_data_total[n][5]
    proc_in_time_1 += prio_data_total[n][6]

tardiness_average_0 = tardiness_1/len(prio_data_total)
lateness_average_0 = lateness_1/len(prio_data_total)
completion_time_average_0 = completion_time_1/len(prio_data_total)
throughput_time_average_0 = throughput_time_1/len(prio_data_total)
time_to_EDD_average_0 = time_to_EDD_1/len(prio_data_total)
proc_in_time_average_0 = proc_in_time_1/len(prio_data_total)


# print("All orders: " , len(prio_data_total))
# print("tardiness_average: ", tardiness_average_0)
# print("lateness_average: ", lateness_average_0)
# print("completion_time_average: ", completion_time_average_0)
# print("throughput_time_average: ", throughput_time_average_0)
# print("time_to_EDD_average_0: ", time_to_EDD_average_0)
# print("proc_in_time: ", proc_in_time_average_0, ", count: ", proc_in_time_1)
# print("________________________________________________________")
print("all")
print(len(prio_data_total))
print(tardiness_average_0)
print(lateness_average_0)
print(completion_time_average_0)
print(throughput_time_average_0)
print(proc_in_time_average_0)
print(proc_in_time_1)
print("_")