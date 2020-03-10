#coding=utf-8
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# get the raw data
topoName = "Cer"
day_num = 3
start_day = 0
synthesis_type = "_exp"

if topoName == "GEA":
    day_step = 96
    sample_setp = 12
else:
    day_step = 288
    sample_setp = 36

input_file = open("traffic/original/" + topoName + "_TMset%s.txt" % (synthesis_type), "r")

data = []
for line in input_file.readlines():
    rates = line.strip().split(',')
    rates = list(map(float, rates))
    data.append(rates)

input_file.close()
data = np.array(data[day_step * (0 + start_day): day_step * (day_num + start_day)])
dataT = data.T
print(dataT.shape)

data_fit15T = []
for i in range(dataT.shape[0]):
    fitfunc = np.poly1d(np.polyfit(range(dataT.shape[1]), dataT[i], 15))
    line = fitfunc(range(dataT.shape[1]))
    data_fit15T.append(line)
data_fit15T = np.array(data_fit15T)
data_fit15 = data_fit15T.T
data_fit15_final = np.maximum(0, data_fit15)

fmt_data = np.around(data_fit15_final, 2)
out_file = open("traffic/fitting/" + topoName + "_TMset%s.txt" % (synthesis_type), "w")

for i in range(0, fmt_data.shape[0], sample_setp):
    line = ','.join(map(str, list(fmt_data[i])))
    out_file.write(line + "\n")
out_file.close()


# plot
if not os.path.exists("./data_plot"):
    os.mkdir("data_plot")

for i in range(dataT.shape[0]):
    plt.plot(list(np.arange(0, dataT.shape[1])), dataT[i], linewidth=0.5)
plt.xlabel("Time")
plt.ylabel("Traffic Amount (Mbps)")
plt.savefig("./data_plot/" + topoName + "_" + str(day_num) + "day_origin%s.png" % (synthesis_type))
plt.clf()

for i in range(dataT.shape[0]):
    fitfunc = np.poly1d(np.polyfit(range(dataT.shape[1]), dataT[i], 15))
    line = fitfunc(range(dataT.shape[1]))
    plt.plot(list(np.arange(0, dataT.shape[1])), list(np.maximum(0, line)), linewidth=0.5)

plt.xlabel("Time")
plt.ylabel("Traffic Amount (Mbps)")
plt.savefig("./data_plot/" + topoName + "_" + str(day_num) + "day_fitting%s.png" % (synthesis_type))
plt.clf()


'''
out_file = open("CER_OBL_3_0_864_fit15_filtered_smallrate-24.txt", "w")
data_count = 0
for i in content:
    out_file.write(i.strip() + "\n")
for i in range(fmt_data.shape[0]):
    list_data = list(fmt_data[i])
    smallsess_count = 0
    for j in list_data:
        if j < 500:
            smallsess_count += 1
    if smallsess_count >= 24:
        data_count += 1
        continue
    line = ','.join(map(str, list(fmt_data[i])))
    out_file.write(line + "\n")
out_file.close()
print("filtered data count: %d" % (data_count))
'''