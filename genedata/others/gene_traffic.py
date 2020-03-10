from __future__ import division
import copy
import random
import numpy as np
from solver import *

topo = "HWS"
method = "bimodal28var20scale10" #bimodal28sameODcyc5 "grav60bigN", bimodal28cycle5, grav20smallN, bimodal28sameODLvar
trainLen = 60
mean = 3000 # for gravExp only


# read topo file, store: weight matrix, capacity matrix, node direct capacity
nodeNum = 0
linkNum = 0
wMatrix = []
cMatrix = []
linkSet = []
nodeOutCapa = []
file = open("./topology/%s_topo.txt" % (topo))
lines = file.readlines()
file.close()
lineList = lines[0].strip().split()
nodeNum = int(lineList[0])
linkNum = int(lineList[1])
demNum = nodeNum*(nodeNum-1)
demands = []
for src in range(nodeNum):
    for dst in range(nodeNum):
        if src == dst:
            continue
        demands.append([src,dst])
for i in range(nodeNum):
    wMatrix.append([])
    cMatrix.append([])
    nodeOutCapa.append(0)
    for j in range(nodeNum):
        if i == j:
            wMatrix[i].append(0)
        else:
            wMatrix[i].append(9999)
        cMatrix[i].append(0)

for i in range(1, linkNum+1):
    lineList = lines[i].strip().split()
    left = int(lineList[0]) - 1
    right = int(lineList[1]) - 1
    capa = int(lineList[3])
    weight = int(lineList[2])
    linkSet.append([left, right, weight, capa])
    wMatrix[left][right] = weight
    wMatrix[right][left] = weight
    cMatrix[left][right] = capa
    cMatrix[right][left] = capa
    nodeOutCapa[left] += capa
    nodeOutCapa[right] += capa
# print(linkSet)
# floyd algorithm
dist = []
R = []
dist = copy.deepcopy(wMatrix)
for i in range(nodeNum):
    R.append([j for j in range(nodeNum)])

for i in range(nodeNum):
    for j in range(nodeNum):
        for k in range(nodeNum):
            if dist[j][k] > dist[j][i] + dist[i][k]:
                dist[j][k] = dist[j][i] + dist[i][k]
                R[j][k] = R[j][i]

def exp_distr(F):
    return -mean*math.log(1 - F)

def gravity_model(p_in, p_out):
    return mean*p_in*p_out

def get_one_TM():
    T_in = [exp_distr(random.uniform(0, 1)) for i in range(nodeNum)]
    T_out = [exp_distr(random.uniform(0, 1)) for i in range(nodeNum)]
    sumT_in = sum(T_in)
    sumT_out = sum(T_out)
    TM = []
    demand = []
    for i in range(nodeNum):
        TM.append([])
        for j in range(nodeNum):
            if i == j:
                TM[i].append(0.0)
            else:
                rate = gravity_model(T_in[i]/sumT_in, T_out[j]/sumT_out)
                TM[i].append(rate)
                demand.append(str(round(rate,6)))
    return TM, demand

def gravity(src, dst):
    if method[0:6] == "grav20":
        scale = 0.0000005*10*1.451
    elif method[0:6] == "grav60":
        scale = 0.0000005*10*1.451*3
    else:
        # print("scale = 0.0000005*10")
        scale = 0.0000005*10
    res = scale*nodeOutCapa[src]*nodeOutCapa[dst]/(dist[src][dst]*dist[src][dst])
    # if res < 1.0:
    #     res = 1.0
    return res

tMatrix = []
demand = []
for i in range(nodeNum):
    tMatrix.append([])
    for j in range(nodeNum):
        if i == j:
            tMatrix[i].append(0.0)
        else:
            traffic = gravity(i, j)
            tMatrix[i].append(traffic)
            demand.append(round(traffic, 2))

# print(demand)
# objval, _, __ = mcfsolver(nodeNum, linkNum, demNum, demands, demand, linkSet, wMatrix, 9999)
# print(objval)

# exit()
rates = []
if method in ["grav20", "grav30", "grav60", "grav30largeN2", "grav30smallN", "grav30largeN"]:
    for i in range(400):
        res = []
        for dem in demand:
            if method == "grav60bigN":
                noise = random.randint(0,30) - 15
            elif method == "grav30smallN":
                noise = random.randint(0,6) - 3
            elif method == "grav30largeN":
                noise = random.randint(0,20) - 10
            elif method == "grav30largeN2":
                noise = random.randint(0,30) - 15
            else:
                noise = random.randint(0,10) - 5
            r = dem + noise
            if r <= 0:
                r = 0
            res.append(str(round(r, 3)))
        rates.append(res)
if method == "gravExp":
    for i in range(400):
        _, demand = get_one_TM()
        rates.append(demand)
if method == "bimodal28var20scale10":
    for i in range(400):
        demand = []
        for _ in range(demNum):
            prop = np.random.uniform(0, 1.0)
            if prop < 0.8:
                demSize = np.random.normal(30, 20)
            else:
                demSize = np.random.normal(80, 20)
            demand.append(str(max([0.0, round(demSize, 3)])/10.0))
        rates.append(demand)

if method == "bimodal28cycle5":
    ratesTmp = []
    for i in range(5):
        demand = []
        for _ in range(demNum):
            prop = np.random.uniform(0, 1.0)
            if prop < 0.8:
                demSize = np.random.normal(30, 20)
            else:
                demSize = np.random.normal(80, 20)
            demand.append(str(max([0.0, round(demSize, 3)])/10.0))
        ratesTmp.append(demand)
    for i in range(80):
        for j in range(5):
            rates.append(ratesTmp[j])

if method == "bimodal28sameOD" or method == "bimodal28sameODLvar" or method == "bimodal28sameODvar10":
    demMean = []
    for _ in range(demNum):
        prop = np.random.uniform(0, 1.0)
        if prop < 0.8:
            demMean.append(30)
        else:
            demMean.append(80)
    for i in range(400):
        demand = []
        for j in range(demNum):
            demSize = np.random.normal(demMean[j], 10)
            demand.append(str(max([0.0, round(demSize, 3)])/10.0))
        rates.append(demand)
if method == "bimodal28sameODvar20":
    demMean = []
    for _ in range(demNum):
        prop = np.random.uniform(0, 1.0)
        if prop < 0.8:
            demMean.append(30)
        else:
            demMean.append(80)
    for i in range(400):
        demand = []
        for j in range(demNum):
            demSize = np.random.normal(demMean[j], 20)
            demand.append(str(max([0.0, round(demSize, 3)])/10.0))
        rates.append(demand)

if method == "bimodal28sameODcyc5" or method == "bimodal28sameODcyc5var20drlte":
    demMean = []
    for _ in range(demNum):
        prop = np.random.uniform(0, 1.0)
        if prop < 0.8:
            demMean.append(30)
        else:
            demMean.append(80)
    ratesTmp = []
    for i in range(5):
        demand = []
        for j in range(demNum):
            demSize = np.random.normal(demMean[j], 20)
            demand.append(str(max([0.0, round(demSize, 3)])/10.0))
        ratesTmp.append(demand)
    for i in range(80):
        for j in range(5):
            rates.append(ratesTmp[j])

# print(rates[0])
# objval, _, __ = mcfsolver(nodeNum, linkNum, demNum, demands, rates[0], linkSet, wMatrix, 9999)
# print(objval)
# exit()
fileout = open("./traffic/original/%s_TMset_%s.txt" % (topo, method), 'w')
for item in rates:
    fileout.write(','.join(item) + '\n')
fileout.close()

fileout = open("./traffic/fitting/%s_TMset_%s.txt" % (topo, method), 'w')
for i in range(trainLen):
    fileout.write(','.join(rates[i]) + '\n')
fileout.close()


print("Generate %s TM over!" % (method))



# for period in range(10):
#     for i in range(100): # scale factor
#         for j in range(2): # noise number
#             res = []
#             for dem in demand:
#                 r = dem*(1 + i*0.02) + random.randint(0,10) - 5
#                 if r <= 0:
#                     r = 0
#                 res.append(str(round(r, 3)))
#             fileout.write(','.join(res) + '\n')
# fileout.close()