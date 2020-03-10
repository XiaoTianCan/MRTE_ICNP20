from __future__ import division
import copy
import random
import numpy as np
from solver import *

# topoList = ["HWS"]*4
# topo = '-'.join(topoList)
topo = 'googlec' # briten15r5line briten12r16grid
method = "gravNR250" # bimoSame28c gravNR250
trainLen = 400

# read topo file, store: weight matrix, capacity matrix, node direct capacity
nodeNum = 0
linkNum = 0
wMatrix = []
cMatrix = []
linkSet = []
nodeOutCapa = []
file = open("../../inputs/region/%s.txt" % (topo))
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
        cMatrix[i].append(0.0)

for i in range(1, linkNum+1):
    lineList = lines[i].strip().split()
    left = int(lineList[0]) - 1
    right = int(lineList[1]) - 1
    capa = float(lineList[3])
    weight = 1 #int(lineList[2])
    linkSet.append([left, right, weight, capa])
    wMatrix[left][right] = weight
    wMatrix[right][left] = weight
    cMatrix[left][right] = capa
    cMatrix[right][left] = capa
    nodeOutCapa[left] += capa
    nodeOutCapa[right] += capa
lineList = lines[linkNum+1].strip().split()
nodeRegionId = list(map(int, lineList))
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

def gravity(src, dst):
    if topo == "Cer-Cer":
        scale = 0.0000015
    elif topo == "Cer-Cer-Cer":
        scale = 0.0000006
    elif topo == "Abi-Abi":
        scale = 0.000007/1.5
    elif topo == "Abi-Abi-Abi":
        scale = 0.000006/2.5
    elif topo == "Abi-Abi-Abi-Abi":
        scale = 0.000004/8
    elif topo == "NSF-NSF":
        scale = 0.000006/1.5
    elif topo == "NSF-NSF-NSF":
        scale = 0.000005/2.5
    elif topo == "NSF-NSF-NSF-NSF":
        scale = 0.000003/5.5
    elif topo == "HWS-HWS":
        scale = 0.000003/1.5
    elif topo == "HWS-HWS-HWS":
        scale = 0.000002/6
    elif topo == "HWS-HWS-HWS-HWS":
        scale = 0.0000016/8
    elif topo == 'Ntt':
        scale = 0.0000015/2
    elif topo == 'Hws':
        scale = 0.0000015
    elif topo == 'Bell':
        scale = 0.0000015/3
    elif topo == "briten8r3":
        scale = 0.0000005
    elif topo == "briten25r2":
        scale = 0.00000012
    elif topo == "briten25r3":
        scale = 0.00000018
    elif topo == "briten25r4":
        scale = 0.00000003
    elif topo == "briten15r5":
        scale = 0.00000015
    elif topo == "briten15r5b":
        scale = 0.00000015
    elif topo == "briten15r5inf":
        scale = 0.00000012
    elif topo == "briten15r5infb":
        scale = 0.00000012
    elif topo == "sprint":
        scale = 0.0000005
    elif topo == "sprint2":
        scale = 0.0000005
    elif topo == "sprint3":
        scale = 0.0000005
    elif topo == "sprint3r2_topo":
        scale = 0.0000007
    elif topo == "GEA_topo":
        scale = 0.0000024
    elif topo == "HWS_topo":
        scale = 0.0000045
    elif topo == "google":
        scale = 0.00000012
    elif topo == "googleb":
        scale = 0.00000012
    elif topo == "googlec":
        scale = 0.00000012
    elif topo == "briten15r5treeb":
        scale = 0.00000010
    elif topo == "briten15r5line":
        scale = 0.000000055
    elif topo == "briten15r5loopb":
        scale = 0.00000007
    elif topo == "briten12r16grid":
        scale = 0.00000002
    elif topo == "briten12r16gridb":
        scale = 0.00000002
    elif topo == "1755":
        scale = 0.00000006
    elif topo == "1221":
        scale = 0.00000006
    elif topo == "1221b":
        scale = 0.00000006
    elif topo == "1221c":
        scale = 0.00000006
    elif topo == "1221d":
        scale = 0.00000006
    else:
        scale = 0.0000015
    res = scale*nodeOutCapa[src]*nodeOutCapa[dst]/(dist[src][dst]*dist[src][dst])
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

# print(tMatrix[0])
# print(len(demand))
# objval, _, __ = mcfsolver(nodeNum, linkNum, demNum, demands, demand, linkSet, wMatrix, 9999)
# print(objval)

# exit()

if method[:4] == "grav":
    testrate = []
    rates = []
    ttt = 0
    for i in range(400):
        res = []
        tmp = []
        for dem in demand:
            if method == "gravNR50":
                noise = random.uniform(-0.5, 0.5)
                r = dem*(1 + noise)
                if ttt == 0:
                    print("noise ratio", noise)
                    ttt = 1
            elif method == "gravNR50b":
                noise = random.uniform(-0.5, 1.0)
                r = dem*(1 + noise)
                if ttt == 0:
                    print("noise ratio b", noise)
                    ttt = 1
            elif method == "gravNR50c":
                noise = random.uniform(-1.0, 1.0)
                r = dem*(1 + noise)
                if ttt == 0:
                    print("noise ratio c", noise)
                    ttt = 1
            elif method == "gravNR100":
                noise = random.uniform(-1.0, 1.0)
                r = dem*(1 + noise)
                if ttt == 0:
                    print("noise ratio 100", noise)
                    ttt = 1
            elif method == "gravNR150":
                noise = random.uniform(-0.5, 1.0)
                r = dem*(1 + noise)
                if ttt == 0:
                    print("noise ratio [-0.5,1.0]", noise)
                    ttt = 1
            elif method == "gravNR250":
                noise = random.uniform(-0.5, 2.0)
                r = dem*(1 + noise)
                if ttt == 0:
                    print("noise ratio [-0.5,2.0]", noise)
                    ttt = 1
            elif method == "gravN30":
                noise = random.randint(0,30) - 15
                r = dem + noise
            else:
                print("error")
            if r <= 0:
                r = 0
            res.append(str(round(r, 3)))
            tmp.append(round(r, 3))
        rates.append(res)
        # print(res[0])
        testrate.append(tmp)

if method == "bimoSame28":
    rates = []
    demMean = []
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            if nodeRegionId[src] != nodeRegionId[dst]:
                demMean.append(2)
                continue
            prop = np.random.uniform(0, 1.0)
            if prop < 0.8:
                demMean.append(60)
            else:
                demMean.append(160)
    for i in range(400):
        demand = []
        demId = 0
        for src in range(nodeNum):
            for dst in range(nodeNum):
                if src == dst:
                    continue
                if nodeRegionId[src] != nodeRegionId[dst]:
                    var = 4
                else:
                    var = 80
                demSize = np.random.normal(demMean[demId], var)
                demand.append(str(max([0.0, round(demSize, 3)])))
                demId += 1
        rates.append(demand)
if method == "bimoSame28b":
    rates = []
    demMean = []
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            if nodeRegionId[src] != nodeRegionId[dst]:
                demMean.append(2)
                continue
            prop = np.random.uniform(0, 1.0)
            if prop < 0.8:
                demMean.append(60)
            else:
                demMean.append(160)
    for i in range(400):
        demand = []
        demId = 0
        for src in range(nodeNum):
            for dst in range(nodeNum):
                if src == dst:
                    continue
                if nodeRegionId[src] != nodeRegionId[dst]:
                    var = 4
                else:
                    var = 50
                demSize = np.random.normal(demMean[demId], var)
                demand.append(str(max([0.0, round(demSize, 3)])))
                demId += 1
        rates.append(demand)
if method == "bimoSame28c":
    rates = []
    demMean = []
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            if nodeRegionId[src] != nodeRegionId[dst]:
                prop = np.random.uniform(0, 1.0)
                if prop < 0.8:
                    demMean.append(2)
                else:
                    demMean.append(20)
            else:
                prop = np.random.uniform(0, 1.0)
                if prop < 0.8:
                    demMean.append(20)
                else:
                    demMean.append(80)
    for i in range(400):
        demand = []
        demId = 0
        for src in range(nodeNum):
            for dst in range(nodeNum):
                if src == dst:
                    continue
                if nodeRegionId[src] != nodeRegionId[dst]:
                    var = 10
                else:
                    var = 40
                demSize = np.random.normal(demMean[demId], var)
                demand.append(str(max([0.0, round(demSize, 3)])))
                demId += 1
        rates.append(demand)
# print(len(rates))
# print(rates[0])
# exit()
# print(testrate[0])
# objval, _, __ = mcfsolver(nodeNum, linkNum, demNum, demands, testrate[0], linkSet, wMatrix, 9999)
# print(objval)
# exit()
fileout = open("../../inputs/traffic/testset/%s_TMset_%s.txt" % (topo, method), 'w')
for item in rates:
    fileout.write(','.join(item) + '\n')
fileout.close()

fileout = open("../../inputs/traffic/trainset/%s_TMset_%s.txt" % (topo, method), 'w')
for i in range(trainLen):
    fileout.write(','.join(rates[i]) + '\n')
fileout.close()


print("Generate %s %s TM over!" % (topo, method))
