#!/usr/bin/python3
# -*- coding: UTF-8 -*-
'''
Objective: generate region files
'''
from __future__ import division
import random

def select_borders(exists, num, nodeNum):
    if len(exists) + num > nodeNum:
        print("ERROR")
        exit()
    res = []
    count = num
    while count > 0:
        rval = random.randint(1, nodeNum)
        if rval not in exists and rval not in res:
            res.append(rval)
            count -= 1
    return res

def compute_nodeid(currentId, rightBorders, leftBorders, totalNodeNum, idList):
    res = None
    if currentId in rightBorders:
        index = rightBorders.index(currentId)
        res = idList[leftBorders[index]]
    else:
        res = currentId + totalNodeNum
        for border in rightBorders:
            if currentId > border:
                res -= 1
    return res

topoList = ["HWS", "HWS", "HWS", "HWS"]
borderNum = [2]*(len(topoList)-1)
borderNodes = [[] for _ in range(len(topoList))]
leftBorderNodes = [[] for _ in range(len(topoList))]
topoNodeNums = []
nodeidMapping = []

totalNodeNum = 0
totalLinkNum = 0
wholeLinkSet = []
for topoid in range(len(topoList)):
    filein = open("../../inputs/topology/" + topoList[topoid] + "_topo.txt", 'r')
    lines = filein.readlines()
    filein.close()
    lineList = lines[0].strip().split()
    nodeNum = int(lineList[0])
    linkNum = int(lineList[1])
    topoNodeNums.append(nodeNum)

    linkSet = []
    for l in range(1, linkNum+1):
        lineList = lines[l].strip().split()
        node1 = int(lineList[0])
        node2 = int(lineList[1])
        weight = int(lineList[2])
        capacity = float(lineList[3])
        if topoList[topoid] == "Abi":
            capacity = 1000
        regionId = topoid
        linkSet.append([node1, node2, weight, capacity, regionId])
    if len(wholeLinkSet) == 0:
        wholeLinkSet += linkSet
        totalNodeNum += nodeNum
        totalLinkNum += linkNum
        idList = [i for i in range(nodeNum+1)]
        nodeidMapping.append(idList)
    else:
        # select border nodes for left region
        leftBorders = select_borders(borderNodes[topoid-1], borderNum[topoid-1], nodeNum)
        leftBorderNodes[topoid-1] += leftBorders
        borderNodes[topoid-1] += leftBorders
        # select border nodes for right region
        rightBorders = select_borders(borderNodes[topoid], borderNum[topoid-1], nodeNum)
        borderNodes[topoid] += rightBorders
        idList = [i for i in range(nodeNum+1)]
        for link in linkSet:
            node1 = link[0]
            link[0] = compute_nodeid(node1, rightBorders, leftBorders, totalNodeNum, nodeidMapping[topoid-1])
            idList[node1] = link[0]
            node2 = link[1]
            link[1] = compute_nodeid(node2, rightBorders, leftBorders, totalNodeNum, nodeidMapping[topoid-1])
            idList[node2] = link[1]
            flag = False
            for item in wholeLinkSet:
                if link[0:2] == item[0:2] or [link[1], link[0]] == item[0:2]:
                    flag = True
            if not flag:
                wholeLinkSet.append(link)
                totalLinkNum += 1
        nodeidMapping.append(idList)
        totalNodeNum += nodeNum - len(rightBorders)
    print(nodeidMapping)

print("-----------")
print("topoList", topoList)
print("topoNodeNums", topoNodeNums)
print("borderNum", borderNum)
print("borderNodes", borderNodes)
print("totalNodeNum", totalNodeNum)
print("totalLinkNum", totalLinkNum)
print("wholeLinkSet", wholeLinkSet)
print("-----------")

fileout = open("../../inputs/region/" + '-'.join(topoList) + ".txt", 'w')
fileout.write("%d %d\n" %(totalNodeNum, totalLinkNum))
for link in wholeLinkSet:
    fileout.write(' '.join(list(map(str, link))) + '\n')
for topoid in range(len(topoList)):
    fileout.write(' '.join(list(map(str, nodeidMapping[topoid]))) + '\n')
for topoid in range(len(topoList)-1):
    fileout.write(' '.join(list(map(str, leftBorderNodes[topoid]))) + '\n')
fileout.close()
