#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import xml.etree.ElementTree as ET
import numpy as np

# Europe:0, Asia:1, America: 2, Africa:3, Australia: 4, Sourth America: 5
regionAllocate = {'United Arab Emirates':1, 'Italy':0, 'United States':2, 'Slovakia':0, 'Macau':1, 'Canada':2, 'Russia':0, 'Thailand':1, 'New Zealand':0, 'Singapore':1, 'South Africa':3, 'Egypt':3, 'Sweden':0, 'Netherlands':0, 'Brunei':1, 'South Korea':1, 'Myanmar [Burma]':1, 'Romania':0, 'Norway':0, 'Poland':0, 'Spain':0, 'Nigeria':3, 'Philippines':1, 'Sri Lanka':1, 'Denmark':0, 'Portugal':0, 'Hungary':0, 'Australia':4, 'Austria':0, 'United Kingdom':0, 'Brazil':5, 'France':0, 'India':1, 'Greece':0, 'Vietnam':1, 'Malaysia':1, 'Hong Kong':1, 'Costa Rica':2, 'Indonesia':1, 'Germany':0, 'Ireland':0, 'China':1, 'Switzerland':0, 'Czech Republic':0, 'Bulgaria':0, 'Belgium':0, 'Saudi Arabia':1, 'Japan':1, 'Luxembourg':0, 'Ukraine':0, 'Mexico':2, 'Croatia':0, 'Slovenia':0, 'Estonia':0, 'Finland':0, 'Serbia':0, 'Montenegro':0}
maxRegionNum = 7
country = set()
pathPre = "/home/server/gengnan/NATE_project/inputs/"
namespace = "{http://graphml.graphdrawing.org/xmlns}"

def parse_xml(pathPre, graphName):
    print("--parse_xml()--")
    tree = ET.parse(pathPre + "zoo_xml/" + graphName + '.graphml.xml')
    root = tree.getroot()
    graph = root[-1]
    
    nodeNum = 0
    regionNodeNum = [0]*maxRegionNum
    nodeRegionId = []
    for child in graph.iter(namespace + 'node'):
        nodeId = int(child.attrib['id'])
        if nodeNum != nodeId:
            print("node id ERROR")
        nodeNum += 1
        flag = False
        for tnode in child:
            if graphName in ["DeutscheTelekom", "Bandcon", "Bellcanada", "Cogentco"]:
                if tnode.attrib['key'] == 'd31':
                    country.add(tnode.text)
                    regionId = regionAllocate[tnode.text]
                    regionNodeNum[regionId] += 1
                    nodeRegionId.append(regionId)
                    flag = True
                    break
            else:
                if tnode.attrib['key'] == 'd30':
                    country.add(tnode.text)
                    regionId = regionAllocate[tnode.text]
                    regionNodeNum[regionId] += 1
                    nodeRegionId.append(regionId)
                    flag = True
                    break
        if not flag:
            print("else")
            regionId = 6
            regionNodeNum[regionId] += 1
            nodeRegionId.append(regionId)

    linkSet = []
    for child in graph.iter(namespace + 'edge'):
        source = int(child.attrib['source'])
        target = int(child.attrib['target'])
        if [source, target] not in linkSet and [target, source] not in linkSet:
            linkSet.append([source, target])
        else:
            continue
        if source >= nodeNum or target >= nodeNum:
            print("ERROR")
            exit()

    print(nodeNum, regionNodeNum)
    print(nodeRegionId)
    return nodeNum, regionNodeNum, linkSet, nodeRegionId

def del_low_degree_node(nodeNum, regionNodeNum, linkSet, nodeRegionId, excludeRegion = [], excludeNode = []): # del node with one degree
    print("--del_low_degree_node()--")
    nodeDegree = [0]*nodeNum
    linkNum = len(linkSet)
    for i in range(linkNum):
        source = linkSet[i][0]
        target = linkSet[i][1]
        nodeDegree[source] += 1
        nodeDegree[target] += 1

    print(nodeDegree)
    delNodes = []
    for i in range(nodeNum):
        if nodeDegree[i] <= 1 or nodeRegionId[i] in excludeRegion or i in excludeNode:
            delNodes.append(i)
            for l in range(linkNum):
                source = linkSet[l][0]
                target = linkSet[l][1]
                if i == source:
                    nodeDegree[target] -= 1
                if i == target:
                    nodeDegree[source] -= 1
    flag = False
    while True:
        for i in range(nodeNum):
            if nodeDegree[i] <= 1:
                if i not in delNodes:
                    delNodes.append(i)
                    flag = True
                    for l in range(linkNum):
                        source = linkSet[l][0]
                        target = linkSet[l][1]
                        if i == source:
                            nodeDegree[target] -= 1
                        if i == target:
                            nodeDegree[source] -= 1
        if not flag:
            break
        else:
            flag = False
    print(nodeDegree)
    print(delNodes)

    offset = [0]*nodeNum
    for item in delNodes:
        for i in range(item+1, nodeNum):
            offset[i] -= 1
    newLinkSet = []
    count = 0
    for link in linkSet:
        source = link[0]
        target = link[1]
        if source in delNodes or target in delNodes:
            count += 1
            print("(%d %d)" % (source, target))
            continue
        newLinkSet.append([source+offset[source], target+offset[target]])
    newNodeRegionId = []
    count = 0
    for i in range(nodeNum):
        if i not in delNodes:
            newNodeRegionId.append(nodeRegionId[i])
            if count == 3: # Bell
                print("3--", i)
            if count == 16: #  Ntt
                print("16--", i)
            if count == 25: # Bell
                print("25--", i)
            count += 1
    print(nodeRegionId)
    newNodeNum = nodeNum - len(delNodes)
    # print("hhkshhk", newNodeRegionId[3])
    # for link in newLinkSet:
    #     if 3 in link[0:2]:
    #         print(link)

    return newNodeNum, newLinkSet, newNodeRegionId

def count_inter_link(nodeNum, linkSet, nodeRegionId):
    print("--count_inter_link()--")
    interLinkMatrix = []
    for _ in range(maxRegionNum):
        interLinkMatrix.append([0]*maxRegionNum)
    for link in linkSet:
        source = link[0]
        target = link[1]
        sRegion = nodeRegionId[source]
        tRegion = nodeRegionId[target]
        interLinkMatrix[sRegion][tRegion] += 1
        interLinkMatrix[tRegion][sRegion] += 1
    print(np.array(interLinkMatrix))
    regionNodeNum = [0]*maxRegionNum
    for i in range(nodeNum):
        regionId = nodeRegionId[i]
        regionNodeNum[regionId] += 1
    print(nodeNum, regionNodeNum)
    return regionNodeNum

def write_file(topoName, nodeNum, linkSet, nodeRegionId):
    print("--write_file()--")
    nodeDegree = [0]*nodeNum
    linkNum = len(linkSet)
    for i in range(linkNum):
        source = linkSet[i][0]
        target = linkSet[i][1]
        nodeDegree[source] += 1
        nodeDegree[target] += 1
    print(nodeDegree)

    outfile = open(pathPre + "region/" + topoName + ".txt", 'w')
    outfile.write("%d %d\n" % (nodeNum, linkNum))
    for l in range(linkNum):
        source = linkSet[l][0]
        target = linkSet[l][1]
        linkWeight = 10
        if nodeDegree[source] <= 1 or nodeDegree[target] <= 1:
            print("hhhhhhhhhh---%d %d---" % (source+1, target+1))
        if nodeDegree[source] >= 5 and nodeDegree[target] >= 5:
            capacity = 9953
        else:
            capacity = 2488
        sRegion = nodeRegionId[source]
        tRegion = nodeRegionId[target]
        if sRegion == tRegion:
            linkRegion = sRegion
        else:
            linkRegion = -1
        outfile.write("%d %d %d %d %d\n" % (source+1, target+1, linkWeight, capacity, linkRegion))
    outfile.write(' '.join(list(map(str, nodeRegionId))))
    outfile.close()

def deal_region_id(nodeNum, nodeRegionId, regionNodeNum):
    miss = []
    for i in range(maxRegionNum):
        if regionNodeNum[i] == 0:
            miss.append(i)
    offset = [0]*maxRegionNum
    for item in miss:
        for i in range(item+1, maxRegionNum):
            offset[i] -= 1
    print(offset)
    newNodeRegionId = []
    for item in nodeRegionId:
        newNodeRegionId.append(item + offset[item])
    return newNodeRegionId

#--main()--
# graphList = ["DeutscheTelekom", "Ntt", "Highwinds", "Bellcanada"]
# for graphName in graphList:
#     print("graphName:", graphName)
#     nodeNum, regionNodeNum, linkSet, nodeRegionId = parse_xml(pathPre, graphName)
#     nodeNum, linkSet, nodeRegionId = del_low_degree_node(nodeNum, regionNodeNum, linkSet, nodeRegionId)
#     count_inter_link(nodeNum, linkSet, nodeRegionId)
# exit()
# print(country)

##****************************************##
graphName = "Ntt"
print("graphName:", graphName)
nodeNum, regionNodeNum, linkSet, nodeRegionId = parse_xml(pathPre, graphName)
nodeNum, linkSet, nodeRegionId = del_low_degree_node(nodeNum, regionNodeNum, linkSet, nodeRegionId, excludeRegion = [4], excludeNode = [36])
regionNodeNum = count_inter_link(nodeNum, linkSet, nodeRegionId)
nodeRegionId = deal_region_id(nodeNum, nodeRegionId, regionNodeNum)
write_file("Ntt", nodeNum, linkSet, nodeRegionId)
exit()
##****************************************##
# graphName = "Highwinds"
# print("graphName:", graphName)
# nodeNum, regionNodeNum, linkSet, nodeRegionId = parse_xml(pathPre, graphName)
# nodeNum, linkSet, nodeRegionId = del_low_degree_node(nodeNum, regionNodeNum, linkSet, nodeRegionId, excludeRegion = [5])
# print('\n', nodeRegionId)
# regionNodeNum = count_inter_link(nodeNum, linkSet, nodeRegionId)
# nodeRegionId = deal_region_id(nodeNum, nodeRegionId, regionNodeNum)
# write_file("Hws", nodeNum, linkSet, nodeRegionId)

##****************************************##
graphName = "Bellcanada"
print("graphName:", graphName)
regionAllocate["United States"] = 6
nodeNum, regionNodeNum, linkSet, nodeRegionId = parse_xml(pathPre, graphName)
regionAllocate["United States"] = 2
nodeNum, linkSet, nodeRegionId = del_low_degree_node(nodeNum, regionNodeNum, linkSet, nodeRegionId, excludeNode = [5, 34])
print('\n', nodeRegionId)
regionNodeNum = count_inter_link(nodeNum, linkSet, nodeRegionId)
nodeRegionId = deal_region_id(nodeNum, nodeRegionId, regionNodeNum)
write_file("Bell", nodeNum, linkSet, nodeRegionId)