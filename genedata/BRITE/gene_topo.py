import random

# regionNum = 3
# nodeNumList = [8, 8, 8]
regionNum = 5
nodeNumList = [10, 12, 11, 12, 14]
aveNode = 15
NL = 2
## tree
# regionLinkSet = [
#     [0, 2], [1, 2], [2, 3], [3, 4]
# ]
## loop
# regionLinkSet = [
#     [0, 1], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]
# ]
## line
regionLinkSet = [
    [0, 1], [1, 2], [2, 3], [3, 4]
]

totalNodeNum = sum(nodeNumList)
totalLinkNum = 0
linkSetList = []
for rid in range(regionNum):
    filein = open("./res/topo_n%d_m%d_r%d.txt" % (aveNode, NL, rid))
    lines = filein.readlines()
    filein.close()

    nodeNum = nodeNumList[rid]
    linkNum = len(lines)
    totalLinkNum += linkNum
    linkSet = []
    nodeDegree = [0]*nodeNum
    for line in lines:
        lineList = line.strip().split()
        node1 = int(lineList[1])
        node2 = int(lineList[2])
        weight = 10
        capacity = 1000
        linkSet.append([node1, node2, weight, capacity, rid])
        nodeDegree[node1] += 1
        nodeDegree[node2] += 1
    for l in range(linkNum):
        node1 = linkSet[l][0]
        node2 = linkSet[l][1]
        if nodeDegree[node1] >= 5 and nodeDegree[node2] >= 5:
            linkSet[l][3] = 9953
        else:
            linkSet[l][3] = 2488
    print(nodeDegree)
    print(linkSet)
    linkSetList.append(linkSet)

    # fileout = open('./res/topo%d_%d_%d.txt' % (nodeNumList[rid], NL, rid), 'w')
    # fileout.write("%d %d\n" % (nodeNum, linkNum))
    # for link in linkSet:
    #     fileout.write(' '.join(list(map(str, link))) + '\n')
    # fileout.close()
    
gwNum = [2, 4]
topoLinkSet = []
nodeDegree = [0]*totalNodeNum
for rid in range(regionNum):
    for l in range(len(linkSetList[rid])):
        link = linkSetList[rid][l]
        link[0] = sum(nodeNumList[0:rid]) + link[0]
        link[1] = sum(nodeNumList[0:rid]) + link[1]
        topoLinkSet.append(link)
        nodeDegree[link[0]] += 1
        nodeDegree[link[1]] += 1
# region-level topo network
outNodeId = [[] for _ in range(regionNum)]
for rlink in regionLinkSet:
    sRid = rlink[0]
    tRid = rlink[1]
    gwnum = random.randint(gwNum[0], gwNum[1])
    gwNode = [[], []]
    count = 0
    while count < gwnum:
        nodeId = random.randint(sum(nodeNumList[0:sRid]), sum(nodeNumList[0:sRid+1])-1)
        if nodeId not in gwNode[0]: # outNodeId[sRid]:
            outNodeId[sRid].append(nodeId)
            gwNode[0].append(nodeId)
            count += 1
    count = 0
    while count < gwnum:
        nodeId = random.randint(sum(nodeNumList[0:tRid]), sum(nodeNumList[0:tRid+1])-1)
        if nodeId not in gwNode[1]: # outNodeId[tRid]:
            outNodeId[tRid].append(nodeId)
            gwNode[1].append(nodeId)
            count += 1
    for i in range(gwnum):
        topoLinkSet.append([gwNode[0][i], gwNode[1][i], 10, 1000, -1])
        nodeDegree[gwNode[0][i]] += 1
        nodeDegree[gwNode[1][i]] += 1
        totalLinkNum += 1

# region-level network, inter link
for l in range(totalLinkNum):
    node1 = topoLinkSet[l][0]
    node2 = topoLinkSet[l][1]
    if nodeDegree[node1] >= 4 or nodeDegree[node2] >= 4:
        topoLinkSet[l][3] = 10000
    elif topoLinkSet[l][4] == -1:
        topoLinkSet[l][3] = 10000
    else:
        topoLinkSet[l][3] = 5000

fileout = open('./res/topo_n%d_m%d_r%d.txt' % (aveNode, NL, regionNum), 'w')
fileout.write("%d %d\n" % (totalNodeNum, totalLinkNum))
for link in topoLinkSet:
    link[0] += 1
    link[1] += 1
    fileout.write(' '.join(list(map(str, link))) + '\n')
nodeRegionId = []
for rid in range(regionNum):
    nodeRegionId += [rid]*nodeNumList[rid]
fileout.write(' '.join(list(map(str, nodeRegionId))) + '\n')
for rlink in regionLinkSet:
    fileout.write("%d %d\n" % (rlink[0], rlink[1]))
fileout.close()