from __future__ import division
from gurobipy import *


def mcfsolver_single_region(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, nodeRegionId, targetrid, bgFlowmap):
    # Create optimization model
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    # Create variables
    # flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}
    demId = 0
    regionDemIds = []
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            srid = nodeRegionId[src]
            trid = nodeRegionId[dst]
            if srid == targetrid and trid == targetrid:
                regionDemIds.append(demId)
                for l in range(linkNum):
                    if linkSet[l][4] == targetrid:
                        Maps[(demId, (linkSet[l][0], linkSet[l][1]))] = flowVarID
                        flowVarID += 1
                        Maps[(demId, (linkSet[l][1], linkSet[l][0]))] = flowVarID
                        flowVarID += 1
            demId += 1
    flowVarNum = flowVarID
    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")
            
    for l in range(linkNum):
        i = linkSet[l][0]
        j = linkSet[l][1]
        if linkSet[l][4] != targetrid:
            continue
        sum1 = 0
        sum2 = 0
        for demId in regionDemIds:
            sum1 += flow[Maps[(demId,(i,j))]]
            sum2 += flow[Maps[(demId,(j,i))]]
        
        model.addConstr(sum1 + bgFlowmap[i][j] <= phi*linkSet[l][3])
        model.addConstr(sum2 + bgFlowmap[j][i] <= phi*linkSet[l][3])

    # print("add conservation constraints")
    demId = 0
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            srid = nodeRegionId[src]
            trid = nodeRegionId[dst]
            if srid == targetrid and trid == targetrid:
                for j in range(nodeNum):
                    sumin = 0
                    sumout = 0
                    for i in range(nodeNum):
                        if wMatrix[i][j] < MAXWEIGHT and i != j and nodeRegionId[i] == targetrid and nodeRegionId[j] == targetrid:
                            sumin += flow[Maps[(demId,(i,j))]]
                            sumout += flow[Maps[(demId,(j,i))]]
                    if j == src:
                        model.addConstr(sumin == 0)
                        model.addConstr(sumout == rates[demId])
                    elif j == dst:
                        model.addConstr(sumout == 0)
                        model.addConstr(sumin == rates[demId])
                    else:
                        model.addConstr(sumin == sumout)
            demId += 1

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        return optVal
    else:
        return 0.0



def sorsolver_bindR(nodeNum, demNum, totalPathNum, pathSet, pathRates, cMatrix, smallDemIdMap):
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    
    pathLimit = 2
    largeThr = 9
    fpath = {}
    for demId in range(demNum):
        if smallDemIdMap[demId] >= largeThr:
            fpath[demId] = []
            sum0 = 0
            for _ in range(len(pathSet[demId])):
                fpath[demId].append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "path"))
                sum0 += fpath[demId][-1]
            model.addConstr(sum0 == 1)
    fpath[-1] = []
    sum0 = 0
    for _ in range(pathLimit):
        fpath[-1].append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "path"))
        sum0 += fpath[-1][-1]
    model.addConstr(sum0 == 1)
    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")
    
    # compute flowmap
    flowmap = []
    for _ in range(nodeNum):
        flowmap.append([0.0]*nodeNum)
    pathRateId = 0
    demId = 0
    for i in range(nodeNum):
        for j in range(nodeNum):
            if i == j:
                continue
            if smallDemIdMap[demId] >= largeThr:
                action = fpath[demId]
            else:
                action = fpath[-1]
            paths = pathSet[demId]
            demId += 1
            for k in range(len(paths)):
                size = action[k] * pathRates[pathRateId]
                pathRateId += 1
                for p in range(len(paths[k])-1):
                    node1 = paths[k][p]
                    node2 = paths[k][p+1]
                    flowmap[node1][node2] += size
            
    # compute util
    for i in range(nodeNum):
        for j in range(nodeNum):
            if cMatrix[i][j] > 0:
                edgeload = flowmap[i][j]
                model.addConstr(edgeload <= phi*cMatrix[i][j])

    ###print("set objective and solve the problem")
    model.setObjective(phi, GRB.MINIMIZE)
    model.optimize()
    ratios = []
    if model.status == GRB.Status.OPTIMAL:
        ###print(model.objVal)
        pass
        # for pathVarID in range(nodeNum*pathLimit):
        #     splitRatio = fpath[pathVarID].getAttr(GRB.Attr.X)
        #     ratios.append(round(splitRatio, 4))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
                
    return ratios, model.objVal

def orisolver(nodeNum, linkNum, demNum, demands, linkSet, wMatrix, MAXWEIGHT):
    sSet = []
    tSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])

    # Create optimization model
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    # Create variables
    r = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")
    piVarNum = linkNum * linkNum
    flowVarNum = nodeNum * nodeNum * linkNum * 2
    pVarNum = nodeNum * nodeNum * linkNum
    pi = [model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "pi") for i in range(piVarNum)]
    flow = [model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f") for i in range(flowVarNum)]
    p = [model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "p") for i in range(pVarNum)]

    # Cons 0: 
    for src in range(nodeNum):
            for dst in range(nodeNum):
                if src == dst:
                    continue
    for s in range(demNum):
        if True:
            src = demands[s][0]
            dst = demands[s][1]
            for i in range(nodeNum):
                sumin = 0
                sumout = 0
                for l in range(linkNum):
                    varId = l*nodeNum*nodeNum + src*nodeNum + dst
                    if i == linkSet[l][1]:
                        sumin += flow[varId]
                        sumout += flow[varId + nodeNum*nodeNum*linkNum]
                    if i == linkSet[l][0]:
                        sumin += flow[varId + nodeNum*nodeNum*linkNum]
                        sumout += flow[varId]
                if src == i:
                    model.addConstr(sumout - sumin == 1)
                elif dst == i:
                    model.addConstr(sumout - sumin == -1)
                else:
                    model.addConstr(sumout - sumin == 0)

    # Cons 1:
    for l1 in range(linkNum):
        sum = 0
        for l2 in range(linkNum):
            sum += linkSet[l2][3] * pi[l1*linkNum + l2] # linkSet[l2][3] is the capacity of link l2
        model.addConstr(sum <= r)

    # Cons 2:
    for l in range(linkNum):
            for i in range(nodeNum):
                for j in range(nodeNum):
                    varId = l*nodeNum*nodeNum + i*nodeNum + j
                    model.addConstr((flow[varId] + flow[varId + nodeNum*nodeNum*linkNum]) / linkSet[l][3] <= p[varId])
                    # model.addConstr((flow[varId]) / linkSet[l][3] <= p[varId])
                    # model.addConstr((flow[varId + nodeNum*nodeNum*linkNum]) / linkSet[l][3] <= p[varId])

    # Cons 3:
    for l in range(linkNum):
        for e in range(linkNum):
            for i in range(nodeNum):
                model.addConstr(pi[l*linkNum + e] + p[l*nodeNum*nodeNum + i*nodeNum + linkSet[e][0]] - p[l*nodeNum*nodeNum + i*nodeNum + linkSet[e][1]] >= 0)
                model.addConstr(pi[l*linkNum + e] + p[l*nodeNum*nodeNum + i*nodeNum + linkSet[e][1]] - p[l*nodeNum*nodeNum + i*nodeNum + linkSet[e][0]] >= 0)

    # Cons 5:
    for l1 in range(linkNum):
        for l2 in range(linkNum):
            model.addConstr(pi[l1*linkNum + l2] >= 0)

    # Cons 6-7:
    for l in range(linkNum):
        for i in range(nodeNum):
            model.addConstr(p[l*nodeNum*nodeNum + i*nodeNum + i] == 0)
            for j in range(nodeNum):
                model.addConstr(p[l*nodeNum*nodeNum + i*nodeNum + j] >= 0)
    for varId in range(flowVarNum):
        model.addConstr(flow[varId] <= 1.0)


    model.setObjective(r, GRB.MINIMIZE)
    model.optimize()
    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optCost = model.objVal
        print("\nOR optCost: %f\n" % optCost)

        validateFlag = False
        if validateFlag:
            flowV = []
            for i in range(flowVarNum):
                item = flow[i].getAttr(GRB.Attr.X)
                flowV.append(item)
            #print(flowV)
            #exit()
            # print("len(flowV): %d\n" % len(flowV))
            orvalidate(nodeNum, linkNum, demNum, demands, rates, linkSet, flowV)
            print("finished!")
        soluLinkSet = []
        soluRatioSet = []
        for i in range(demNum):
            soluLinks = []
            soluRatios = []
            for j in range(nodeNum):
                for k in range(linkNum):
                    varId = k*nodeNum*nodeNum + demands[i][0]*nodeNum + demands[i][1]
                    if j == linkSet[k][0] or j == linkSet[k][1]:
                        solu1 = flow[varId].getAttr(GRB.Attr.X)
                        solu2 = flow[varId + nodeNum*nodeNum*linkNum].getAttr(GRB.Attr.X)
                        if solu1 > solu2:
                            data_in = solu1 - solu2
                            if data_in > 0.0001:
                                soluLinks.append([linkSet[k][0], linkSet[k][1]])
                                soluRatios.append(data_in)

                        else:
                            data_in = solu2 - solu1
                            if data_in > 0.0001:
                                soluLinks.append([linkSet[k][1], linkSet[k][0]])
                                soluRatios.append(data_in)
            soluLinkSet.append(soluLinks)
            soluRatioSet.append(soluRatios)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
    pathSet, routing = parsePaths(soluLinkSet, soluRatioSet, nodeNum, demNum, demands, MAXWEIGHT)
    return pathSet, routing

def find_path(src, dst, onepath, subgraph, nodeNum, MAXWEIGHT):
    tmp = []
    onepath.append(src)
    if src == dst:
        return onepath
    for u in range(nodeNum):
        if subgraph[src][u] != MAXWEIGHT:
            break

    if u >= nodeNum:
        return tmp
    for u in range(nodeNum):
        if subgraph[src][u] != MAXWEIGHT:
            if u not in onepath:
                tmp = find_path(u, dst, onepath, subgraph, nodeNum, MAXWEIGHT)
                if tmp != []:
                    return tmp
    return tmp

def parsePaths(soluLinkSet, soluRatioSet, nodeNum, demNum, demands, MAXWEIGHT, rates = []):
    pathSet = []
    routing = []
    for demId in range(demNum):
        if rates != [] and rates[demId] <= 0.0001:
            continue
        demPaths = []
        src = demands[demId][0]
        dst = demands[demId][1]

        subgraph = [[MAXWEIGHT]*nodeNum for i in range(nodeNum)]
        ratioMatrix = [([0.0]*nodeNum) for i in range(nodeNum)]
        graphNotMaxNum = len(soluLinkSet[demId])
        for i in range(graphNotMaxNum):
            subgraph[soluLinkSet[demId][i][0]][soluLinkSet[demId][i][1]] = 1
            ratioMatrix[soluLinkSet[demId][i][0]][soluLinkSet[demId][i][1]] = soluRatioSet[demId][i]
        while graphNotMaxNum > 0:
            onepath = []
            onepath = find_path(src, dst, onepath, subgraph, nodeNum, MAXWEIGHT)
            if onepath == []:
                break
            minweight = 2.0
            pathlen = len(onepath)
            for i in range(pathlen - 1):
                if ratioMatrix[onepath[i]][onepath[i + 1]] < minweight:
                    minweight = ratioMatrix[onepath[i]][onepath[i + 1]]

            for i in range(pathlen - 1):
                ratioMatrix[onepath[i]][onepath[i + 1]] -= minweight
                if ratioMatrix[onepath[i]][onepath[i + 1]] < 0.001:
                    ratioMatrix[onepath[i]][onepath[i + 1]] = 0
                    subgraph[onepath[i]][onepath[i + 1]] = MAXWEIGHT
                    graphNotMaxNum -= 1

            if minweight < 0.001:
                continue
            demPaths.append(onepath)
            
            if minweight > 1.01:
                print("error\n\n")
            routing.append(minweight)
        pathSet.append(demPaths)
    return pathSet, routing

# semi-oblivious routing; OR with path limitations
def sorsolver(nodeNum, demNum, totalPathNum, pathSet, pathRates, cMatrix, mode = 1):
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    #####print("generate path variable constraints")

    pathVarNum = totalPathNum
    Maps = [[[] for i in range(nodeNum)] for i in range(nodeNum)]
    pathVarID = 0
    for i in range(demNum):
        for j in range(len(pathSet[i])):
            pathlen = len(pathSet[i][j])
            for k in range(pathlen - 1):
                Maps[pathSet[i][j][k]][pathSet[i][j][k + 1]].append(pathVarID)
            pathVarID += 1

    ###print("add capacaity and objective constraints")
    sum0 = 0
    sum1 = 0
    sum2 = 0
    fpath = []
    for i in range(pathVarNum):
        fpath.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "path"))
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    pathVarID = 0
    for h in range(demNum):
        sum0 = 0
        sum1 = 0
        for k in range(len(pathSet[h])):
            sum0 += fpath[pathVarID]
            sum1 = fpath[pathVarID]
            pathVarID += 1
            model.addConstr(sum1 >= 0)
        model.addConstr(sum0 == 1)
    
    if mode == 0:
        for i in range(nodeNum):
            for j in range(i, nodeNum): # modified by gn in 2019.4.24 for computing util in undirected graph
                Maps[i][j] += Maps[j][i] # added by gn in 2019.4.24 for computing util in undirected graph
                tmp = len(Maps[i][j])
                if tmp == 0:
                    continue
                sum2 = 0
                for k in range(tmp):
                    sum2 += fpath[Maps[i][j][k]] * pathRates[Maps[i][j][k]]
                
                model.addConstr(sum2 <= phi*cMatrix[i][j])
    else:
        for i in range(nodeNum):
            for j in range(nodeNum):
                tmp = len(Maps[i][j])
                if tmp == 0:
                    continue
                sum2 = 0
                for k in range(tmp):
                    sum2 += fpath[Maps[i][j][k]] * pathRates[Maps[i][j][k]]
                
                model.addConstr(sum2 <= phi*cMatrix[i][j])

    ###print("set objective and solve the problem")
    model.setObjective(phi, GRB.MINIMIZE)
    model.optimize()
    ratios = []
    if model.status == GRB.Status.OPTIMAL:
        ###print(model.objVal)
        
        pathVarID = 0
        for h in range(demNum):
            for k in range(len(pathSet[h])):
                splitRatio = fpath[pathVarID].getAttr(GRB.Attr.X)
                ratios.append(round(splitRatio, 4))
                pathVarID += 1
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
                
    return ratios, model.objVal

# semi-oblivious routing; OR with path limitations
def sorsolver_new2(nodeNum, pathSet, TM, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId):
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    #####print("generate path variable constraints")

    pathNums = []
    for rid in range(regionNum):
        pathNums.append(sum(pathNumListReduced[rid][0]))
    # print(pathNums)

    pathSet2 = []
    demNum = 0
    totalPathNum = 0
    pathRates = []
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            if nodeRegionId[src] == nodeRegionId[dst]:
                pathSet2.append(pathSet[src][dst])
                demNum += 1
                totalPathNum += len(pathSet[src][dst])
                for _ in range(len(pathSet[src][dst])):
                    pathRates.append(TM[src][dst])
    ratios, objVal = sorsolver(nodeNum, demNum, totalPathNum, pathSet2, pathRates, cMatrix)

    return [], objVal



    fpath = []
    for rid in range(regionNum):
        fpath.append([])
        for _ in range(pathNums[rid]):
            fpath[rid].append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "path"))
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    ###print("add capacaity and objective constraints")
    for rid in range(regionNum):
        pathVarID = 0
        pathNumList = pathNumListReduced[rid][0]
        for pathNum in pathNumList:
            sumw = 0
            for _ in range(pathNum):
                sumw += fpath[rid][pathVarID]
                pathVarID += 1
            model.addConstr(sumw == 1)

    actionmatrix = []
    for src in range(nodeNum):
        actionmatrix.append([])
        for dst in range(nodeNum):
            actionmatrix[src].append([])
    
    actCountList = [0]*regionNum
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            sRegion = nodeRegionId[src]
            tRegion = nodeRegionId[dst]
            pathNum = len(pathSet[src][dst])
            if sRegion == tRegion:
                action = fpath[sRegion][actCountList[sRegion]:actCountList[sRegion]+pathNum]
                actionmatrix[src][dst] = action
                actCountList[sRegion] += pathNum

    # compute flowmap
    flowmap = []
    for _ in range(nodeNum):
        flowmap.append([0.0]*nodeNum)
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            
            if nodeRegionId[src] == nodeRegionId[dst]:
                pathset = pathSet[src][dst]
                action = actionmatrix[src][dst]

                size = TM[src][dst]
                pathnum = len(pathset)
                for i in range(pathnum):
                    length = len(pathset[i])
                    subsize = action[i]*size
                    for j in range(length-1):
                        node1 = pathset[i][j]
                        node2 = pathset[i][j+1]
                        flowmap[node1][node2] += subsize

    # compute util
    for i in range(nodeNum):
        for j in range(nodeNum):
            if cMatrix[i][j] > 0:
                util = flowmap[i][j]/cMatrix[i][j]
                model.addConstr(util <= phi)

    ###print("set objective and solve the problem")
    model.setObjective(phi, GRB.MINIMIZE)
    model.optimize()
    ratios = []
    if model.status == GRB.Status.OPTIMAL:
        pass
        # print(model.objVal)
        
        # pathVarID = 0
        # for h in range(demNum):
        #     for k in range(len(pathSet[h])):
        #         splitRatio = fpath[pathVarID].getAttr(GRB.Attr.X)
        #         ratios.append(splitRatio)
        #         pathVarID += 1
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
                
    return ratios, model.objVal

def sorsolver_new(nodeNum, pathSet, TM, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix):
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    #####print("generate path variable constraints")
    # print(pathNumListReduced)
    # exit()
    pathNums = []
    for rid in range(regionNum):
        pathNums.append([sum(pathNumListReduced[rid][0]), sum(pathNumListReduced[rid][1])])
    # pathVarNum = sum([sum(item) for item in pathNums])
    # print(pathNums)
    fpath = []
    for rid in range(regionNum):
        fpath.append([])
        for i in range(2):
            fpath[rid].append([])
            for _ in range(pathNums[rid][i]):
                fpath[rid][i].append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "path"))
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    ###print("add capacaity and objective constraints")
    for rid in range(regionNum):
        for i in range(2):
            pathVarID = 0
            pathNumList = pathNumListReduced[rid][i]
            for pathNum in pathNumList:
                sumw = 0
                for _ in range(pathNum):
                    sumw += fpath[rid][i][pathVarID]
                    pathVarID += 1
                model.addConstr(sumw == 1)

    actionmatrix = []
    for src in range(nodeNum):
        actionmatrix.append([])
        for dst in range(nodeNum):
            actionmatrix[src].append([])
    if True: # MMA
        # print("hkshgjkahg")
        actionRangeMap = []
        for src in range(nodeNum):
            actionRangeMap.append([])
            for rid in range(regionNum):
                actionRangeMap[src].append([])
        actCountList = [0]*regionNum
        for src in range(nodeNum):
            sRegion = nodeRegionId[src]
            for tRegion in range(regionNum):
                if sRegion == tRegion:
                    continue
                if regionrMatrix[sRegion][tRegion] != tRegion:  # region-level
                    continue
                actionRangeMap[src][tRegion] = [actCountList[sRegion], actCountList[sRegion]+pathNumMapRegion[src][tRegion]]
                actCountList[sRegion] += pathNumMapRegion[src][tRegion]
        for src in range(nodeNum):
            sRegion = nodeRegionId[src]
            for tRegion in range(regionNum):
                if sRegion == tRegion:
                    continue
                if regionrMatrix[sRegion][tRegion] != tRegion:
                    nextRegionHop = regionrMatrix[sRegion][tRegion]
                    actionRangeMap[src][tRegion] = actionRangeMap[src][nextRegionHop]
        
        actCountList = [0]*regionNum
        for src in range(nodeNum):
            for dst in range(nodeNum):
                if src == dst:
                    continue
                sRegion = nodeRegionId[src]
                tRegion = nodeRegionId[dst]
                pathNum = len(pathSet[src][dst])
                if sRegion == tRegion:
                    action = fpath[sRegion][0][actCountList[sRegion]:actCountList[sRegion]+pathNum]
                    actionmatrix[src][dst] = action
                    actCountList[sRegion] += pathNum
                else:
                    actionRange = actionRangeMap[src][tRegion]
                    action = fpath[sRegion][1][actionRange[0]:actionRange[1]]
                    actionmatrix[src][dst] = action
    for src in range(nodeNum):
        row1 = [len(item) for item in actionmatrix[src]]
        row2 = [len(item) for item in pathSet[src]]
        if row1 != row2:
            print(src)
            print(row1)
            print(row2)
    # exit()
    # compute flowmap
    flowmap = []
    for _ in range(nodeNum):
        flowmap.append([0.0]*nodeNum)
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            
            sources = [src]
            sizes = [TM[src][dst]]
            while True:
                if len(sources) == 0:
                    break
                pathset = pathSet[sources[0]][dst]
                action = actionmatrix[sources[0]][dst]

                # subsizes, gates = com_path_flow(flowmap, pathset, action, sizes[0])
                subsizes = []
                gates = []
                size = sizes[0]
                pathnum = len(pathset)
                tmp = 0
                s = pathset[0][0]
                t = pathset[0][-1]
                gates.append(t)
                for i in range(pathnum):
                    length = len(pathset[i])
                    subsize = action[i]*size
                    if t != pathset[i][-1]:
                        t = pathset[i][-1]
                        gates.append(t)
                        subsizes.append(tmp)
                        tmp = 0
                    tmp += subsize
                    for j in range(length-1):
                        node1 = pathset[i][j]
                        node2 = pathset[i][j+1]
                        flowmap[node1][node2] += subsize
                subsizes.append(tmp)

                break
                sources.pop(0)
                sizes.pop(0)
                for gwid in range(len(gates)):
                    if gates[gwid] == dst:
                        continue
                    sources.append(gates[gwid])
                    sizes.append(subsizes[gwid])

    # compute util
    for i in range(nodeNum):
        for j in range(nodeNum):
            if cMatrix[i][j] > 0:
                edgeload = flowmap[i][j]
                model.addConstr(edgeload <= phi*cMatrix[i][j])

    ###print("set objective and solve the problem")
    model.setObjective(phi, GRB.MINIMIZE)
    model.optimize()
    ratios = []
    if model.status == GRB.Status.OPTIMAL:
        pass
        # print(model.objVal)
        
        # pathVarID = 0
        # for h in range(demNum):
        #     for k in range(len(pathSet[h])):
        #         splitRatio = fpath[pathVarID].getAttr(GRB.Attr.X)
        #         ratios.append(splitRatio)
        #         pathVarID += 1
        # compute flowmap
        pathw = []
        for rid in range(regionNum):
            pathw.append([])
            for i in range(2):
                pathw[rid].append([])
                for j in range(pathNums[rid][i]):
                    pathw[rid][i].append(fpath[rid][i][j].getAttr(GRB.Attr.X))
        actionmatrix = []
        for src in range(nodeNum):
            actionmatrix.append([])
            for dst in range(nodeNum):
                actionmatrix[src].append([])
        if True: # MMA
            # print("hkshgjkahg")
            actionRangeMap = []
            for src in range(nodeNum):
                actionRangeMap.append([])
                for rid in range(regionNum):
                    actionRangeMap[src].append([])
            actCountList = [0]*regionNum
            for src in range(nodeNum):
                sRegion = nodeRegionId[src]
                for tRegion in range(regionNum):
                    if sRegion == tRegion:
                        continue
                    if regionrMatrix[sRegion][tRegion] != tRegion:  # region-level
                        continue
                    actionRangeMap[src][tRegion] = [actCountList[sRegion], actCountList[sRegion]+pathNumMapRegion[src][tRegion]]
                    actCountList[sRegion] += pathNumMapRegion[src][tRegion]
            for src in range(nodeNum):
                sRegion = nodeRegionId[src]
                for tRegion in range(regionNum):
                    if sRegion == tRegion:
                        continue
                    if regionrMatrix[sRegion][tRegion] != tRegion:
                        nextRegionHop = regionrMatrix[sRegion][tRegion]
                        actionRangeMap[src][tRegion] = actionRangeMap[src][nextRegionHop]
            
            actCountList = [0]*regionNum
            for src in range(nodeNum):
                for dst in range(nodeNum):
                    if src == dst:
                        continue
                    sRegion = nodeRegionId[src]
                    tRegion = nodeRegionId[dst]
                    pathNum = len(pathSet[src][dst])
                    if sRegion == tRegion:
                        action = pathw[sRegion][0][actCountList[sRegion]:actCountList[sRegion]+pathNum]
                        actionmatrix[src][dst] = action
                        actCountList[sRegion] += pathNum
                    else:
                        actionRange = actionRangeMap[src][tRegion]
                        action = pathw[sRegion][1][actionRange[0]:actionRange[1]]
                        actionmatrix[src][dst] = action
        flowmap = []
        for _ in range(nodeNum):
            flowmap.append([0.0]*nodeNum)
        for src in range(nodeNum):
            for dst in range(nodeNum):
                if src == dst:
                    continue
                
                sources = [src]
                sizes = [TM[src][dst]]
                while True:
                    if len(sources) == 0:
                        break
                    pathset = pathSet[sources[0]][dst]
                    action = actionmatrix[sources[0]][dst]

                    # subsizes, gates = com_path_flow(flowmap, pathset, action, sizes[0])
                    subsizes = []
                    gates = []
                    size = sizes[0]
                    pathnum = len(pathset)
                    tmp = 0
                    s = pathset[0][0]
                    t = pathset[0][-1]
                    gates.append(t)
                    for i in range(pathnum):
                        length = len(pathset[i])
                        subsize = action[i]*size
                        if t != pathset[i][-1]:
                            t = pathset[i][-1]
                            gates.append(t)
                            subsizes.append(tmp)
                            tmp = 0
                        tmp += subsize
                        for j in range(length-1):
                            node1 = pathset[i][j]
                            node2 = pathset[i][j+1]
                            flowmap[node1][node2] += subsize
                    subsizes.append(tmp)

                    # break
                    sources.pop(0)
                    sizes.pop(0)
                    for gwid in range(len(gates)):
                        if gates[gwid] == dst:
                            continue
                        sources.append(gates[gwid])
                        sizes.append(subsizes[gwid])

        # compute util
        edgeutils = []
        for i in range(nodeNum):
            for j in range(nodeNum):
                if cMatrix[i][j] > 0:
                    util = flowmap[i][j] / cMatrix[i][j]
                    edgeutils.append(util)
        print(model.objVal, max(edgeutils))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
                
    return ratios, model.objVal

# mcfsolver_v3(nodeNum, linkNum, demNum, demands, demRates[TMid], linkSet, wMatrix, MAXWEIGHT)
def mcfsolver(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, nodeRegionId, regionrMatrix, mode = 1):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]


    # Create optimization model
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}
    #for i in range(demNum):
    #    Maps.append([[0]*nodeNum for j in range(nodeNum)])

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            # Maps[k][linkSet[i][0]][linkSet[i][1]] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1

    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    # print("add capacaity and objective constraints")
    #phi = model.addVar(name="phi")

    for k in range(demNum):
        src = demands[k][0]
        dst = demands[k][1]
        if nodeRegionId[src] == nodeRegionId[dst]:
            for h in range(linkNum):
                i = linkSet[h][0]
                j = linkSet[h][1]
                if nodeRegionId[i] != nodeRegionId[src] or nodeRegionId[j] != nodeRegionId[src]:
                    model.addConstr(flow[Maps[(k,(i,j))]] == 0)
        # will this be reserved?
        if nodeRegionId[src] != nodeRegionId[dst]:
            rpath = [nodeRegionId[src]]
            nRegion = nodeRegionId[src]
            while nRegion != nodeRegionId[dst]:
                nRegion = regionrMatrix[rpath[-1]][nodeRegionId[dst]]
                rpath.append(nRegion)
            for h in range(linkNum):
                i = linkSet[h][0]
                j = linkSet[h][1]
                if nodeRegionId[i] not in rpath:
                    model.addConstr(flow[Maps[(k,(i,j))]] == 0)
                    model.addConstr(flow[Maps[(k,(j,i))]] == 0)
                elif nodeRegionId[j] not in rpath:
                    model.addConstr(flow[Maps[(k,(i,j))]] == 0)
                    model.addConstr(flow[Maps[(k,(j,i))]] == 0)
                else:
                    pass

            
    for h in range(linkNum):
        i = linkSet[h][0]
        j = linkSet[h][1]
        sum1 = 0
        sum2 = 0
        for k in range(demNum):
            sum1 += flow[Maps[(k,(i,j))]]
            sum2 += flow[Maps[(k,(j,i))]]
        if mode == 1:
            model.addConstr(sum1 <= phi*linkSet[h][3])
            model.addConstr(sum2 <= phi*linkSet[h][3])
        else:
            model.addConstr(sum1 + sum2 <= phi*linkSet[h][3])

    # print("add conservation constraints")
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(demNum):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += flow[Maps[(k,(i,j))]]
                    sumout += flow[Maps[(k,(j,i))]]
            if j == demands[k][0]:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == inflow[k][j])
            elif j == demands[k][1]:
                model.addConstr(sumout == 0)
                model.addConstr(sumin + inflow[k][j] == 0)
            else:
                model.addConstr(sumin + inflow[k][j] == sumout)

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        # get solution
        soluLinkSet = []
        soluRatioSet = []

        for k in range(demNum):
            soluLinks = []
            soluRatios = []
            if rates[k] <= 0.00001:
                soluLinkSet.append(soluLinks)
                soluRatioSet.append(soluRatios)
                continue
            for i in range(nodeNum):
                for j in range(i):
                    if wMatrix[i][j] < MAXWEIGHT:
                        solu1 = flow[Maps[(k,(i,j))]].getAttr(GRB.Attr.X)
                        solu2 = flow[Maps[(k,(j,i))]].getAttr(GRB.Attr.X)
                        if solu1 >= solu2:
                            data_in = solu1 - solu2
                            if data_in > 0.0001:
                                soluLinks.append([i, j])
                                soluRatios.append(data_in/rates[k])

                        else:
                            data_in = solu2 - solu1
                            if data_in > 0.0001:
                                soluLinks.append([j, i])
                                soluRatios.append(data_in/rates[k])
            soluLinkSet.append(soluLinks)
            soluRatioSet.append(soluRatios)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
    pathSet, routing = parsePaths(soluLinkSet, soluRatioSet, nodeNum, demNum, demands, MAXWEIGHT, rates)
    return optVal, pathSet, routing


def mcfsolver_normal(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, mode = 1):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]


    # Create optimization model
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}
    #for i in range(demNum):
    #    Maps.append([[0]*nodeNum for j in range(nodeNum)])

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            # Maps[k][linkSet[i][0]][linkSet[i][1]] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1

    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    # print("add capacaity and objective constraints")
    #phi = model.addVar(name="phi")

            
    for h in range(linkNum):
        i = linkSet[h][0]
        j = linkSet[h][1]
        sum1 = 0
        sum2 = 0
        for k in range(demNum):
            sum1 += flow[Maps[(k,(i,j))]]
            sum2 += flow[Maps[(k,(j,i))]]
        if mode == 1:
            model.addConstr(sum1 <= phi*linkSet[h][3])
            model.addConstr(sum2 <= phi*linkSet[h][3])
        else:
            model.addConstr(sum1 + sum2 <= phi*linkSet[h][3])

    # print("add conservation constraints")
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(demNum):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += flow[Maps[(k,(i,j))]]
                    sumout += flow[Maps[(k,(j,i))]]
            if j == demands[k][0]:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == inflow[k][j])
            elif j == demands[k][1]:
                model.addConstr(sumout == 0)
                model.addConstr(sumin + inflow[k][j] == 0)
            else:
                model.addConstr(sumin + inflow[k][j] == sumout)

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        # get solution
        soluLinkSet = []
        soluRatioSet = []

        for k in range(demNum):
            soluLinks = []
            soluRatios = []
            if rates[k] <= 0.00001:
                soluLinkSet.append(soluLinks)
                soluRatioSet.append(soluRatios)
                continue
            for i in range(nodeNum):
                for j in range(i):
                    if wMatrix[i][j] < MAXWEIGHT:
                        solu1 = flow[Maps[(k,(i,j))]].getAttr(GRB.Attr.X)
                        solu2 = flow[Maps[(k,(j,i))]].getAttr(GRB.Attr.X)
                        if solu1 >= solu2:
                            data_in = solu1 - solu2
                            if data_in > 0.0001:
                                soluLinks.append([i, j])
                                soluRatios.append(data_in/rates[k])

                        else:
                            data_in = solu2 - solu1
                            if data_in > 0.0001:
                                soluLinks.append([j, i])
                                soluRatios.append(data_in/rates[k])
            soluLinkSet.append(soluLinks)
            soluRatioSet.append(soluRatios)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
    pathSet, routing = parsePaths(soluLinkSet, soluRatioSet, nodeNum, demNum, demands, MAXWEIGHT, rates)
    return optVal, pathSet, routing


def halosolver(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]


    # Create optimization model
    model = Model('netflow')
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}
    #for i in range(demNum):
    #    Maps.append([[0]*nodeNum for j in range(nodeNum)])

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            # Maps[k][linkSet[i][0]][linkSet[i][1]] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1

    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    phi_tmp = 0
    for h in range(linkNum):
        i = linkSet[h][0]
        j = linkSet[h][1]
        sum1 = 0
        sum2 = 0
        for k in range(demNum):
            sum1 += flow[Maps[(k,(i,j))]]
            sum2 += flow[Maps[(k,(j,i))]]
        
        model.addConstr(sum1 <= linkSet[h][3])
        model.addConstr(sum2 <= linkSet[h][3])
        phi_tmp += sum1/(linkSet[h][3] - sum1)
        phi_tmp += sum2/(linkSet[h][3] - sum2)

    model.addQConstr(phi == phi_tmp)

    # print("add conservation constraints")
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(demNum):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += flow[Maps[(k,(i,j))]]
                    sumout += flow[Maps[(k,(j,i))]]
            if j == demands[k][0]:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == inflow[k][j])
            elif j == demands[k][1]:
                model.addConstr(sumout == 0)
                model.addConstr(sumin + inflow[k][j] == 0)
            else:
                model.addConstr(sumin + inflow[k][j] == sumout)

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal

    return optVal
