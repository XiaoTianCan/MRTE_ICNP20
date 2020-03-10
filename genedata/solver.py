from __future__ import division
from gurobipy import *

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
                ratios.append(splitRatio)
                pathVarID += 1
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
                
    return ratios, model.objVal

# mcfsolver_v3(nodeNum, linkNum, demNum, demands, demRates[TMid], linkSet, wMatrix, MAXWEIGHT)
def mcfsolver(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, mode = 1):
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
