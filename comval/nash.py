from __future__ import division
from topo import *
from region2 import *
import random
import queue
import copy
from numpy.linalg import solve
import numpy as np
import scipy.optimize as sco
import math, time
from collections import Counter

def get_nash_bargin_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType, iterationNum, alpha):
    # read topo and TM
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()
    regionBorderNodes = env.read_hp_info()

    # preliminary info
    regionNodeNum = Counter(nodeRegionId)
    nodeIdOffset = [0]*regionNum
    for rid in range(1, regionNum):
        nodeIdOffset[rid] = nodeIdOffset[rid-1] + regionNodeNum[rid-1]
    print(regionNodeNum)
    print(nodeIdOffset)
    regionLinkSet = [[] for _ in range(regionNum)]
    peerLinkMatrix = []
    for lrid in range(regionNum):
        peerLinkMatrix.append([])
        for rrid in range(regionNum):
            peerLinkMatrix[lrid].append([])
    for link in linkSet:
        if link[-1] != -1:
            regionLinkSet[link[-1]].append(link)
        else:
            lrid = nodeRegionId[link[0]]
            rrid = nodeRegionId[link[1]]
            peerLinkMatrix[lrid][rrid].append(link)
            peerLinkMatrix[rrid][lrid].append(link)

    peerNumList = [[] for _ in range(regionNum)]
    peerFlagList = [[] for _ in range(regionNum)]
    for lrid in range(regionNum):
        for rrid in range(regionNum):
            if len(peerLinkMatrix[lrid][rrid]) > 0:
                if lrid < rrid:
                    peerFlagList[lrid] += [1]*len(peerLinkMatrix[lrid][rrid])
                else:
                    peerFlagList[lrid] += [-1]*len(peerLinkMatrix[lrid][rrid])
            peerNumList[lrid].append(len(peerLinkMatrix[lrid][rrid]))
            for link in peerLinkMatrix[lrid][rrid]:
                regionLinkSet[lrid].append(link)
    print(np.array(peerNumList))
    # print(peerFlagList)
    # exit()
    # prepare incidence maxtrix
    A = []
    for rid in range(regionNum):
        A.append([])
        for i in range(regionNodeNum[rid]):
            A[rid].append([0]*(len(regionLinkSet[rid])*2))
        # print(len(A[rid][0]))
        for j in range(len(regionLinkSet[rid])):
            left = regionLinkSet[rid][j][0]
            lrid = nodeRegionId[left]
            left = left - nodeIdOffset[lrid]
            right = regionLinkSet[rid][j][1]
            rrid = nodeRegionId[right]
            right = right - nodeIdOffset[rrid]
            
            if lrid == rid:
                A[rid][left][j*2] = 1
                A[rid][left][j*2+1] = -1
            if rrid == rid:
                A[rid][right][j*2] = -1
                A[rid][right][j*2+1] = 1
        # break
    # print(np.array(A[1]))
    # exit()
    
    for TMid in range(totalTMNum):
        S = []
        for rid in range(regionNum):
            S.append([])
            for i in range(nodeNum):
                S[rid].append([0]*regionNodeNum[rid])
        
        demId = 0
        for i in range(nodeNum):
            for j in range(nodeNum):
                if i == j:
                    continue
                demSize = demRates[TMid][demId]
                irid = nodeRegionId[i]
                jrid = nodeRegionId[j]
                sNodeId = i - nodeIdOffset[irid]
                tNodeId = j - nodeIdOffset[jrid]

                
                S[irid][j][sNodeId] += demSize
                S[jrid][j][tNodeId] -= demSize

                demId += 1
        # print(S[0])
        # exit()
        # iteration loop
        gradientsList = []
        for rid in range(regionNum):
            gradientsList.append([])
            for i in range(nodeNum):
                gradientsList[rid].append([0]*(2*sum(peerNumList[rid])))
        for iteration in range(iterationNum):
            print("iteration:", iteration)
            ysolution = []
            for rid in range(regionNum):
                rid = 1
                # solve subproblems & update gradients
                u, y = solve_qp(rid, regionNum, A[rid], S[rid], nodeRegionId, regionLinkSet[rid], sum(peerNumList[rid]), peerNumList, peerFlagList[rid], regionrMatrix, gradientsList[rid], alpha)
                ysolution.append(y)
                print(u, len(y))
                return
            # compute gradientsList
            for rid in range(regionNum):
                for nrid in range(regionNum):
                    if peerNumList[rid][nrid] == 0:
                        continue
                    edgepeernum = 2*sum(peerNumList[rid])
                    offset = 2*sum(peerNumList[rid][:nrid])
                    edgepeernum2 = 2*sum(peerNumList[nrid])
                    offset2 = 2*sum(peerNumList[nrid][:rid])
                    # print(rid, nrid, edgepeernum, edgepeernum2, offset, offset2)
                    for d in range(nodeNum):
                        # print("d:", d)
                        for l in range(peerNumList[rid][nrid]):
                            # print("l:", l)
                            if rid < nrid:
                                gradientsList[rid][d][offset+l*2] = ysolution[rid][d*edgepeernum+offset+l*2] - ysolution[nrid][d*edgepeernum2+offset2+l*2]
                                gradientsList[rid][d][offset+l*2 + 1] = ysolution[rid][d*edgepeernum+offset+l*2+1] - ysolution[nrid][d*edgepeernum2+offset2+l*2+1]
                            else:
                                gradientsList[rid][d][offset+l*2] = ysolution[nrid][d*edgepeernum2+offset2+l*2] - ysolution[rid][d*edgepeernum+offset+l*2]
                                gradientsList[rid][d][offset+l*2 + 1] = ysolution[nrid][d*edgepeernum2+offset2+l*2+1] - ysolution[rid][d*edgepeernum+offset+l*2+1]

    # output results and save data

def solve_qp(currid, regionNum, A, S, nodeRegionId, linkset, peernum, peerNumList, peerFlag, regionrMatrix, gradients, alpha):
    print("currid:", currid)
    nodenum = len(A)
    nodeNum = len(gradients)
    linknum = len(linkset)
    intranum = linknum - peernum
    edgenum = 2*linknum
    edgepeernum = 2*peernum
    edgeintranum = 2*intranum
    def obj_func(x):
        obj = math.log(10 - x[0])
        x_var_num = nodeNum*edgeintranum
        y_var = x[1 + x_var_num : (nodeNum*edgepeernum) + 1 + x_var_num]
        lamda = x[(nodeNum*edgepeernum) + 1 + x_var_num:]
        # for d in range(nodeNum):
        #     for peer in range(edgepeernum):
        #         index = d*edgepeernum + peer
        #         obj += peerFlag[peer//2]*(lamda[index] - alpha*gradients[d][peer])*y_var[index]
        return -1 * obj

    constraints = []
    for d in range(nodeNum):
        for s in range(nodenum):
            def cons_func(x):
                equation = -1 * S[d][s]
                for l in range(edgenum):
                    if l < edgeintranum:
                        equation += A[s][l]*x[1 + d*edgeintranum + l]
                    else:
                        equation += A[s][l]*x[1 + nodeNum*edgeintranum + d*edgepeernum + l - edgeintranum]
                return equation
            cons = {'type':'eq', 'fun': cons_func}
            constraints.append(cons)

    for i in range(nodeNum*edgenum):
        cons_func = lambda x:x[1+i]
        cons = {'type':'ineq', 'fun': cons_func}
        constraints.append(cons)

    # for l in range(linknum*2):
    #     def cons_func(x):
    #         equation = x[0]*linkset[l//2][3]
    #         if linkset[l//2][-1] != -1:
    #             for d in range(nodeNum):
    #                 equation -= x[1 + d*edgeintranum + l]
    #         else:
    #             for d in range(nodeNum):
    #                 equation -= x[1 + nodeNum*edgeintranum + d*edgepeernum + l - edgeintranum]
    #         return equation
    #     cons = {'type':'ineq', 'fun': cons_func}
    #     constraints.append(cons)

    # y constraints
    # for d in range(nodeNum):
    #     drid = nodeRegionId[d]
    #     if currid == drid:
    #         for l in range(edgepeernum):
    #             cons_func = lambda x:x[1 + nodeNum*edgeintranum + d*edgepeernum + l]
    #             cons = {'type':'eq', 'fun': cons_func}
    #             constraints.append(cons)
    #     else:
    #         for nrid in range(regionNum):
    #             if peerNumList[currid][nrid] == 0:
    #                 continue
    #             if regionrMatrix[currid][drid] == nrid:
    #                 for l in range(2*peerNumList[currid][nrid]):
    #                     cons_func = lambda x:x[1 + nodeNum*edgeintranum + d*edgepeernum + 2*sum(peerNumList[currid][:nrid]) + l]
    #                     cons = {'type':'ineq', 'fun': cons_func}
    #                     constraints.append(cons)
    #             else:
    #                 for l in range(2*peerNumList[currid][nrid]):
    #                     cons_func = lambda x:x[1 + nodeNum*edgeintranum + d*edgepeernum + 2*sum(peerNumList[currid][:nrid]) + l]
    #                     cons = {'type':'eq', 'fun': cons_func}
    #                     constraints.append(cons)

    constraints = tuple(constraints)
    print("solve subproblem:", currid)
    opt = sco.minimize(fun=obj_func, x0=[0] + [0]*(nodeNum*edgeintranum + 2*nodeNum*edgepeernum), constraints=constraints)
    
    # print(opt.keys())
    print(opt['success'])
    print(opt['x'])
    print(opt['fun'])
    x = opt['x']
    print(sum(x))
    return x[0], x[1 + nodeNum*edgeintranum : nodeNum*edgepeernum + 1 + nodeNum*edgeintranum]


infilePrefix = '../../inputs/'
outfilePrefix = '../../outputs/'
start_time = time.time()
get_nash_bargin_obj_vals(infilePrefix, outfilePrefix, "google", 1, "p3_3_1", "_gravNR50c", iterationNum = 1, alpha = 0.01)
end_time = time.time()
interval = int((end_time-start_time)*1000)
timeMs = interval%1000
timeS = int(interval/1000)%60
timeMin = int((interval/1000-timeS)/60)%60
timeH = int(interval/1000)/3600
print("Running time: %dh-%dmin-%ds-%dms\n" % (timeH, timeMin, timeS, timeMs))


# def obj_func(x):
#     return math.log(x[0]*x[1])

# cons = []
# a = [[-1,0], [0, -1], [1, 1]]
# b = [-2, -2, 2]
# for i in range(len(b)):
#     def f(x):
#         y = - b[i]
#         for j in range(2):
#             y += x[j]*a[i][j]
#         return y
#     tmp = {'type':'ineq', 'fun': f}
#     cons.append(tmp)
# cons = tuple(cons)

# opt = sco.minimize(fun=obj_func,x0=[1,1],constraints=cons)
# print(opt.keys())
# print(opt['x'])
# print(opt['fun'])