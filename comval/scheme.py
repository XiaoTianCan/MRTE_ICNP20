from __future__ import division
from solver import *
from solver2 import *
from topo import *
from region2 import *
from tqdm import tqdm
import random
import queue
import copy
from numpy.linalg import solve
import numpy as np
import pickle

def com_hp_pathset_failure(infilePrefix, topoName, nodeNum, pathSet, regionNum, nodeRegionId, regionrMatrix, wMatrix, regionBorderNodes, interEdgeSet):
    pathSetMatrix = {}
    disMatrixDic = {}
    for sRid in range(regionNum):
        for tRid in range(regionNum):
            wMatrix_tmp = copy.deepcopy(wMatrix)
            if sRid == tRid:
                for i in range(nodeNum):
                    for j in range(nodeNum):
                        if i == j:
                            continue
                        if nodeRegionId[i] != sRid or nodeRegionId[j] != sRid:
                            wMatrix_tmp[i][j] = 999999

                pathSet, disMatrix = com_shr_pathset(nodeNum, wMatrix_tmp)
                pathSetMatrix[(sRid, tRid)] = pathSet
                disMatrixDic[(sRid, tRid)] = disMatrix

    # get candidate path set
    oripathmaxtrix = []
    for src in range(nodeNum):
        oripathmaxtrix.append([])
        for dst in range(nodeNum):
            if src == dst:
                oripathmaxtrix[src].append([])
                continue
            sRid = nodeRegionId[src]
            tRid = nodeRegionId[dst]
            if sRid == tRid:
                path = pathSetMatrix[(sRid,tRid)][src][dst]
            else:
                src_tmp = src
                path = []
                nRid = sRid
                while nRid != tRid:
                    pRid = nRid
                    nRid = regionrMatrix[pRid][tRid]
                    if src_tmp in regionBorderNodes[pRid][nRid]:
                        path += [src_tmp]
                        outBorderNode = src_tmp
                    else:
                        pathws = []
                        for borderNode in regionBorderNodes[pRid][nRid]:
                            pathws.append(disMatrixDic[(pRid,pRid)][src_tmp][borderNode])
                        index = pathws.index(min(pathws))
                        # print(regionBorderNodes[pRid][nRid])
                        outBorderNode = regionBorderNodes[pRid][nRid][index]
                        # print(outBorderNode)
                        path += pathSetMatrix[(pRid,pRid)][src_tmp][outBorderNode]
                    for interlink in interEdgeSet[pRid][nRid]:
                        # print(interlink)
                        if interlink[0] == outBorderNode:
                            src_tmp = interlink[1]
                            break
                # print(src_tmp, dst)
                if src_tmp == dst:
                    path += [dst]
                else:
                    path += pathSetMatrix[(nRid,nRid)][src_tmp][dst]
                # print(src,dst)
                # print(path)
                # exit()
            oripathmaxtrix[src].append([path])
    return oripathmaxtrix

def get_hpmcf_obj_vals_failure(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = "", TMoffset = 40, totalDownLinkNum = 100):
    if totalTMNum > totalDownLinkNum:
        print("ERROR")
        return

    filein = open(infilePrefix + "brokenlink/" + topoName + "_%dlinks.txt" % totalDownLinkNum, 'r')
    lines = filein.readlines()
    brokenLinkList = []
    for line in lines:
        lineList = line.strip().split()
        brokenLinkList.append(list(map(int, lineList)))
    filein.close()
    totalDownLinkNum = len(brokenLinkList)

    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()

    regionBorderNodes = env.read_hp_info()
    interEdgeSet = []
    for rid in range(regionNum):
        interEdgeSet.append([])
        for _ in range(regionNum):
            interEdgeSet[rid].append([])
    for link in linkSet:
        if link[-1] == -1:
            left = link[0]
            right = link[1]
            lRegionId = nodeRegionId[left]
            rRegionId = nodeRegionId[right]
            interEdgeSet[lRegionId][rRegionId].append((left,right))
            interEdgeSet[rRegionId][lRegionId].append((right,left))

    outfileName = outfilePrefix + "objvals/" + topoName + "_hpmcf_obj_vals%s_failure.txt" % (synthesisType)
    fileout = open(outfileName, 'w', 1)
    pbar = tqdm(total=totalDownLinkNum)
    for linkId in range(totalDownLinkNum):
        brokenlink = brokenLinkList[linkId]
        TMid = TMoffset + linkId//totalTMNum

        # TODO: Test

        linkSetTmp = copy.deepcopy(linkSet)
        linkSetTmp[brokenlink[0]][3] = 0.01
        cMatrixTmp = copy.deepcopy(cMatrix)
        cMatrixTmp[brokenlink[1]][brokenlink[2]] = 0.01
        cMatrixTmp[brokenlink[2]][brokenlink[1]] = 0.01
        wMatrixTmp = copy.deepcopy(wMatrix)
        wMatrixTmp[brokenlink[1]][brokenlink[2]] = MAXWEIGHT
        wMatrixTmp[brokenlink[2]][brokenlink[1]] = MAXWEIGHT

        regionBorderNodesTmp = copy.deepcopy(regionBorderNodes)
        interEdgeSetTmp = copy.deepcopy(interEdgeSet)
        if linkSetTmp[brokenlink[0]][-1] == -1:
            lrid = nodeRegionId[brokenlink[1]]
            rrid = nodeRegionId[brokenlink[2]]
            interEdgeSetTmp[lrid][rrid].remove((brokenlink[1], brokenlink[2]))
            interEdgeSetTmp[rrid][lrid].remove((brokenlink[2], brokenlink[1]))
            count = 0
            for interlink in interEdgeSetTmp[lrid][rrid]:
                if interlink[0] == brokenlink[1]:
                    count = 1
                    break
            if count == 0:
                regionBorderNodesTmp[lrid][rrid].remove(brokenlink[1])
            count = 0
            for interlink in interEdgeSetTmp[rrid][lrid]:
                if interlink[0] == brokenlink[2]:
                    count = 1
                    break
            if count == 0:
                regionBorderNodesTmp[rrid][lrid].remove(brokenlink[2])


        oripathmaxtrix = com_hp_pathset_failure(infilePrefix, topoName, nodeNum, pathSet, regionNum, nodeRegionId, regionrMatrix, wMatrixTmp, regionBorderNodesTmp, interEdgeSetTmp)
        env.change_pathset(oripathmaxtrix)
        
        maxpeerutil, bgFlowmap, ternimalTM = env.cal_hpmcf_hp_val(demRates[TMid], cMatrixTmp)
        objval = com_hpmcf_obj_val(regionNum, nodeNum, linkNum, demNum, demands, demRates[TMid], linkSetTmp, wMatrix, MAXWEIGHT, nodeRegionId, maxpeerutil, bgFlowmap, ternimalTM)
        fileout.write(str(objval) + "\n")
        pbar.update(1)
        # break
    pbar.close()
    # print(objval)
    # return # !!!!!!!!!!!!!!!!!!!!!!!!!
    fileout.close()

def com_hpmcf_obj_val(regionNum, nodeNum, linkNum, demNum, demands, demrate, linkSet, wMatrix, MAXWEIGHT, nodeRegionId, maxpeerutil, bgFlowmap, ternimalTM):
    maxutilList = [maxpeerutil]
    # print(maxutilList)
    for targetrid in range(regionNum):
        demrate_region = copy.deepcopy(demrate)
        demId = 0
        for src in range(nodeNum):
            for dst in range(nodeNum):
                if src == dst:
                    continue
                srid = nodeRegionId[src]
                trid = nodeRegionId[dst]
                if srid == targetrid and trid == targetrid:
                    demrate_region[demId] += ternimalTM[src][dst]
                else:
                    demrate_region[demId] = 0.0
                demId += 1
        maxutilRegion = mcfsolver_single_region(nodeNum, linkNum, demNum, demands, demrate_region, linkSet, wMatrix, MAXWEIGHT, nodeRegionId, targetrid, bgFlowmap)
        maxutilList.append(maxutilRegion)
    # print(maxutilList)

    return max(maxutilList)

def get_hpmcf_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = ""):
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()

    outfileName = outfilePrefix + "objvals/" + topoName + "_hpmcf_obj_vals%s.txt" % (synthesisType)
    fileout = open(outfileName, 'w', 1)
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        pbar.update(1)
        maxpeerutil, bgFlowmap, ternimalTM = env.cal_hpmcf_hp_val(demRates[TMid])
        objval = com_hpmcf_obj_val(regionNum, nodeNum, linkNum, demNum, demands, demRates[TMid], linkSet, wMatrix, MAXWEIGHT, nodeRegionId, maxpeerutil, bgFlowmap, ternimalTM)
        # print(TMid, objval)
        fileout.write(str(round(objval, 6)) + "\n")
    pbar.close()
    fileout.close()
    print("hpmcf finished")

def com_hp_val_failure(infilePrefix, topoName, nodeNum, pathSet, regionNum, nodeRegionId, regionrMatrix, wMatrix, synthesisType, TM, regionBorderNodes, interEdgeSet):
    pathSetMatrix = {}
    disMatrixDic = {}
    for sRid in range(regionNum):
        for tRid in range(regionNum):
            wMatrix_tmp = copy.deepcopy(wMatrix)
            if sRid == tRid:
                for i in range(nodeNum):
                    for j in range(nodeNum):
                        if i == j:
                            continue
                        if nodeRegionId[i] != sRid or nodeRegionId[j] != sRid:
                            wMatrix_tmp[i][j] = 999999

                pathSet, disMatrix = com_shr_pathset(nodeNum, wMatrix_tmp)
                pathSetMatrix[(sRid, tRid)] = pathSet
                disMatrixDic[(sRid, tRid)] = disMatrix

    # get candidate path set
    candidatePaths = []
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            sRid = nodeRegionId[src]
            tRid = nodeRegionId[dst]
            if sRid == tRid:
                path = pathSetMatrix[(sRid,tRid)][src][dst]
            else:
                src_tmp = src
                path = []
                nRid = sRid
                while nRid != tRid:
                    pRid = nRid
                    nRid = regionrMatrix[pRid][tRid]
                    if src_tmp in regionBorderNodes[pRid][nRid]:
                        path += [src_tmp]
                        outBorderNode = src_tmp
                    else:
                        pathws = []
                        for borderNode in regionBorderNodes[pRid][nRid]:
                            pathws.append(disMatrixDic[(pRid,pRid)][src_tmp][borderNode])
                        index = pathws.index(min(pathws))
                        # print(regionBorderNodes[pRid][nRid])
                        outBorderNode = regionBorderNodes[pRid][nRid][index]
                        # print(outBorderNode)
                        path += pathSetMatrix[(pRid,pRid)][src_tmp][outBorderNode]
                    for interlink in interEdgeSet[pRid][nRid]:
                        # print(interlink)
                        if interlink[0] == outBorderNode:
                            src_tmp = interlink[1]
                            break
                # print(src_tmp, dst)
                if src_tmp == dst:
                    path += [dst]
                else:
                    path += pathSetMatrix[(nRid,nRid)][src_tmp][dst]
                # print(src,dst)
                # print(path)
                # exit()
            candidatePaths.append([path])

    # compute sp value
    env = ReadTopo(infilePrefix, topoName, "hp", synthesis_type = synthesisType)
    env.change_pathset(candidatePaths)
    routing = []
    objval = env.calVal(routing, TM)
    return round(objval, 6)

def get_hp_obj_vals_failure(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = "", TMoffset = 40, totalDownLinkNum = 100):
    if totalTMNum > totalDownLinkNum:
        print("ERROR")
        return

    # read broken links
    filein = open(infilePrefix + "brokenlink/" + topoName + "_%dlinks.txt" % totalDownLinkNum, 'r')
    lines = filein.readlines()
    brokenLinkList = []
    for line in lines:
        lineList = line.strip().split()
        brokenLinkList.append(list(map(int, lineList)))
    filein.close()
    totalDownLinkNum = len(brokenLinkList)

    # read topo info
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()
    regionBorderNodes = env.read_hp_info()
    interEdgeSet = []
    for rid in range(regionNum):
        interEdgeSet.append([])
        for _ in range(regionNum):
            interEdgeSet[rid].append([])
    for link in linkSet:
        if link[-1] == -1:
            left = link[0]
            right = link[1]
            lRegionId = nodeRegionId[left]
            rRegionId = nodeRegionId[right]
            interEdgeSet[lRegionId][rRegionId].append((left,right))
            interEdgeSet[rRegionId][lRegionId].append((right,left))

    # cal obj vals
    outfileName = outfilePrefix + "objvals/" + topoName + "_hp_obj_vals%s_failure.txt" % (synthesisType)
    fileout = open(outfileName, 'w', 1)
    pbar = tqdm(total=totalDownLinkNum)
    for linkId in range(totalDownLinkNum):
        brokenlink = brokenLinkList[linkId]
        TMid = TMoffset + linkId//totalTMNum

        wMatrix_tmp = copy.deepcopy(wMatrix)
        wMatrix_tmp[brokenlink[1]][brokenlink[2]] = MAXWEIGHT
        wMatrix_tmp[brokenlink[2]][brokenlink[1]] = MAXWEIGHT

        # TODO: Test
        regionBorderNodesTmp = copy.deepcopy(regionBorderNodes)
        interEdgeSetTmp = copy.deepcopy(interEdgeSet)
        if linkSetTmp[brokenlink[0]][-1] == -1:
            lrid = nodeRegionId[brokenlink[1]]
            rrid = nodeRegionId[brokenlink[2]]
            interEdgeSetTmp[lrid][rrid].remove((brokenlink[1], brokenlink[2]))
            interEdgeSetTmp[rrid][lrid].remove((brokenlink[2], brokenlink[1]))
            count = 0
            for interlink in interEdgeSetTmp[lrid][rrid]:
                if interlink[0] == brokenlink[1]:
                    count = 1
                    break
            if count == 0:
                regionBorderNodesTmp[lrid][rrid].remove(brokenlink[1])
            count = 0
            for interlink in interEdgeSetTmp[rrid][lrid]:
                if interlink[0] == brokenlink[2]:
                    count = 1
                    break
            if count == 0:
                regionBorderNodesTmp[rrid][lrid].remove(brokenlink[2])
        
        objval = com_hp_val_failure(infilePrefix, topoName, nodeNum, pathSet, regionNum, nodeRegionId, regionrMatrix, wMatrix_tmp, synthesisType, demRates[TMid], regionBorderNodesTmp, interEdgeSetTmp)
        fileout.write(str(objval) + "\n")
        pbar.update(1)
    pbar.close()
    fileout.close()

def com_sp_val_failure(infilePrefix, topoName, nodeNum, pathSet, regionNum, nodeRegionId, regionrMatrix, wMatrix, synthesisType, TM):
    pathSetMatrix = {}
    for sRid in range(regionNum):
        for tRid in range(regionNum):
            wMatrix_tmp = copy.deepcopy(wMatrix)
            if sRid == tRid:
                for i in range(nodeNum):
                    for j in range(nodeNum):
                        if i == j:
                            continue
                        if nodeRegionId[i] != sRid or nodeRegionId[j] != sRid:
                            wMatrix_tmp[i][j] = 999999
            else:
                rpath = [sRid]
                nRegion = sRid
                while nRegion != tRid:
                    nRegion = regionrMatrix[rpath[-1]][tRid]
                    rpath.append(nRegion)
                for i in range(nodeNum):
                    for j in range(nodeNum):
                        if i == j:
                            continue
                        if nodeRegionId[i] not in rpath:
                            wMatrix_tmp[i][j] = 999999
                        elif nodeRegionId[j] not in rpath:
                            wMatrix_tmp[i][j] = 999999
                        else:
                            pass
            pathSet, _ = com_shr_pathset(nodeNum, wMatrix_tmp)
            pathSetMatrix[(sRid, tRid)] = pathSet

    # get candidate path set
    candidatePaths = []
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            sRid = nodeRegionId[src]
            tRid = nodeRegionId[dst]
            path = pathSetMatrix[(sRid,tRid)][src][dst]
            candidatePaths.append([path])

    # compute sp value
    env = ReadTopo(infilePrefix, topoName, "sp", synthesis_type = synthesisType)
    env.change_pathset(candidatePaths)
    routing = []
    objval = env.calVal(routing, TM)
    return round(objval, 6)

def get_sp_obj_vals_failure(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = "", TMoffset = 40, totalDownLinkNum = 100):
    if totalTMNum > totalDownLinkNum:
        print("ERROR")
        return

    # read broken links
    filein = open(infilePrefix + "brokenlink/" + topoName + "_%dlinks.txt" % totalDownLinkNum, 'r')
    lines = filein.readlines()
    brokenLinkList = []
    for line in lines:
        lineList = line.strip().split()
        brokenLinkList.append(list(map(int, lineList)))
    filein.close()
    totalDownLinkNum = len(brokenLinkList)

    # read topo info
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()

    # cal obj vals
    outfileName = outfilePrefix + "objvals/" + topoName + "_sp_obj_vals%s_failure.txt" % (synthesisType)
    fileout = open(outfileName, 'w', 1)
    pbar = tqdm(total=totalDownLinkNum)
    for linkId in range(totalDownLinkNum):
        brokenlink = brokenLinkList[linkId]
        TMid = TMoffset + linkId//totalTMNum

        wMatrix_tmp = copy.deepcopy(wMatrix)
        wMatrix_tmp[brokenlink[1]][brokenlink[2]] = MAXWEIGHT
        wMatrix_tmp[brokenlink[2]][brokenlink[1]] = MAXWEIGHT
        
        objval = com_sp_val_failure(infilePrefix, topoName, nodeNum, pathSet, regionNum, nodeRegionId, regionrMatrix, wMatrix_tmp, synthesisType, demRates[TMid])
        fileout.write(str(objval) + "\n")
        pbar.update(1)
    pbar.close()
    fileout.close()

def get_mcf_obj_vals_failure(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = "", TMoffset = 40, totalDownLinkNum = 100):
    if totalTMNum > totalDownLinkNum:
        print("ERROR")
        return

    filein = open(infilePrefix + "brokenlink/" + topoName + "_%dlinks.txt" % totalDownLinkNum, 'r')
    lines = filein.readlines()
    brokenLinkList = []
    for line in lines:
        lineList = line.strip().split()
        brokenLinkList.append(list(map(int, lineList)))
    filein.close()
    totalDownLinkNum = len(brokenLinkList)

    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()

    objVals = []
    outfileName = outfilePrefix + "objvals/" + topoName + "_mcf_obj_vals%s_failure.txt" % (synthesisType)
    fileout = open(outfileName, 'w', 1)
    # pbar = tqdm(total=totalDownLinkNum)
    for linkId in range(totalDownLinkNum):
        brokenlink = brokenLinkList[linkId]
        TMid = TMoffset + linkId//totalTMNum

        linkSetTmp = copy.deepcopy(linkSet)
        linkSetTmp[brokenlink[0]][3] = 0.01
        
        objval,_,__ = mcfsolver(nodeNum, linkNum, demNum, demands, demRates[TMid], linkSetTmp, wMatrix, MAXWEIGHT, nodeRegionId, regionrMatrix)
        objVals.append(round(objval, 6))
        fileout.write(str(objval) + "\n")
        # pbar.update(1)
        # break
    # pbar.close()
    # print(objval)
    # return # !!!!!!!!!!!!!!!!!!!!!!!!!
    fileout.close()

def get_down_links(infilePrefix, topoName, pathType, synthesisType = "", totalDownLinkNum = 100):
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()
    # print(linkSet)
    print(linkNum)
    # exit()
    # which link can not down, print number
    keyLinkMatrix = []
    for _ in range(nodeNum):
        keyLinkMatrix.append([0]*nodeNum)
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            dicts = {}
            pathNum = len(pathSet[src][dst])
            for k in range(pathNum):
                path = pathSet[src][dst][k]
                pathLen = len(path)
                for l in range(pathLen-1):
                    key1 = (path[l],path[l+1])
                    key2 = (path[l+1],path[l])
                    if key1 in dicts:
                        dicts[key1] += 1
                    else:
                        dicts[key1] = 1
                    if key2 in dicts:
                        dicts[key2] += 1
                    else:
                        dicts[key2] = 1
            for key in dicts.keys():
                if dicts[key] >= pathNum:
                    keyLinkMatrix[key[0]][key[1]] = 1
                    keyLinkMatrix[key[1]][key[0]] = 1
    #                 print(pathSet[src][dst])
    #                 print(key)
    #                 # exit()
    # exit()
    keyLinkList = []
    for i in range(nodeNum):
        for j in range(nodeNum):
            if keyLinkMatrix[i][j] == 1:
                keyLinkList.append([i,j])
                # print(i,j)
    print(len(keyLinkList))

    # count peering link num
    peerLinkNum = {}
    for rid in range(regionNum):
        for rid2 in range(regionNum):
            if rid == rid2:
                continue
            peerLinkNum[(rid, rid2)] = 0
    
    for link in linkSet:
        if link[-1] == -1:
            lrid = nodeRegionId[link[0]]
            rrid = nodeRegionId[link[1]]
            peerLinkNum[(lrid, rrid)] += 1
            peerLinkNum[(rrid, lrid)] += 1
    print(peerLinkNum)
    # select some links by random
    brokenLinkList = []
    for _ in range(totalDownLinkNum):
        while True:
            linkId = random.randint(0,linkNum-1)
            link = linkSet[linkId]
            if link[-1] == -1:
                lrid = nodeRegionId[link[0]]
                rrid = nodeRegionId[link[1]]
                if peerLinkNum[(lrid, rrid)] > 1:
                    brokenLinkList.append([linkId] + link[0:2])
                    break
            else:
                if link[0:2] not in keyLinkList:
                    brokenLinkList.append([linkId] + link[0:2])
                    break
    # print(linkSet)
    # print()
    # print(brokenLinkList)
    # write to a file
    fileout = open(infilePrefix + "brokenlink/" + topoName + "_%dlinks.txt" % totalDownLinkNum, 'w')
    for link in brokenLinkList:
        fileout.write(" ".join([str(item) for item in link]) + "\n")
    fileout.close()

def com_shr_pathset(nodeNum, wMatrix):
    rMatrix = []
    for i in range(nodeNum):
        rMatrix.append([j for j in range(nodeNum)])

    for k in range(nodeNum):
        for i in range(nodeNum):
            for j in range(nodeNum):
                if wMatrix[i][j] > wMatrix[i][k] + wMatrix[k][j]:
                    wMatrix[i][j] = wMatrix[i][k] + wMatrix[k][j]
                    rMatrix[i][j] = rMatrix[i][k]

    pathSet = []
    for src in range(nodeNum):
        pathSet.append([])
        for dst in range(nodeNum):
            if src == dst:
                pathSet[src].append([])
                continue
            path = [src]
            tmp = rMatrix[src][dst]
            path.append(tmp)
            while tmp != dst:
                tmp = rMatrix[tmp][dst]
                path.append(tmp)
            pathSet[src].append(path)
    return pathSet, wMatrix

def get_sp_pathset_region(infilePrefix, topoName, pathType, synthesisType = ""):
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()

    pathSetMatrix = {}

    for sRid in range(regionNum):
        for tRid in range(regionNum):
            wMatrix_tmp = copy.deepcopy(wMatrix)
            if sRid == tRid:
                for i in range(nodeNum):
                    for j in range(nodeNum):
                        if i == j:
                            continue
                        if nodeRegionId[i] != sRid or nodeRegionId[j] != sRid:
                            wMatrix_tmp[i][j] = 999999
            # else:
            #     for i in range(nodeNum):
            #         for j in range(nodeNum):
            #             if i == j:
            #                 continue
            #             if nodeRegionId[i] != sRid and nodeRegionId[i] != tRid:
            #                 wMatrix_tmp[i][j] = 999999
            #             elif nodeRegionId[j] != sRid and nodeRegionId[j] != tRid:
            #                 wMatrix_tmp[i][j] = 999999
            #             else:
            #                 pass
            else:
                rpath = [sRid]
                nRegion = sRid
                while nRegion != tRid:
                    nRegion = regionrMatrix[rpath[-1]][tRid]
                    rpath.append(nRegion)
                for i in range(nodeNum):
                    for j in range(nodeNum):
                        if i == j:
                            continue
                        if nodeRegionId[i] not in rpath:
                            wMatrix_tmp[i][j] = 999999
                        elif nodeRegionId[j] not in rpath:
                            wMatrix_tmp[i][j] = 999999
                        else:
                            pass
            pathSet, _ = com_shr_pathset(nodeNum, wMatrix_tmp)
            pathSetMatrix[(sRid, tRid)] = pathSet

    fileout = open(infilePrefix + "pathset/" + topoName + "_paths_sp.txt", 'w')
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            sRid = nodeRegionId[src]
            tRid = nodeRegionId[dst]
            path = pathSetMatrix[(sRid,tRid)][src][dst]
            pathStr = " ".join([str(item) for item in path])
            fileout.write(pathStr + "\n")
    fileout.close()

def get_hp_pathset_region(infilePrefix, topoName, pathType, synthesisType = ""): # Hot Potato
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()
    regionBorderNodes = env.read_hp_info()
    # print(regionBorderNodes)
    interEdgeSet = []
    for rid in range(regionNum):
        interEdgeSet.append([])
        for _ in range(regionNum):
            interEdgeSet[rid].append([])
    for link in linkSet:
        if link[-1] == -1:
            left = link[0]
            right = link[1]
            lRegionId = nodeRegionId[left]
            rRegionId = nodeRegionId[right]
            interEdgeSet[lRegionId][rRegionId].append((left,right))
            interEdgeSet[rRegionId][lRegionId].append((right,left))
    # print(interEdgeSet)


    pathSetMatrix = {}
    disMatrixDic = {}

    for sRid in range(regionNum):
        for tRid in range(regionNum):
            wMatrix_tmp = copy.deepcopy(wMatrix)
            if sRid == tRid:
                for i in range(nodeNum):
                    for j in range(nodeNum):
                        if i == j:
                            continue
                        if nodeRegionId[i] != sRid or nodeRegionId[j] != sRid:
                            wMatrix_tmp[i][j] = 999999

                pathSet, disMatrix = com_shr_pathset(nodeNum, wMatrix_tmp)
                pathSetMatrix[(sRid, tRid)] = pathSet
                disMatrixDic[(sRid, tRid)] = disMatrix

    fileout = open(infilePrefix + "pathset/" + topoName + "_paths_hp.txt", 'w')
    hpPathMatrix = []
    for src in range(nodeNum):
        hpPathMatrix.append([])
        for dst in range(nodeNum):
            # if src != 0 or dst != 30:
            #     continue
            if src == dst:
                hpPathMatrix[src].append([])
                continue
            sRid = nodeRegionId[src]
            tRid = nodeRegionId[dst]
            if sRid == tRid:
                path = pathSetMatrix[(sRid,tRid)][src][dst]
            else:
                src_tmp = src
                path = []
                nRid = sRid
                while nRid != tRid:
                    pRid = nRid
                    nRid = regionrMatrix[pRid][tRid]
                    if src_tmp in regionBorderNodes[pRid][nRid]:
                        path += [src_tmp]
                        outBorderNode = src_tmp
                    else:
                        pathws = []
                        for borderNode in regionBorderNodes[pRid][nRid]:
                            pathws.append(disMatrixDic[(pRid,pRid)][src_tmp][borderNode])
                        index = pathws.index(min(pathws))
                        # print(regionBorderNodes[pRid][nRid])
                        outBorderNode = regionBorderNodes[pRid][nRid][index]
                        # print(outBorderNode)
                        path += pathSetMatrix[(pRid,pRid)][src_tmp][outBorderNode]
                    for interlink in interEdgeSet[pRid][nRid]:
                        # print(interlink)
                        if interlink[0] == outBorderNode:
                            src_tmp = interlink[1]
                            break
                # print(src_tmp, dst)
                if src_tmp == dst:
                    path += [dst]
                else:
                    path += pathSetMatrix[(nRid,nRid)][src_tmp][dst]
                # print(src,dst)
                # print(path)
                # exit()
            pathStr = " ".join([str(item) for item in path])
            fileout.write(pathStr + "\n")
            hpPathMatrix[src].append([path])
    fileout.close()
    fileout = open(infilePrefix + "pathset/" + topoName + "_hp.pickle", 'wb')
    pickle.dump(hpPathMatrix, fileout)
    fileout.close()

def get_region_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = ""):
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()

    objVals = []
    optRoutings = []
    # pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        # TMid = 4
        # pbar.update(1)
        demrate = env.set_demrate(demRates[TMid])
        TM = env.get_TM(demrate)
        objval = 0.0
        routing, objval = sorsolver_loop(nodeNum, pathSet, TM, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix) # sorsolver_loop, sorsolver_new
        
        objval2 = 0.0
        if topoName not in ["briten12r16gridb", "briten12r16grid"]:
            objval2,_,__ = mcfsolver(nodeNum, linkNum, demNum, demands, demrate, linkSet, wMatrix, MAXWEIGHT, nodeRegionId, regionrMatrix)
        print(TMid, objval, objval2)
        # objVals.append(round(objval, 6))
        # optRoutings.append(routing)
    # pbar.close()
    print()
    # print(objval, objval2)
    return
    # outfileName = outfilePrefix + "objvals/" + topoName + "_" + pathType + "_obj_vals%s.txt" % (synthesisType)
    # fileout = open(outfileName, 'w')
    # for item in objVals:
    #     fileout.write(str(item) + "\n")
    # fileout.close()

def get_mcf_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = ""):
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()

    objVals = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        pbar.update(1)
        objval,_,__ = mcfsolver(nodeNum, linkNum, demNum, demands, demRates[TMid], linkSet, wMatrix, MAXWEIGHT, nodeRegionId, regionrMatrix)
        objVals.append(round(objval, 6))
        # break
    pbar.close()
    # print(objval)
    # return # !!!!!!!!!!!!!!!!!!!!!!!!!
    outfileName = outfilePrefix + "objvals/" + topoName + "_mcf_obj_vals%s.txt" % (synthesisType)
    
    fileout = open(outfileName, 'w')
    for item in objVals:
        fileout.write(str(item) + "\n")
    fileout.close()

def get_mcf_obj_vals_region(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = "", pathNum = 3):
    env = ReadTopo(infilePrefix, topoName, pathType, synthesis_type = synthesisType, path_num = pathNum)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()

    objVals = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        pbar.update(1)
        objval,_,__ = mcfsolver_normal(nodeNum, linkNum, demNum, demands, demRates[TMid], linkSet, wMatrix, MAXWEIGHT)
        objVals.append(round(objval, 6))
        # break
    pbar.close()
    # print(objval)
    # return # !!!!!!!!!!!!!!!!!!!!!!!!!
    outfileName = outfilePrefix + "objvals/" + topoName + "_mcf_obj_vals%s.txt" % (synthesisType)
    
    fileout = open(outfileName, 'w')
    for item in objVals:
        fileout.write(str(item) + "\n")
    fileout.close()

def get_smore_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = ""):
    env = ReadTopo(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()
    objVals = []
    optRoutings = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        pbar.update(1)
        pathRates = []
        for i in range(demNum):
            for j in range(len(pathSet[i])):
                if TMid == 0:
                    pathRates.append(demRates[TMid][i])
                else:
                    pathRates.append(demRates[TMid-1][i])
        routing, objval = sorsolver(nodeNum, demNum, totalPathNum, pathSet, pathRates, cMatrix)
        objval = env.calVal(routing, demRates[TMid])
        objVals.append(round(objval, 6))
        optRoutings.append(routing)
    pbar.close()
    print()
    outfileName = outfilePrefix + "objvals/" + topoName + "_smore_obj_vals%s.txt" % (synthesisType)
    if pathType != "racke":
        outfileName = outfilePrefix + "objvals/" + topoName + "_smore_obj_vals%s_%s.txt" % (synthesisType, pathType)
    fileout = open(outfileName, 'w')
    for item in objVals:
        fileout.write(str(item) + "\n")
    fileout.close()

def get_pathtype_obj_vals_new(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = "", pathNum = 3):
    env = ReadTopo(infilePrefix, topoName, pathType, synthesis_type = synthesisType, path_num = pathNum)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()

    demandNum = len(demRates[0])
    demrate = np.array([0]*demandNum)
    for i in range(20):
        rate = np.array(demRates[i])
        demrate = demrate + rate
    demrate /= 20

    # 2. sort region's demands
    smallDemIdMap = [0]*demandNum
    index = np.argsort(demrate)
    res = [round(demrate[i], 0) for i in index]
    for i in range(demandNum):
        smallDemIdMap[index[i]] = int(i/(demandNum*0.1)) + 1

    demId = 0
    for i in range(nodeNum):
        for j in range(nodeNum):
            if i == j:
                continue

            if smallDemIdMap[demId] <= 9:
                pathSet[demId] = pathSet[demId][:1]
                totalPathNum -= 1
            # elif smallDemIdMap[demId] <= 9:
            #     pathSet[demId] = pathSet[demId][:2]
            #     totalPathNum -= 1

            demId += 1

    objVals = []
    optRoutings = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        pbar.update(1)
        pathRates = []
        for i in range(demNum):
            for j in range(len(pathSet[i])):
                pathRates.append(demRates[TMid][i])
        routing, objval = sorsolver_bindR(nodeNum, demNum, totalPathNum, pathSet, pathRates, cMatrix, smallDemIdMap)
        objVals.append(round(objval, 6))
        optRoutings.append(routing)
    pbar.close()
    print()
    print(routing)
    print(objval)
    exit()

def get_pathtype_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = "", pathNum = 3):
    env = ReadTopo(infilePrefix, topoName, pathType, synthesis_type = synthesisType, path_num = pathNum)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()


    demandNum = len(demRates[0])
    demrate = np.array([0]*demandNum)
    for i in range(20):
        rate = np.array(demRates[i])
        demrate = demrate + rate
    demrate /= 20

    # 2. sort region's demands
    smallDemIdMap = [0]*demandNum
    index = np.argsort(demrate)
    res = [round(demrate[i], 0) for i in index]
    for i in range(demandNum):
        smallDemIdMap[index[i]] = int(i/(demandNum*0.1)) + 1

    nodeDegree = [0]*nodeNum
    for link in linkSet:
        nodeDegree[link[0]] += 1
        nodeDegree[link[1]] += 1
    demId = 0
    # print([len(path) for path in pathSet])
    for i in range(nodeNum):
        for j in range(nodeNum):
            if i == j:
                continue
            # if nodeDegree[i] <= 3 and nodeDegree[j] <= 3:
            #     pathSet[demId] = pathSet[demId][:2]
            #     totalPathNum -= 1
            # elif nodeDegree[i] <= 2 and nodeDegree[j] >= 4:
            #     pathSet[demId] = pathSet[demId][:2]
            #     totalPathNum -= 1
            # elif nodeDegree[i] >= 4 and nodeDegree[j] <= 2:
            #     pathSet[demId] = pathSet[demId][:2]
            #     totalPathNum -= 1

            if smallDemIdMap[demId] <= 9:
                pathSet[demId] = pathSet[demId][:1]
                totalPathNum -= 2
            # elif smallDemIdMap[demId] <= 9:
            #     pathSet[demId] = pathSet[demId][:2]
            #     totalPathNum -= 1

            demId += 1
    
    # print([len(path) for path in pathSet])
    # exit()
    


    objVals = []
    optRoutings = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        pbar.update(1)
        pathRates = []
        for i in range(demNum):
            for j in range(len(pathSet[i])):
                pathRates.append(demRates[TMid][i])
        routing, objval = sorsolver(nodeNum, demNum, totalPathNum, pathSet, pathRates, cMatrix)
        objVals.append(round(objval, 6))
        optRoutings.append(routing)
    pbar.close()
    print()
    # print(routing)
    nodeDegree = [0]*nodeNum
    for link in linkSet:
        nodeDegree[link[0]] += 1
        nodeDegree[link[1]] += 1
    demId = 0
    actionDict = {}
    actCount = 0
    for i in range(nodeNum):
        for j in range(nodeNum):
            if i == j:
                continue
            action = routing[actCount:actCount + len(pathSet[demId])]
            actCount += len(pathSet[demId])
            action = '-'.join(list(map(str, action)))
            item = (nodeDegree[i], nodeDegree[j])
            # item = smallDemIdMap[demId]
            # item = pathSet[demId]
            if action in actionDict:
                actionDict[action].append(item)
            else:
                actionDict[action] = [item]
            demId += 1
    for key in actionDict.keys():
        # if len(actionDict[key]) == 1:
        #     # print(key, pathSet[actionDict[key][0]])
        #     print(key, actionDict[key][0], (nodeDegree[actionDict[key][0][0]], nodeDegree[actionDict[key][0][1]]))
        print(key, actionDict[key])
    
    print(nodeDegree[0:5])
    print(nodeDegree[5:10])
    print(nodeDegree[10:15])
    print(nodeDegree[15:])
    print(objval)
    return # !!!!!!!!!!!!!!!!!!!!!!!!!!
    outfileName = outfilePrefix + "objvals/" + topoName + "_" + pathType + "_obj_vals%s.txt" % (synthesisType)
    fileout = open(outfileName, 'w')
    for item in objVals:
        fileout.write(str(item) + "\n")
    fileout.close()

def get_lb_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = ""):
    env = EnvRegion(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, pathSet, demRates, cMatrix, regionNum, pathNumListReduced, pathNumMapRegion, nodeRegionId, regionrMatrix = env.read_info_MMA()
    demNum, demands, linkSet, wMatrix, MAXWEIGHT = env.read_info()

    objVals = []
    # routing = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        # TMid = 4
        pbar.update(1)
        objval = env.cal_val(demRates[TMid])
        # print(TMid, objval)
        objVals.append(round(objval, 6))
    pbar.close()
    print()
    # print(objval)
    # return
    outfileName = outfilePrefix + "objvals/" + topoName + "_lb_obj_vals%s.txt" % (synthesisType)
    fileout = open(outfileName, 'w')
    for item in objVals:
        fileout.write(str(item) + "\n")
    fileout.close()

def get_sp_pathset(infilePrefix, topoName, synthesisType = ""):
    env = ReadTopo(infilePrefix, topoName, synthesis_type = synthesisType)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()
    rMatrix = []
    for i in range(nodeNum):
        rMatrix.append([j for j in range(nodeNum)])

    for k in range(nodeNum):
        for i in range(nodeNum):
            for j in range(nodeNum):
                if wMatrix[i][j] > wMatrix[i][k] + wMatrix[k][j]:
                    wMatrix[i][j] = wMatrix[i][k] + wMatrix[k][j]
                    rMatrix[i][j] = rMatrix[i][k]

    fileout = open(infilePrefix + "pathset/" + topoName + "_paths_sp.txt", 'w')
    for src in range(nodeNum):
        for dst in range(nodeNum):
            if src == dst:
                continue
            path = [src]
            tmp = rMatrix[src][dst]
            path.append(tmp)
            while tmp != dst:
                tmp = rMatrix[tmp][dst]
                path.append(tmp)
            pathStr = " ".join([str(item) for item in path])
            fileout.write(pathStr + "\n")
    fileout.close()

def get_sp_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = ""):
    env = ReadTopo(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()
    objVals = []
    routing = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        # TMid = 4
        pbar.update(1)
        objval = env.calVal(routing, demRates[TMid])
        # print(TMid, objval)
        objVals.append(round(objval, 6))
    pbar.close()
    print()
    print(objval)
    # return
    outfileName = outfilePrefix + "objvals/" + topoName + "_sp_obj_vals%s.txt" % (synthesisType)
    fileout = open(outfileName, 'w')
    for item in objVals:
        fileout.write(str(item) + "\n")
    fileout.close()

def get_hp_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = ""):
    env = ReadTopo(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()
    objVals = []
    routing = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        # TMid = 4
        pbar.update(1)
        objval = env.calVal(routing, demRates[TMid])
        # print(TMid, objval)
        objVals.append(round(objval, 6))
    pbar.close()
    print()
    print(objval)
    # return
    outfileName = outfilePrefix + "objvals/" + topoName + "_hp_obj_vals%s.txt" % (synthesisType)
    fileout = open(outfileName, 'w')
    for item in objVals:
        fileout.write(str(item) + "\n")
    fileout.close()

def get_or_pathset(infilePrefix, topoName, synthesisType = ""):
    env = ReadTopo(infilePrefix, topoName, synthesis_type = synthesisType)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()

    pathSet, routing = orisolver(nodeNum, linkNum, demNum, demands, linkSet, wMatrix, MAXWEIGHT)

    # write files
    fileout = open(infilePrefix + "pathset/" + topoName + "_paths_or.txt", 'w')
    for demId in range(demNum):
        for i in range(len(pathSet[demId])):
            pathStrList = [str(item) for item in pathSet[demId][i]]
            fileout.write(" ".join(pathStrList) + "\n")
    fileout.close()

    fileout = open(infilePrefix + "routing/" + topoName + "_or_routing.txt", 'w')
    for item in routing:
        fileout.write(str(item) + "\n")
    fileout.close()

def get_or_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType, synthesisType = ""):
    env = ReadTopo(infilePrefix, topoName, pathType, synthesis_type = synthesisType)
    nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()

    file = open(infilePrefix + "routing/" + topoName + "_or_routing.txt")
    lines = file.readlines()
    file.close()
    routing = list(map(float, [item.strip() for item in lines]))

    objVals = []
    pbar = tqdm(total=totalTMNum)
    for TMid in range(totalTMNum):
        pbar.update(1)
        objval = env.calVal(routing, demRates[TMid])
        objVals.append(round(objval, 6))
    pbar.close()
    print()
    outfileName = outfilePrefix + "objvals/" + topoName + "_or_obj_vals%s.txt" % (synthesisType)
    fileout = open(outfileName, 'w')
    for item in objVals:
        fileout.write(str(item) + "\n")
    fileout.close()