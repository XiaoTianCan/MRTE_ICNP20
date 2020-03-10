#!/usr/bin/python3
# -*- coding: UTF-8 -*-

'''
Objective: This Env is used for offline training and testing. 
Created at 12/27/2019
'''

from __future__ import division
import copy
import pickle
import numpy as np

class EnvRegion:
    def __init__(self, infile_prefix, topo_name, path_type = "", synthesis_type = ""):
        # store topo info
        self.__toponame = topo_name
        self.__nodenum = 0
        self.__linknum = 0
        self.__linkset = []
        self.__wMatrix = []
        self.__cMatrix = []
        self.__edgemap = []
        self.__regionwMatrix = [] # region-level
        self.__regionrMatrix = [] # region-level

        self.__regionedgenum = []
        self.__regionnum = 0
        self.__noderegionid = []

        # store demands
        self.__demands = []
        self.__smalldemidmap = []
        
        # store paths
        self.__oripathmaxtrix = []
        self.__interpathedgemap = []
        self.__actionrangemap = []

        # store rates
        self.__totalTMnum = 0
        self.__demrates = []
        self.__demrate = []
        self.__TM = []
        
        # train
        self.__actionmatrix = []
        
        # file path
        self.__topofile = infile_prefix + "region/" + topo_name + ".txt"
        self.__pathfile = infile_prefix + "pathset/" + topo_name + "_" + path_type + ".pickle"
        
        traffic_type = "testset"
        self.__ratefile = infile_prefix + "traffic/" + traffic_type + "/" + topo_name + "_TMset%s.txt" % (synthesis_type)

        # initial functions
        self.get_regions()
        self.get_demands()
        if path_type != '':
            self.get_paths()
        self.get_TMset()

    def get_regions(self):
        file = open(self.__topofile)
        lines = file.readlines()
        file.close()
        lineList = lines[0].strip().split()
        self.__nodenum = int(lineList[0])
        self.__linknum = int(lineList[1])
        for i in range(self.__nodenum):
            self.__wMatrix.append([])
            self.__cMatrix.append([0.0]*self.__nodenum)
            self.__edgemap.append([-1]*self.__nodenum)
            for j in range(self.__nodenum):
                if i == j:
                    self.__wMatrix[i].append(0)
                else:
                    self.__wMatrix[i].append(999999)

        lineList = lines[self.__linknum+1].strip().split()
        self.__noderegionid = list(map(int, lineList))
        self.__regionnum = max(self.__noderegionid) + 1
        self.__regionedgenum = [0]*self.__regionnum
        self.__regionbordernodes = []
        for rid in range(self.__regionnum):
            self.__regionbordernodes.append([])
            for _ in range(self.__regionnum):
                self.__regionbordernodes[rid].append([])
        # print(self.__regionbordernodes)

        for i in range(1, self.__linknum+1):
            lineList = lines[i].strip().split()
            left = int(lineList[0]) - 1
            right = int(lineList[1]) - 1
            weight = int(lineList[2])
            capa = float(lineList[3])
            regionId = int(lineList[4])
            # if regionId == -1:
            #     capa *= 10
            # capa = 9953
            self.__linkset.append([left, right, weight, capa, regionId])
            self.__wMatrix[left][right] = weight
            self.__wMatrix[right][left] = weight
            self.__cMatrix[left][right] = capa 
            self.__cMatrix[right][left] = capa
            self.__edgemap[left][right] = self.__noderegionid[left]
            self.__edgemap[right][left] = self.__noderegionid[right]
            self.__regionedgenum[self.__noderegionid[left]] += 1
            self.__regionedgenum[self.__noderegionid[right]] += 1
            if regionId == -1:
                leftRegionId = self.__noderegionid[left]
                rightRegionId = self.__noderegionid[right]
                if left not in self.__regionbordernodes[leftRegionId][rightRegionId]:
                    self.__regionbordernodes[leftRegionId][rightRegionId].append(left)
                if right not in self.__regionbordernodes[rightRegionId][leftRegionId]:
                    self.__regionbordernodes[rightRegionId][leftRegionId].append(right)
        
        for i in range(self.__regionnum):
            self.__regionwMatrix.append([])
            for j in range(self.__regionnum):
                if i == j:
                    self.__regionwMatrix[i].append(0)
                else:
                    self.__regionwMatrix[i].append(999999)
        for i in range(self.__linknum+2, len(lines)): # region-level
            lineList = lines[i].strip().split()
            left = int(lineList[0])
            right = int(lineList[1])
            self.__regionwMatrix[left][right] = 1
            self.__regionwMatrix[right][left] = 1
    
    def com_shr_path(self, regionNum, wMatrix_ori): # region-level
        rMatrix = []
        for i in range(regionNum):
            rMatrix.append([j for j in range(regionNum)])

        wMatrix = copy.deepcopy(wMatrix_ori)
        for k in range(regionNum):
            for i in range(regionNum):
                for j in range(regionNum):
                    if wMatrix[i][j] > wMatrix[i][k] + wMatrix[k][j]:
                        wMatrix[i][j] = wMatrix[i][k] + wMatrix[k][j]
                        rMatrix[i][j] = rMatrix[i][k]
        return rMatrix

    def get_paths(self):
        # 0. compute region-level path
        self.__regionrMatrix = self.com_shr_path(self.__regionnum, self.__regionwMatrix)

        # 1. read pickle file to get self.__oripathmaxtrix of the whole network
        file = open(self.__pathfile, 'rb')
        self.__oripathmaxtrix = pickle.load(file)
        file.close()

        # 2. get self.__interpathedgemap
        for _ in range(self.__nodenum):
            self.__interpathedgemap.append([-1]*self.__nodenum)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                pathNum = len(self.__oripathmaxtrix[src][dst])
                for k in range(pathNum):
                    path = self.__oripathmaxtrix[src][dst][k]
                    pathLen = len(path)
                    for l in range(pathLen-1):
                        self.__interpathedgemap[path[l]][path[l+1]] = self.__noderegionid[src]
    
    def get_demands(self):
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                self.__demands.append([src,dst])

    def com_path_flow(self, flowmap, pathSet, action, size):
        subsizes = []
        gates = []

        pathNum = len(pathSet)
        tmp = 0
        src = pathSet[0][0]
        dst = pathSet[0][-1]
        gates.append(dst)
        for i in range(pathNum):
            length = len(pathSet[i])
            subsize = action[i]*size
            if dst != pathSet[i][-1]:
                dst = pathSet[i][-1]
                gates.append(dst)
                subsizes.append(tmp)
                tmp = 0
            tmp += subsize
            for j in range(length-1):
                node1 = pathSet[i][j]
                node2 = pathSet[i][j+1]
                flowmap[node1][node2] += subsize
        subsizes.append(tmp)
        return subsizes, gates

    def cal_val(self, rate):
        actionmatrix = []
        for src in range(self.__nodenum):
            actionmatrix.append([])
            for dst in range(self.__nodenum):
                if src == dst:
                    actionmatrix[src].append([])
                else:
                    action = [1.0/len(self.__oripathmaxtrix[src][dst])]*len(self.__oripathmaxtrix[src][dst])
                    actionmatrix[src].append([round(item, 6) for item in action])
        # print(actionmatrix)
        flowmap = []
        for _ in range(self.__nodenum):
            flowmap.append([0.0]*self.__nodenum)
        
        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                
                sources = [src]
                sizes = [rate[demId]]
                demId += 1
                while True:
                    if len(sources) == 0:
                        break

                    pathset = self.__oripathmaxtrix[sources[0]][dst]
                    action = actionmatrix[sources[0]][dst] # [1/len(pathset)]*len(pathset) # actionmatrix[sources[0]][dst]

                    subsizes, gates = self.com_path_flow(flowmap, pathset, action, sizes[0])

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
        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                if self.__cMatrix[i][j] > 0:
                    util = flowmap[i][j] / self.__cMatrix[i][j]
                    edgeutils.append(util)
        return max(edgeutils)
    
    def cal_hpmcf_hp_val(self, rate, cMatrix = []):
        if cMatrix == []:
            cMatrix = self.__cMatrix
        actionmatrix = []
        for src in range(self.__nodenum):
            actionmatrix.append([])
            for dst in range(self.__nodenum):
                if src == dst:
                    actionmatrix[src].append([])
                else:
                    action = [1.0/len(self.__oripathmaxtrix[src][dst])]*len(self.__oripathmaxtrix[src][dst])
                    actionmatrix[src].append([round(item, 6) for item in action])
        # print(actionmatrix)
        flowmap = []
        for _ in range(self.__nodenum):
            flowmap.append([0.0]*self.__nodenum)

        ternimalTM = []
        for _ in range(self.__nodenum):
            ternimalTM.append([0.0]*self.__nodenum)

        subrates = []
        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                srid = self.__noderegionid[src]
                trid = self.__noderegionid[dst]
                if srid != trid:
                    for k in range(len(self.__oripathmaxtrix[src][dst][0])-1):
                        enode1 = self.__oripathmaxtrix[src][dst][0][k]
                        enode2 = self.__oripathmaxtrix[src][dst][0][k+1]
                        if self.__noderegionid[enode1] == trid:
                            # store new TM
                            ternimalTM[enode1][dst] += rate[demId]
                            break
                        flowmap[enode1][enode2] += rate[demId]
                demId += 1

        # compute util
        peeredgeutils = []
        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                if cMatrix[i][j] > 0:
                    if self.__noderegionid[i] != self.__noderegionid[j]:
                        util = flowmap[i][j] / cMatrix[i][j]
                        peeredgeutils.append(util)
        return max(peeredgeutils), flowmap, ternimalTM

    def get_TM(self, demrate):
        TM = []
        demId = 0
        for i in range(self.__nodenum):
            TM.append([0.0]*self.__nodenum)
            for j in range(self.__nodenum):
                if i == j:
                    continue
                TM[i][j] = demrate[demId]
                demId += 1
        return TM

    def change_pathset(self, oripathmaxtrix):
        self.__oripathmaxtrix = oripathmaxtrix

    def get_TMset(self):
        file = open(self.__ratefile)
        lines = file.readlines()
        file.close()
        self.__totalTMnum = len(lines)
        for i in range(self.__totalTMnum):
            lineList = lines[i].strip().split(',')
            rates = list(map(float, lineList))
            self.__demrates.append(rates)

    def change_TM(self):
        self.__demrate = self.__demrates[(self.__episode + self.__start_index) % self.__totalTMnum]
        self.__TM = []
        demId = 0
        for i in range(self.__nodenum):
            self.__TM.append([0.0]*self.__nodenum)
            for j in range(self.__nodenum):
                if i == j:
                    continue
                self.__TM[i][j] = self.__demrate[demId]
                demId += 1
    
    def set_demrate(self, demrate):
        demId = 0
        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                if i == j:
                    continue
                # if self.__noderegionid[i] == self.__noderegionid[j]:
                #     demrate[demId] = 0.0
                # elif self.__noderegionid[i] == 0 or self.__noderegionid[i] == 1:
                #     demrate[demId] = 0.0
                # else:
                #     pass
                demId += 1
        return demrate

    def show_info(self):
        print("--------------------------")
        print("----detail information----")
        print("topology:%s(%d,%d) with %d region(s)" % (self.__toponame, self.__nodenum, self.__linknum, self.__regionnum))
        print("--------------------------")

    def read_hp_info(self):
        return self.__regionbordernodes

    def read_info(self):
        # demNum, demands, demRates[TMid], linkSet, wMatrix, MAXWEIGHT
        return self.__nodenum*(self.__nodenum-1), self.__demands, self.__linkset, self.__wMatrix, 999999

    def read_info_MMA(self):

        pathNumListDuel = [[] for _ in range(self.__regionnum)]
        pathNumMapRegion = []
        for _ in range(self.__nodenum):
            pathNumMapRegion.append([0]*self.__regionnum)
        
        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                if sRegion == tRegion:
                    if len(pathNumListDuel[sRegion]) == 0:
                        pathNumListDuel[sRegion].append([])
                        pathNumListDuel[sRegion].append([])
                    pathNumListDuel[sRegion][0].append(len(self.__oripathmaxtrix[src][dst]))
                else:
                    pathNumMapRegion[src][tRegion] = len(self.__oripathmaxtrix[src][dst])
                demId += 1

        actionRangeMap = []
        for src in range(self.__nodenum):
            actionRangeMap.append([])
            for rid in range(self.__regionnum):
                actionRangeMap[src].append([])

        actCountList = [0]*self.__regionnum
        for src in range(self.__nodenum):
            sRegion = self.__noderegionid[src]
            for tRegion in range(self.__regionnum):
                if sRegion == tRegion:
                    continue
                if self.__regionrMatrix[sRegion][tRegion] != tRegion:  # region-level
                    continue
                pathNumListDuel[sRegion][1].append(pathNumMapRegion[src][tRegion])
                actionRangeMap[src][tRegion] = [actCountList[sRegion], actCountList[sRegion]+pathNumMapRegion[src][tRegion]]
                actCountList[sRegion] += pathNumMapRegion[src][tRegion]

        for src in range(self.__nodenum):
            sRegion = self.__noderegionid[src]
            for tRegion in range(self.__regionnum):
                if sRegion == tRegion:
                    continue
                if self.__regionrMatrix[sRegion][tRegion] != tRegion:
                    nextRegionHop = self.__regionrMatrix[sRegion][tRegion]
                    actionRangeMap[src][tRegion] = actionRangeMap[src][nextRegionHop]
        self.__actionrangemap = actionRangeMap
        
        print("regionedgenum:", self.__regionedgenum)
        print("actionDim:", [(sum(item[0]), sum(item[1])) for item in pathNumListDuel])
        # print("pathNumListDuel:", pathNumListDuel)
        # exit()
        # return self.__regionnum, self.__regionedgenum, pathNumListDuel
        return self.__nodenum, self.__linknum, self.__oripathmaxtrix, self.__demrates, self.__cMatrix, self.__regionnum, pathNumListDuel, pathNumMapRegion, self.__noderegionid, self.__regionrMatrix