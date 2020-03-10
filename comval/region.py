#!/usr/bin/python3
# -*- coding: UTF-8 -*-

'''
Objective: This Env is used for offline training and testing. 
Created at 12/27/2019
'''

from __future__ import division
import copy
import pickle

class EnvRegion:
    def __init__(self, infile_prefix, topo_name, path_type = '', synthesis_type = ""):
        # store topo info
        self.__toponame = topo_name
        self.__nodenum = 0
        self.__linknum = 0
        self.__linkset = []
        self.__wMatrix = []
        self.__cMatrix = []
        self.__edgemap = []

        self.__regionedgenum = []
        self.__regionnum = 0
        self.__noderegionid = []

        # store demands
        self.__demands = []
        
        # store paths
        self.__oripathmaxtrix = []
        self.__interpathedgemap = []
        self.__pathnummapregion = [] # store path num of inter demands, nodenum*regionnum

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

        for i in range(1, self.__linknum+1):
            lineList = lines[i].strip().split()
            left = int(lineList[0]) - 1
            right = int(lineList[1]) - 1
            weight = int(lineList[2])
            capa = float(lineList[3])
            regionId = int(lineList[4])
            self.__linkset.append([left, right, weight, capa, regionId])
            self.__wMatrix[left][right] = weight
            self.__wMatrix[right][left] = weight
            self.__cMatrix[left][right] = capa 
            self.__cMatrix[right][left] = capa
            self.__edgemap[left][right] = self.__noderegionid[left]
            self.__edgemap[right][left] = self.__noderegionid[right]
            self.__regionedgenum[self.__noderegionid[left]] += 1
            self.__regionedgenum[self.__noderegionid[right]] += 1
    
    def get_paths(self):
        # 1. read pickle file to get self.__oripathmaxtrix of the whole network
        file = open(self.__pathfile, 'rb')
        self.__oripathmaxtrix = pickle.load(file)
        file.close()
        # print(self.__oripathmaxtrix)
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

    def com_action_matrix(self, actionList):
        self.__actionmatrix = []
        for src in range(self.__nodenum):
            self.__actionmatrix.append([])
            for dst in range(self.__nodenum):
                self.__actionmatrix[src].append([])
        
        actCountList = [0]*self.__regionnum
        actCountList2 = [0]*self.__regionnum
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                pathNum = len(self.__oripathmaxtrix[src][dst])
                if sRegion == tRegion:
                    action = actionList[sRegion*2][actCountList[sRegion]:actCountList[sRegion]+pathNum]
                    self.__actionmatrix[src][dst] = action
                    actCountList[sRegion] += pathNum
                else:
                    action = actionList[sRegion*2+1][actCountList2[sRegion]:actCountList2[sRegion]+pathNum]
                    self.__actionmatrix[src][dst] = action
                    actCountList2[sRegion] += pathNum

    def com_action_matrix2(self, actionList):
        self.__actionmatrix = []
        for src in range(self.__nodenum):
            self.__actionmatrix.append([])
            for dst in range(self.__nodenum):
                self.__actionmatrix[src].append([])
        
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
                actionRangeMap[src][tRegion] = [actCountList[sRegion], actCountList[sRegion]+self.__pathnummapregion[src][tRegion]]
                actCountList[sRegion] += self.__pathnummapregion[src][tRegion]
        
        actCountList = [0]*self.__regionnum
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                pathNum = len(self.__oripathmaxtrix[src][dst])
                if sRegion == tRegion:
                    action = actionList[sRegion*2][actCountList[sRegion]:actCountList[sRegion]+pathNum]
                    self.__actionmatrix[src][dst] = action
                    actCountList[sRegion] += pathNum
                else:
                    actionRange = actionRangeMap[src][tRegion]
                    action = actionList[sRegion*2+1][actionRange[0]:actionRange[1]]
                    self.__actionmatrix[src][dst] = action

    def compute_flowmap(self):
        flowmap = []
        for _ in range(self.__nodenum):
            flowmap.append([0.0]*self.__nodenum)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                
                sources = [src]
                sizes = [self.__TM[src][dst]]
                while True:
                    if len(sources) == 0:
                        break
                    pathSet = self.__oripathmaxtrix[sources[0]][dst]
                    action = self.__actionmatrix[sources[0]][dst]
                    subsizes, gates = self.com_path_flow(flowmap, pathSet, action, sizes[0])
                    sources.pop(0)
                    sizes.pop(0)
                    for gwid in range(len(gates)):
                        if gates[gwid] == dst:
                            continue
                        sources.append(gates[gwid])
                        sizes.append(subsizes[gwid])
                        
        return flowmap

    def compute_util(self, flowmap):
        maxutilList = []
        netutilList = [[] for _ in range(self.__regionnum)]
        edgeUtilList = [[] for _ in range(self.__regionnum)]
        # for MMA
        netutilListReduced = [[] for _ in range(self.__regionnum)]
        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                if self.__edgemap[i][j] >= 0:
                    util = round(flowmap[i][j]/self.__cMatrix[i][j], 4)
                    regionId = self.__edgemap[i][j]
                    netutilList[regionId].append(util)
                    edgeUtilList[regionId].append([i, j])
                    if self.__interpathedgemap[i][j] >= 0:
                        netutilListReduced[regionId].append(util)
        # for netutil in netutilList:
        #     maxutilList.append(max(netutil))
        maxutilList = []
        bottleedgeList = []
        for regionId in range(self.__regionnum):
            netutil = netutilList[regionId]
            maxutil = 0.0
            bottleedge = []
            for i in range(len(netutil)):
                if netutil[i] > maxutil:
                    maxutil = netutil[i]
                    bottleedge = edgeUtilList[regionId][i]
            maxutilList.append(maxutil)
            bottleedgeList.append(bottleedge + [self.__noderegionid[bottleedge[0]], self.__noderegionid[bottleedge[1]]])
        return maxutilList, netutilList, netutilListReduced, bottleedgeList

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
    
    def set_demrate(self, demrate):
        demId = 0
        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                if i == j:
                    continue
                # if self.__noderegionid[i] != self.__noderegionid[j]:
                #     demrate[demId] = 0.0
                # if self.__noderegionid[i] == 0:
                #     demrate[demId] = 0.0
                demId += 1
        return demrate

    def show_info(self):
        print("--------------------------")
        print("----detail information----")
        print("topology:%s(%d,%d) with %d region(s)" % (self.__toponame, self.__nodenum, self.__linknum, self.__regionnum))
        print("--------------------------")

    def read_info(self):
        # demNum, demands, demRates[TMid], linkSet, wMatrix, MAXWEIGHT
        return self.__nodenum*(self.__nodenum-1), self.__demands, self.__linkset, self.__wMatrix, 999999

    def read_info_MA(self):
        pathNumList = [[] for _ in range(self.__regionnum)]
        pathNumListReduced = [[] for _ in range(self.__regionnum)]
        
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                pathNumList[sRegion].append(len(self.__oripathmaxtrix[src][dst]))
                if len(pathNumListReduced[sRegion]) == 0:
                    pathNumListReduced[sRegion].append([])
                    pathNumListReduced[sRegion].append([])
                if sRegion == tRegion:
                    pathNumListReduced[sRegion][0].append(len(self.__oripathmaxtrix[src][dst]))
                else:
                    pathNumListReduced[sRegion][1].append(len(self.__oripathmaxtrix[src][dst]))

        edgeNumListReduced =[0]*self.__regionnum
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if self.__interpathedgemap[src][dst] >= 0:
                    edgeNumListReduced[self.__interpathedgemap[src][dst]] += 1
        
        print("regionedgenum:", self.__regionedgenum)
        # print("pathNumList:", pathNumList)
        print(pathNumListReduced)
        return self.__nodenum, self.__linknum, self.__oripathmaxtrix, self.__demrates, self.__cMatrix, self.__regionnum, pathNumListReduced, [], self.__noderegionid

    def read_info_MMA(self):
        pathNumList = [[] for _ in range(self.__regionnum)]
        pathNumListReduced = [[] for _ in range(self.__regionnum)]
        pathNumMapRegion = []
        for _ in range(self.__nodenum):
            pathNumMapRegion.append([0]*self.__regionnum)
        
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                pathNumList[sRegion].append(len(self.__oripathmaxtrix[src][dst]))
                if sRegion == tRegion:
                    if len(pathNumListReduced[sRegion]) == 0:
                        pathNumListReduced[sRegion].append([])
                        pathNumListReduced[sRegion].append([])
                    pathNumListReduced[sRegion][0].append(len(self.__oripathmaxtrix[src][dst]))
                else:
                    pathNumMapRegion[src][tRegion] = len(self.__oripathmaxtrix[src][dst])

        self.__pathnummapregion = pathNumMapRegion
        for src in range(self.__nodenum):
            sRegion = self.__noderegionid[src]
            for tRegion in range(self.__regionnum):
                if sRegion == tRegion:
                    continue
                pathNumListReduced[sRegion][1].append(pathNumMapRegion[src][tRegion])

        edgeNumListReduced =[0]*self.__regionnum
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if self.__interpathedgemap[src][dst] >= 0:
                    edgeNumListReduced[self.__interpathedgemap[src][dst]] += 1
        
        print("regionedgenum:", self.__regionedgenum)
        # print("pathNumList:", pathNumList)

        return self.__nodenum, self.__linknum, self.__oripathmaxtrix, self.__demrates, self.__cMatrix, self.__regionnum, pathNumListReduced, pathNumMapRegion, self.__noderegionid

