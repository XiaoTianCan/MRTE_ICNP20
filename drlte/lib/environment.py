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
import time
import _thread
import threading
import multiprocessing as mp

class Environment:
    def __init__(self, infile_prefix, topo_name, episode, epoch, start_index, train_flag, path_type, synthesis_type, small_ratio, failure_flag, block_num):
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
        self.__regionnodeneibor = [] # region-level

        self.__regionedgenum = []
        self.__regionnum = 0
        self.__noderegionid = []
        self.__bordernodes = []

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
        self.__start_index = start_index
        self.__epoch = epoch
        self.__episode = -1
        self.__maxepoch = episode * epoch
        self.__updatenum = 0
        self.__actionmatrix = []
        self.__smallratio = small_ratio
        self.__failure_flag = failure_flag
        self.__failurelink = None

        # multi process
        self.__procnum = 6
        self.__partitions = []

        # for grid topo; diff region num
        self.__blockflag = False
        if self.__toponame == "briten12r16grid":
            if block_num == 16:
                pass
            else:
                self.__blockflag = True
                if block_num == 8:
                    self.__blockrule = [
                        [0, 1], [2, 3], [4, 5], [6, 7], 
                        [8, 9], [10, 11], [12, 13], [14, 15]
                    ]
                elif block_num == 4:
                    self.__blockrule = [
                        [0, 1, 4, 5], [2, 3, 6, 7], 
                        [8, 9, 12, 13], [10, 11, 14, 15]
                    ]
                elif block_num == 2:
                    self.__blockrule = [
                        [0, 1, 2, 3, 4, 5, 6, 7], 
                        [8, 9, 10, 11, 12, 13, 14, 15]
                    ]
                elif block_num == 1:
                    self.__blockrule = [
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    ]
        
        # file path
        self.__topofile = infile_prefix + "inputs/region/" + topo_name + ".txt"
        self.__pathfile = infile_prefix + "inputs/pathset/" + topo_name + "_" + path_type + ".pickle"
        
        if train_flag and episode > 1:
            traffic_type = "trainset"
        else:
            traffic_type = "testset"
        self.__ratefile = infile_prefix + "inputs/traffic/" + traffic_type + "/" + topo_name + "_TMset_%s.txt" % (synthesis_type)

        # initial functions
        self.get_regions()
        # self.get_demands()
        self.get_paths()
        self.get_TMset()
        if self.__failure_flag == 1:
            self.__brokenlinkfile = infile_prefix + "inputs/brokenlink/" + topo_name + "_%dlinks.txt" % 100
            self.get_broken_link()
    
    def get_broken_link(self):
        filein = open(self.__brokenlinkfile, 'r')
        lines = filein.readlines()
        self.__brokenlinklist = []
        for line in lines:
            lineList = line.strip().split()
            self.__brokenlinklist.append(list(map(int, lineList[1:])))
        filein.close()

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
        self.__bordernodes = [[] for _ in range(self.__regionnum)]

        for i in range(1, self.__linknum+1):
            lineList = lines[i].strip().split()
            left = int(lineList[0]) - 1
            right = int(lineList[1]) - 1
            weight = int(lineList[2])
            capa = float(lineList[3])
            regionId = int(lineList[4])

            if regionId == -1:
                lRegion = self.__noderegionid[left]
                rRegion = self.__noderegionid[right]
                if left not in self.__bordernodes[lRegion]:
                    self.__bordernodes[lRegion].append(left)
                if right not in self.__bordernodes[rRegion]:
                    self.__bordernodes[rRegion].append(right)
            self.__linkset.append([left, right, weight, capa, regionId])
            self.__wMatrix[left][right] = weight
            self.__wMatrix[right][left] = weight
            self.__cMatrix[left][right] = capa 
            self.__cMatrix[right][left] = capa
            self.__edgemap[left][right] = self.__noderegionid[left]
            self.__edgemap[right][left] = self.__noderegionid[right]
            self.__regionedgenum[self.__noderegionid[left]] += 1
            self.__regionedgenum[self.__noderegionid[right]] += 1
        
        for i in range(self.__regionnum):
            self.__regionwMatrix.append([])
            for j in range(self.__regionnum):
                if i == j:
                    self.__regionwMatrix[i].append(0)
                else:
                    self.__regionwMatrix[i].append(999999)
        self.__regionnodeneibor = [[] for _ in range(self.__regionnum)]
        for i in range(self.__linknum+2, len(lines)): # region-level
            lineList = lines[i].strip().split()
            left = int(lineList[0])
            right = int(lineList[1])
            self.__regionnodeneibor[left].append(right)
            self.__regionnodeneibor[right].append(left)
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

    def com_action_matrix(self, actionList):
        # TODO: failure
        if self.__failure_flag == 1:
            actionList = self.action_failure(actionList)

        self.__actionmatrix = []
        for src in range(self.__nodenum):
            self.__actionmatrix.append([])
            for dst in range(self.__nodenum):
                self.__actionmatrix[src].append([])
        
        actCountList = [0]*self.__regionnum
        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                pathNum = len(self.__oripathmaxtrix[src][dst])
                if sRegion == tRegion:
                    if self.__smalldemidmap[demId] == 0:
                        action = actionList[sRegion*2][actCountList[sRegion]:actCountList[sRegion]+pathNum]
                        self.__actionmatrix[src][dst] = action
                        actCountList[sRegion] += pathNum
                    else:
                        action = [round(1.0/pathNum, 6)]*pathNum
                        if self.__failure_flag == 1:
                            edgepaths = self.convert_edge_paths(self.__oripathmaxtrix[src][dst])
                            action = self.rescale_action(pathNum, edgepaths, action)
                        self.__actionmatrix[src][dst] = action
                        # self.__actionmatrix[src][dst] = [1] + [0]*(pathNum-1)
                else:
                    actionRange = self.__actionrangemap[src][tRegion] # region-level
                    action = actionList[sRegion*2+1][actionRange[0]:actionRange[1]]
                    self.__actionmatrix[src][dst] = action
                demId += 1

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
                        if subsizes[gwid] <= 0.001:
                            continue
                        sources.append(gates[gwid])
                        sizes.append(subsizes[gwid])
        return flowmap

    def process_com_flowmap_pipe(self, start, end, conn):
        TM = self.__TM
        while True:
            try:
                msg = conn.recv()
                if msg[0] == 0:
                    actionmatrix = msg[1]
                elif msg[0] == 1:
                    TM = msg[1]
                    continue
                else:
                    pass
                # t0 = time.time()
                flowmap = []
                for _ in range(self.__nodenum):
                    flowmap.append([0.0]*self.__nodenum)
                for src in range(self.__nodenum):
                    for dst in range(start, end):
                        if src == dst:
                            continue
                        sources = [src]
                        sizes = [TM[src][dst]]
                        while True:
                            if len(sources) == 0:
                                break
                            pathSet = self.__oripathmaxtrix[sources[0]][dst]
                            action = actionmatrix[sources[0]][dst]
                            subsizes, gates = self.com_path_flow(flowmap, pathSet, action, sizes[0])
                            sources.pop(0)
                            sizes.pop(0)
                            for gwid in range(len(gates)):
                                if gates[gwid] == dst:
                                    continue
                                if subsizes[gwid] <= 0.001:
                                    continue
                                sources.append(gates[gwid])
                                sizes.append(subsizes[gwid])
                # t1 = time.time()
                # print("finish time", t1 - t0)
                conn.send(flowmap)
            except EOFError:
                break

    def compute_flowmap_paralell(self):
        # t0 = time.time()
        if self.__updatenum == 0:
            step = self.__nodenum//self.__procnum
            self.__partitions = [procid*step for procid in range(self.__procnum)] + [self.__nodenum]
            pool = mp.Pool(self.__procnum)
            self.__connList = []
            for procid in range(self.__procnum):
                parent_conn, child_conn = mp.Pipe()
                self.__connList.append(parent_conn)
                pool.apply_async(self.process_com_flowmap_pipe, args = (self.__partitions[procid], self.__partitions[procid+1], child_conn, ))

        if self.__updatenum%self.__epoch == 0:
            # print("TM", self.__updatenum)
            for parent_conn in self.__connList:
                parent_conn.send((1, self.__TM))

        # print("update", self.__updatenum)
        for parent_conn in self.__connList:
            parent_conn.send((0, self.__actionmatrix))
        
        flowmapList = []
        for parent_conn in self.__connList:
            flowmapList.append(parent_conn.recv())
            # print(parent_conn.recv())

        if self.__updatenum == self.__maxepoch - 1:
            # print("shut", self.__updatenum)
            for parent_conn in self.__connList:
                parent_conn.close()

        # t2 = time.time()
        # print("get", t2 - t1)

        flowmap = []
        for _ in range(self.__nodenum):
            flowmap.append([0.0]*self.__nodenum)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                for flowmap_sub in flowmapList:
                    flowmap[src][dst] += flowmap_sub[src][dst]
        # t3 = time.time()
        # print("add", t3 - t2)

        return flowmap

    def compute_util(self, flowmap):
        maxutilList = []
        netutilList = [[] for _ in range(self.__regionnum)]
        if self.__failure_flag == 0:
            for i in range(self.__nodenum):
                for j in range(self.__nodenum):
                    if self.__edgemap[i][j] >= 0:
                        util = round(flowmap[i][j]/self.__cMatrix[i][j], 4)
                        regionId = self.__edgemap[i][j]
                        netutilList[regionId].append(util)
            for netutil in netutilList:
                maxutilList.append(max(netutil))
            return maxutilList, netutilList
        else: # failure
            # TODO
            netutilList2 = [[] for _ in range(self.__regionnum)]
            for i in range(self.__nodenum):
                for j in range(self.__nodenum):
                    if self.__edgemap[i][j] >= 0:
                        regionId = self.__edgemap[i][j]
                        if (i,j) == self.__failurelink or (j,i) == self.__failurelink:
                            util = 1.0
                        else:
                            util = round(flowmap[i][j]/self.__cMatrix[i][j], 4)
                            netutilList2[regionId].append(util)
                        netutilList[regionId].append(util)
            for netutil in netutilList2:
                maxutilList.append(max(netutil))
            return maxutilList, netutilList

    def convert_block_action(self, actions):
        actionsSplit = [[] for _ in range(2*self.__regionnum)]
        for bid in range(self.__blocknum):
            for index in range(len(self.__blockrule[bid])):
                for agentType in range(2):
                    start = self.__actionBorderInBlock[bid][agentType][index]
                    end = self.__actionBorderInBlock[bid][agentType][index + 1]
                    actionsSplit[self.__blockrule[bid][index]*2+agentType] = actions[bid*2+agentType][start:end]
        return actionsSplit
    
    def convert_block_util(self, maxutilList, netutilList):
        maxutilListMerge = [0]*self.__blocknum
        netutilListMerge = [[] for _ in range(self.__blocknum)]
        for bid in range(self.__blocknum):
            for rid in self.__blockrule[bid]:
                maxutilListMerge[bid] = max([maxutilListMerge[bid], maxutilList[rid]])
                netutilListMerge[bid] += netutilList[rid]
        return maxutilListMerge, netutilListMerge

    def update(self, actions):
        if self.__updatenum % self.__epoch == 0:
            self.__episode += 1
            self.change_TM()
        if self.__blockflag:
            actions = self.convert_block_action(actions)
        self.com_action_matrix(actions)
        if self.__toponame == "briten12r16grid":
            flowmap = self.compute_flowmap_paralell()
        else:
            flowmap = self.compute_flowmap()

        # self.validate_correctness(flowmap) # comment self.change_TM() during validation
        
        maxutilList, netutilList = self.compute_util(flowmap)
        if self.__blockflag:
            maxutilList, netutilList = self.convert_block_util(maxutilList, netutilList)
        self.__updatenum += 1
        return max(maxutilList), maxutilList, netutilList
    
    def action_failure(self, actionList):
        failureLinkIndex = self.__episode*10 + (self.__updatenum%self.__epoch)//5
        left = self.__brokenlinklist[failureLinkIndex][0]
        right = self.__brokenlinklist[failureLinkIndex][1]
        self.__failurelink = (left, right)

        # rescale, self.__act2edgepath
        newActionList = []
        for agentId in range(len(actionList)):
            action = actionList[agentId]
            newAction = []
            count = 0
            for i in range(len(self.__act2edgepath[agentId])):
                pathNum = len(self.__act2edgepath[agentId][i])
                edgepaths = self.__act2edgepath[agentId][i]
                newAction += self.rescale_action(pathNum, edgepaths, action[count:count+pathNum])
                count += pathNum
            newActionList.append(newAction)
        return newActionList
    
    def rescale_action(self, pathNum, edgepaths, subaction):
        action_tmp = []
        action_flag = []
        split_more = 0.0
        (left, right) = self.__failurelink
        for j in range(pathNum):
            if (left, right) in edgepaths[j] or (right, left) in edgepaths[j]:
                action_tmp.append(0.0)
                action_flag.append(0)
                split_more += subaction[j]
            else:
                action_flag.append(1)
                action_tmp.append(subaction[j])

        sums = 0.0
        for i in range(len(action_flag)):
            if action_flag[i] == 1:
                sums += action_tmp[i]

        res = []
        if sum(action_flag) == 0:
            print(self.__updatenum, self.__failurelink, edgepaths)
            exit()
        if sums <= 0.0001:
            w = 1.0/sum(action_flag)
            for i in range(len(action_flag)):
                if action_flag[i] == 1:
                    res.append(w)
                else:
                    res.append(0.0)
        else:
            for i in range(len(action_flag)):
                if action_flag[i] == 1:
                    res.append(action_tmp[i] + (action_tmp[i]/sums)*split_more)
                else:
                    res.append(0.0)
        return res

    def validate_correctness(self, flowmap):
        for row in self.__TM:
            for item in row:
                print("%5s  " % str(round(item, 1)), end='')
            print('\n')
        for src in range(self.__nodenum):
            if src == 0:
                print("%5s  " % str(0), end = '')
                for dst in range(self.__nodenum):
                    print("%5s  " % str(dst), end='')
                print('\n')
            print("%5s  " % str(src), end = '')
            for dst in range(self.__nodenum):
                print("%5s  " % str(round(flowmap[src][dst], 0)), end='')
            print('\n')

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
                # if self.__noderegionid[i] != self.__noderegionid[j]:
                #     self.__TM[i][j] = 0.0
                # elif self.__noderegionid[i] == 0 or self.__noderegionid[i] == 1:
                #     self.__TM[i][j] = 0.0
                # else:
                #     self.__TM[i][j] = self.__demrate[demId]
                self.__TM[i][j] = self.__demrate[demId]
                demId += 1
        # print(self.__TM[0])
        # exit()

    def set_TM(self, s, t, size):
        self.__TM = []
        for i in range(self.__nodenum):
            self.__TM.append([0.0]*self.__nodenum)
        self.__TM[s][t] = size

    def show_info(self):
        print("--------------------------")
        print("----detail information----")
        print("topology:%s(%d,%d) with %d region(s)" % (self.__toponame, self.__nodenum, self.__linknum, self.__regionnum))
        print("--------------------------")

    def cal_terminal_demands(self, TM):
        ternimalTM = copy.deepcopy(TM)

        actionmatrix = []
        for src in range(self.__nodenum):
            actionmatrix.append([])
            for dst in range(self.__nodenum):
                if src == dst:
                    actionmatrix[src].append([])
                else:
                    action = [1.0/len(self.__oripathmaxtrix[src][dst])]*len(self.__oripathmaxtrix[src][dst])
                    actionmatrix[src].append([round(item, 6) for item in action])

        flowmap = []
        for _ in range(self.__nodenum):
            flowmap.append([0.0]*self.__nodenum)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                srid = self.__noderegionid[src]
                trid = self.__noderegionid[dst]
                if srid == trid:
                    continue
                sources = [src]
                sizes = [TM[src][dst]]
                ingressNodes = {}
                while True:
                    if len(sources) == 0:
                        break
                    pathSet = self.__oripathmaxtrix[sources[0]][dst]
                    action = actionmatrix[sources[0]][dst]
                    subsizes, gates = self.com_path_flow(flowmap, pathSet, action, sizes[0])
                    sources.pop(0)
                    sizes.pop(0)
                    for gwid in range(len(gates)):
                        if gates[gwid] == dst:
                            continue
                        sources.append(gates[gwid])
                        sizes.append(subsizes[gwid])
                        if self.__noderegionid[gates[gwid]] == trid:
                            if gates[gwid] not in ingressNodes:
                                ingressNodes[gates[gwid]] = subsizes[gwid]
                            else:
                                ingressNodes[gates[gwid]] += subsizes[gwid]
                for ingress in ingressNodes.keys():
                    ternimalTM[ingress][dst] += ingressNodes[ingress]

        return ternimalTM

    def sort_intra_demand(self, aveNum = 40):
        # 1. get average demand rates
        demandNum = len(self.__demrates[0])
        demrate = np.array([0]*demandNum)
        for i in range(aveNum):
            rate = np.array(self.__demrates[i])
            demrate = demrate + rate
        demrate /= aveNum
        TM = []
        demId = 0
        for i in range(self.__nodenum):
            TM.append([0.0]*self.__nodenum)
            for j in range(self.__nodenum):
                if i == j:
                    continue
                TM[i][j] = demrate[demId]
                demId += 1
        ternimalTM = self.cal_terminal_demands(TM)

        # 2. get region demand rates
        regionRates = [[] for _ in range(self.__regionnum)]
        regionDemIds = [[] for _ in range(self.__regionnum)]
        totalTraffic_tmp = [[0, 0] for _ in range(self.__regionnum)]
        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                if sRegion == tRegion:
                    regionRates[sRegion].append(ternimalTM[src][dst])
                    # if (src not in self.__bordernodes[sRegion]):
                    #     regionDemIds[sRegion].append(0)
                    # else:
                    #     regionDemIds[sRegion].append(1)
                    regionDemIds[sRegion].append(demId)
                    totalTraffic_tmp[sRegion][0] += ternimalTM[src][dst]
                else:
                    totalTraffic_tmp[sRegion][1] += ternimalTM[src][dst]
                demId += 1
        # print(totalTraffic_tmp)
        # print(self.__bordernodes)
        # 3. sort region's demands
        smallDemIdMap = [0]*demandNum
        for rid in range(self.__regionnum):
            index = np.argsort(regionRates[rid])
            res = [round(regionRates[rid][i], 0) for i in index]
            index = index[:int(len(regionRates[rid])*self.__smallratio)]
            # print("small demand num", int(len(regionRates[rid])), len(index))
            for i in index:
                smallDemIdMap[regionDemIds[rid][i]] = 1
        # exit()

        return smallDemIdMap

    def sort_intra_demand_old(self, aveNum = 40):
        # 1. get average demand rates
        demandNum = len(self.__demrates[0])
        demrate = np.array([0]*demandNum)
        for i in range(aveNum):
            rate = np.array(self.__demrates[i])
            demrate = demrate + rate
        demrate /= aveNum

        # 2. get region demand rates
        regionRates = [[] for _ in range(self.__regionnum)]
        regionDemIds = [[] for _ in range(self.__regionnum)]
        totalTraffic_tmp = [[0, 0] for _ in range(self.__regionnum)]
        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                if sRegion == tRegion and (src not in self.__bordernodes[sRegion]):
                    regionRates[sRegion].append(demrate[demId])
                    regionDemIds[sRegion].append(demId)
                    totalTraffic_tmp[sRegion][0] += demrate[demId]
                else:
                    totalTraffic_tmp[sRegion][1] += demrate[demId]
                demId += 1
        # print(totalTraffic_tmp)
        # print(self.__bordernodes)
        # 3. sort region's demands
        smallDemIdMap = [0]*demandNum
        for rid in range(self.__regionnum):
            index = np.argsort(regionRates[rid])
            res = [round(regionRates[rid][i], 0) for i in index]
            index = index[:int(len(regionRates[rid])*(1-self.__smallratio))]
            for i in index:
                smallDemIdMap[regionDemIds[rid][i]] = 1
        # print(smallDemIdMap)
        # print(sum(smallDemIdMap))
        # fileout = open("smallDemIdMap.txt", 'w')
        # print(smallDemIdMap, file = fileout)
        # fileout.close()
        # print("small demand num", int(len(regionRates[rid])), demandNum)
        # exit()

        return smallDemIdMap

    def convert_edge_paths(self, paths):
        edgepaths = []
        for path in paths:
            pathLen = len(path)
            epath = []
            for l in range(pathLen-1):
                epath.append((path[l], path[l+1]))
            edgepaths.append(epath)
        return edgepaths

    def get_info(self):
        self.__smalldemidmap = self.sort_intra_demand()

        pathNumListDuel = [[] for _ in range(self.__regionnum)]
        pathNumMapRegion = []
        for _ in range(self.__nodenum):
            pathNumMapRegion.append([0]*self.__regionnum)
        
        # TODO: failure
        self.__act2edgepath = [[] for _ in range(self.__regionnum*2)]
        edgepathsMapRegion = []
        for i in range(self.__nodenum):
            edgepathsMapRegion.append([])
            for _ in range(self.__regionnum):
                edgepathsMapRegion[i].append([])

        demId = 0
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                sRegion = self.__noderegionid[src]
                tRegion = self.__noderegionid[dst]
                if sRegion == tRegion:
                    if self.__smalldemidmap[demId] == 0:
                        if len(pathNumListDuel[sRegion]) == 0:
                            pathNumListDuel[sRegion].append([])
                            pathNumListDuel[sRegion].append([])
                        pathNumListDuel[sRegion][0].append(len(self.__oripathmaxtrix[src][dst]))
                        edgepaths = self.convert_edge_paths(self.__oripathmaxtrix[src][dst])
                        self.__act2edgepath[sRegion*2].append(edgepaths)
                else:
                    pathNumMapRegion[src][tRegion] = len(self.__oripathmaxtrix[src][dst])
                    edgepaths = self.convert_edge_paths(self.__oripathmaxtrix[src][dst])
                    edgepathsMapRegion[src][tRegion] = edgepaths
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
                self.__act2edgepath[sRegion*2+1].append(edgepathsMapRegion[src][tRegion]) # failure
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
        # print("regionNodeNeibor:", self.__regionnodeneibor)
        # exit()
        if not self.__blockflag:
            return self.__regionnum, self.__regionedgenum, pathNumListDuel, self.__regionnodeneibor
        
        print("\nBlock Block Block")
        blockNum = len(self.__blockrule)
        self.__blocknum = blockNum
        regionEdgeNum = [0]*blockNum
        pathNumListDuelBlock = []
        regionNodeNeibor = [[] for _ in range(blockNum)]
        ridMap = [0]*self.__regionnum
        self.__actionBorderInBlock = []
        for bid in range(blockNum):
            pathNumListDuelBlock.append([[], []])
            self.__actionBorderInBlock.append([[0], [0]])
            for rid in self.__blockrule[bid]:
                regionEdgeNum[bid] += self.__regionedgenum[rid]
                pathNumListDuelBlock[bid][0] += pathNumListDuel[rid][0]
                pathNumListDuelBlock[bid][1] += pathNumListDuel[rid][1]
                self.__actionBorderInBlock[bid][0].append(sum(pathNumListDuelBlock[bid][0]))
                self.__actionBorderInBlock[bid][1].append(sum(pathNumListDuelBlock[bid][1]))
                ridMap[rid] = bid
        
        for bid in range(blockNum):
            for rid in self.__blockrule[bid]:
                for nrid in self.__regionnodeneibor[rid]:
                    if ridMap[nrid] not in regionNodeNeibor[bid] and ridMap[nrid] != bid:
                        regionNodeNeibor[bid].append(ridMap[nrid])
        print("regionEdgeNum:", regionEdgeNum)
        print("regionNodeNeibor:", regionNodeNeibor)
        print("self.__actionBorderInBlock", self.__actionBorderInBlock)
        # exit()
        return blockNum, regionEdgeNum, pathNumListDuelBlock, regionNodeNeibor
        
        

