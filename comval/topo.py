#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Objective: routing selection idea (TM classification idea)
'''

from __future__ import division
import numpy as np

class ReadTopo:
    def __init__(self, infile_prefix, topo_name, path_type = "", traffic_type = "testset", path_num = 3, burstFlag = 0, synthesis_type = ""):
        self.__topofile = infile_prefix + "region/" + topo_name + ".txt"
        if path_type in ["racke", "mcf", "shr"]:
            self.__pathfile = infile_prefix + "pathset/" + topo_name + "_path" + str(path_num) + "_" + path_type + ".txt"
        elif path_type == "region":
            self.__pathfile = infile_prefix + "pathset/" + topo_name + "_path" + str(path_num) + ".txt"
        else:
            self.__pathfile = infile_prefix + "pathset/" + topo_name + "_paths_" + path_type + ".txt"

        # self.__ratefile = infile_prefix + "traffic/" + traffic_type + "/" + topo_name + "_TMset.txt"
        self.__ratefile = infile_prefix + "traffic/" + traffic_type + "/" + topo_name + "_TMset%s.txt" % (synthesis_type)
        
        # store topo info
        self.__nodenum = 0
        self.__linknum = 0
        self.__linkset = []
        self.__edgemap = []
        self.__wMatrix = []
        self.__cMatrix = []
        self.__MAXWEIGHT = 99999

        # store self.__demands
        self.__demnum = 0
        self.__demands = []

        # store paths and rates
        self.__totalpathnum = 0
        self.__totalTMnum = 0
        self.__candidatepaths = []
        self.__demrates = []
        self.__pathnum = []

        # for failure test
        self.__updatenum = 0
        self.__rate_ind = 0
        self.__link_ind = 0
        
        self.get_topo()
        self.get_demands()
        if path_type != "":
            self.get_paths()
        self.get_demrates()

    def get_topo(self):
        file = open(self.__topofile)
        lines = file.readlines()
        file.close()
        lineList = lines[0].strip().split()
        self.__nodenum = int(lineList[0])
        self.__linknum = int(lineList[1])
        for i in range(self.__nodenum):
            self.__wMatrix.append([])
            self.__cMatrix.append([])
            for j in range(self.__nodenum):
                if i == j:
                    self.__wMatrix[i].append(0)
                else:
                    self.__wMatrix[i].append(self.__MAXWEIGHT)
                self.__cMatrix[i].append(0.0)

        for i in range(1, self.__linknum+1):
            lineList = lines[i].strip().split()
            left = int(lineList[0]) - 1
            right = int(lineList[1]) - 1
            capa = float(lineList[3])
            weight = int(lineList[2])
            # regionId = int(lineList[4])
            # if regionId == -1:
            # capa = 9953
            self.__linkset.append([left, right, weight, capa])
            self.__wMatrix[left][right] = weight
            self.__wMatrix[right][left] = weight
            self.__cMatrix[left][right] = capa 
            self.__cMatrix[right][left] = capa

    def get_demands(self):
        self.__demnum = self.__nodenum*(self.__nodenum - 1)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                self.__demands.append([src,dst])

    def get_paths(self):
        file = open(self.__pathfile)
        lines = file.readlines()
        file.close()
        demId = 0
        candidatepaths = []
        self.__totalpathnum = len(lines)
        for i in range(self.__totalpathnum):
            lineList = lines[i].strip().split()
            path = list(map(int, lineList))
            if self.__demands[demId][0] != path[0] or self.__demands[demId][1] != path[-1]:
                self.__candidatepaths.append(candidatepaths)
                demId += 1
                candidatepaths = []
            candidatepaths.append(path)
        self.__candidatepaths.append(candidatepaths)
        for i in range(self.__demnum):
            self.__pathnum.append(len(self.__candidatepaths[i]))

        for i in range(self.__nodenum):
            self.__edgemap.append([])
            for j in range(self.__nodenum):
                self.__edgemap[i].append(0)
        for i in range(self.__demnum):
            for j in range(self.__pathnum[i]):
                for k in range(len(self.__candidatepaths[i][j])-1):
                    enode1 = self.__candidatepaths[i][j][k]
                    enode2 = self.__candidatepaths[i][j][k+1]
                    self.__edgemap[enode1][enode2] = 1

    def get_demrates(self):
        file = open(self.__ratefile)
        lines = file.readlines()
        file.close()
        self.__totalTMnum = len(lines)
        for i in range(self.__totalTMnum):
            lineList = lines[i].strip().split(',')
            rates = list(map(float, lineList))
            self.__demrates.append(rates)

    def change_pathset(self, pathset):
        self.__candidatepaths = pathset
        for i in range(self.__demnum): 
            self.__pathnum[i] = len(pathset[i])

    def calVal(self, action, rates):
        if action == []:
            for item in self.__pathnum:
                action += [round(1.0/item, 6) for i in range(item)]
        flowmap = []
        subrates = []
        count = 0
        # get split rates
        for i in range(self.__demnum):
            subrates.append([])
            for j in range(self.__pathnum[i]):
                tmp = 0
                if j == self.__pathnum[i] - 1:
                    tmp = rates[i] - sum(subrates[i])
                else:
                    tmp = rates[i]*action[count]
                count += 1
                subrates[i].append(tmp)
        # get flowmap
        for i in range(self.__nodenum):
            flowmap.append([])
            for j in range(self.__nodenum):
                flowmap[i].append(0.0)

        for i in range(self.__demnum): 
            for j in range(self.__pathnum[i]):
                for k in range(len(self.__candidatepaths[i][j])-1):
                    enode1 = self.__candidatepaths[i][j][k]
                    enode2 = self.__candidatepaths[i][j][k+1]
                    flowmap[enode1][enode2] += subrates[i][j]
        # calculate utilization
        netutil = []
        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                if self.__edgemap[i][j] != 0:
                    netutil.append(round((flowmap[i][j])/self.__cMatrix[i][j], 4))
        maxutil = max(netutil)
        return maxutil

    def read_info(self):
        return self.__nodenum, self.__linknum, self.__linkset, self.__demnum, self.__demands, self.__totalpathnum, self.__candidatepaths, self.__demrates, self.__cMatrix, self.__wMatrix, self.__MAXWEIGHT

    def show_info(self):
        # store topo info
        print(self.__nodenum)
        print(self.__linknum)
        print(self.__linkset)
        print(self.__wMatrix)
        print(self.__cMatrix)
        print(self.__MAXWEIGHT)
        # store demands
        print(self.__demnum)
        print(self.__demands)
        # store paths and rates
        print(self.__totalpathnum)
        print(self.__totalTMnum)
        print(self.__candidatepaths[0])
        print(self.__demrates[0])

    def update(self):
        # testing the broken link edge
        self.updateCapaMatrix()
        # print(self.__updatenum, self.__rate_ind, self.__link_ind)
        self.__updatenum += 1

    def updateCapaMatrix(self):
        decay_rates = [1, 0.9, 0.7, 0.5, 0.3, 0.1]

        left = self.__linkset[self.__link_ind][0]
        right = self.__linkset[self.__link_ind][1]
        self.__cMatrix[left][right] /= decay_rates[self.__rate_ind]
        self.__cMatrix[right][left] /= decay_rates[self.__rate_ind]
        self.__rate_ind = self.__updatenum // (self.__linknum)
        self.__link_ind = self.__updatenum % (self.__linknum)
        left = self.__linkset[self.__link_ind][0]
        right = self.__linkset[self.__link_ind][1]
        self.__cMatrix[left][right] *= decay_rates[self.__rate_ind]
        self.__cMatrix[right][left] *= decay_rates[self.__rate_ind]
        # print(left, right)
