from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np
import copy
import math
from numpy.linalg import solve
import os
import random

class SimEnv(Env):
    def __init__(self, pathPre, topo, synthesis_type, win_size, max_itr_num, log_name):
        print("SimEnv construction")
        self._update_count = 0
        self._win_size = win_size
        self._max_iter = max_itr_num
        self._failure_flag = False
        
        self._path_pre = pathPre
        self._topo_name = topo
        self._traffic_type = "testset"
        self._synthesis_type = "_" + synthesis_type
        self.get_topo()
        self.show_info()
        self.get_TMset()
        self.get_opt_vals()
        self._max_epoch = self._total_TMnum - self._win_size
        self._state = self._state_set[0]
        self._action_dim = 2*self._linknum
        self._state_dim = len(self._state_set[0])

        self._log_name = log_name
        dirpath = self._path_pre + "outputs/log/" + log_name
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        # self._log_maxutil = open(dirpath + "/maxutil.log", 'w', 1)
        # self._log_reward = open(dirpath + "/rwd.log", 'w', 1)
        # print("Log started")
        # else:
        self._log_maxutil = None
        self._log_reward = None

    def reset_env(self, failure_flag = False):
        self._path_pre = "/home/server/gengnan/NATE_project/"
        if not os.path.exists(self._path_pre):
            print("root dir ERROR")
            exit()
        # exit()
        self._failure_flag = failure_flag
        self.show_info()
        self._update_count = 0
        self._traffic_type = "testset"

        if not self._failure_flag:
            self._max_iter = 200
            self._index = [i for i in range(self._max_iter)]
            random.shuffle(self._index)
            dirpath = self._path_pre + "outputs/log/" + self._log_name.replace("train", "infer")
        else:
            self._max_iter = 100
            self._index = [i for i in range(self._max_iter)]
            dirpath = self._path_pre + "outputs/log/" + self._log_name.replace("train", "failure")
            self._TM_offset = 40 - self._win_size
            self.get_broken_links()
        self.get_TMset(True)
        self._max_epoch = self._total_TMnum - self._win_size
        self._state = self._state_set[0]
        self.get_opt_vals(True)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
            self._log_maxutil = open(dirpath + "/maxutil.log", 'w', 1)
            self._log_reward = open(dirpath + "/rwd.log", 'w', 1)
        else:
            self._log_maxutil = open(dirpath + "/maxutil.log", 'w', 1)
            self._log_reward = open(dirpath + "/rwd.log", 'w', 1)

    def get_topo(self):
        topo_file = self._path_pre + "inputs/region/" + self._topo_name + ".txt"
        file = open(topo_file)
        lines = file.readlines()
        file.close()
        lineList = lines[0].strip().split()
        self._nodenum = int(lineList[0])
        self._linknum = int(lineList[1])
        self._neighbors = [[] for _ in range(self._nodenum)]
        self._wMatrix = []
        self._cMatrix = []
        for i in range(self._nodenum):
            self._wMatrix.append([])
            self._cMatrix.append([0.0]*self._nodenum)
            for j in range(self._nodenum):
                if i == j:
                    self._wMatrix[i].append(0)
                else:
                    self._wMatrix[i].append(999999)
        self._linkset = []
        for i in range(1, self._linknum+1):
            lineList = lines[i].strip().split()
            left = int(lineList[0]) - 1
            right = int(lineList[1]) - 1
            weight = int(lineList[2])
            capa = float(lineList[3])
            
            self._linkset.append([left, right, weight, capa])
            self._wMatrix[left][right] = weight
            self._wMatrix[right][left] = weight
            self._cMatrix[left][right] = capa 
            self._cMatrix[right][left] = capa
            self._neighbors[left].append(right)
            self._neighbors[right].append(left)
    
    def show_info(self):
        print("Topology:%s(%d,%d)" % (self._topo_name, self._nodenum, self._linknum))

    def get_TMset(self, inferFlag = False):
        # get TM set and state set
        rate_file = self._path_pre + "inputs/traffic/" + self._traffic_type + "/" + self._topo_name + "_TMset%s.txt" % (self._synthesis_type)
        file = open(rate_file)
        lines = file.readlines()
        file.close()
        self._total_TMnum = min(len(lines), 200)
        if self._failure_flag:
            lines = lines[self._TM_offset:]
            self._total_TMnum = min(len(lines), 100)
        
        print("totally %d TMs" % self._total_TMnum)
        self._demrates = []
        for i in range(self._total_TMnum):
            lineList = lines[i].strip().split(',')
            rates = list(map(float, lineList))
            self._demrates.append(rates)
        if inferFlag:
            tmp = [self._demrates[idx] for idx in self._index]
            self._demrates = tmp

        self._state_set = []
        for epoch in range(self._total_TMnum - self._win_size):
            state_tmp = []
            for rateid in range(self._win_size):
                state_tmp += self._demrates[epoch+rateid]
            self._state_set.append(state_tmp)
    
    def get_opt_vals(self, inferFlag = False):
        if self._topo_name == "briten12r16grid":
            obj_file = self._path_pre + "outputs/objvals/" + self._topo_name + "_p3_3_1_obj_vals%s.txt" % (self._synthesis_type)
        else:
            obj_file = self._path_pre + "outputs/objvals/" + self._topo_name + "_mcf_obj_vals%s.txt" % (self._synthesis_type)
        infile = open(obj_file, "r")
        lines = infile.readlines()
        if self._failure_flag:
            lines = lines[self._TM_offset:]
        print("obj val num: %d" % len(lines))
        self._mcf_vals = []
        for line in lines:
            self._mcf_vals.append(float(line.strip()))
        infile.close()
        if inferFlag:
            tmp = [self._mcf_vals[idx] for idx in self._index]
            self._mcf_vals = tmp

    def get_reward(self, action):
        curr_demid = self._win_size + self._update_count % self._max_epoch
        curr_demrate = self._demrates[curr_demid]
        opt_val = self._mcf_vals[curr_demid]
        obj_val = self.get_obj_val(action, curr_demrate)
        reward = -1*obj_val/opt_val
        if self._log_reward != None:
            print(obj_val, file=self._log_maxutil)
            print(reward, file=self._log_reward)
        return reward
    
    def get_reward_failure(self, action, TM_offset = 40):
        # curr_demid = TM_offset + self._win_size + (self._update_count//10) % self._max_epoch
        curr_demid = self._win_size + (self._update_count//10) % self._max_epoch
        curr_demrate = self._demrates[curr_demid]
        opt_val = self._mcf_vals[curr_demid]
        self.convert_action(action)
        obj_val = self.get_obj_val(action, curr_demrate)
        reward = -1*obj_val/opt_val
        if self._log_reward != None:
            print(obj_val, file=self._log_maxutil)
            print(reward, file=self._log_reward)
        return reward
    
    def convert_action(self, action):
        brokenLinkIndex = self._broken_links[self._update_count][0]
        action[brokenLinkIndex*2] = 99
        action[brokenLinkIndex*2 + 1] = 99
        return action

    def get_broken_links(self):
        filein = open(self._path_pre + "inputs/brokenlink/" + self._topo_name + "_%dlinks.txt" % 100, 'r')
        lines = filein.readlines()
        brokenLinkList = []
        for line in lines:
            lineList = line.strip().split()
            brokenLinkList.append(list(map(int, lineList)))
        filein.close()
        self._broken_links = brokenLinkList

    def step(self, action):
        if self._update_count == -1:
            print("Inference!!!!")
            self.reset_env()
        if self._failure_flag: # self._failure_falg = Ture
            self._state = self._state_set[(self._update_count//10) % self._max_epoch]
            next_observation = np.copy(self._state)
            reward = self.get_reward_failure(action)
        else:
            self._state = self._state_set[self._update_count % self._max_epoch]
            next_observation = np.copy(self._state)
            reward = self.get_reward(action)
            
        if self._update_count % 100 == 0:
            print(self._update_count, "reward:", reward)
        self._update_count += 1
        if self._update_count == self._max_iter:
            if self._log_reward != None:
                self._log_maxutil.close()
                self._log_reward.close()
            self._log_reward = None
            self._log_maxutil = None
            self._demrates = []
            self._update_count = -1
        return Step(observation=next_observation, reward=reward, done=False)

    def compute_sp(self, wmatrix_ori):
        # rmatrix = []
        # for i in range(self._nodenum):
        #     rmatrix.append([i for i in range(self._nodenum)])

        wmatrix = copy.deepcopy(wmatrix_ori)
        for k in range(self._nodenum):
            for i in range(self._nodenum):
                for j in range(self._nodenum):
                    if wmatrix[i][j] > wmatrix[i][k] + wmatrix[k][j]:
                        wmatrix[i][j] = wmatrix[i][k] + wmatrix[k][j]
                        # rmatrix[i][j] = rmatrix[i][k]
        return wmatrix

    def get_split_ratio(self, action, gama = 2):
        edge_wMatrix = copy.deepcopy(self._wMatrix)
        for l in range(self._linknum):
            left = self._linkset[l][0]
            right = self._linkset[l][1]
            edge_wMatrix[left][right] = action[l*2]
            edge_wMatrix[right][left] = action[l*2+1]
        sp_weight_maxtrix = self.compute_sp(edge_wMatrix)

        alphas = []
        for t in range(self._nodenum):
            alphas.append([])
            for u in range(self._nodenum):
                if u == t:
                    alphas[t].append([])
                    continue
                path_weights = []
                for nei in self._neighbors[u]:
                    path_weight = edge_wMatrix[u][nei] + sp_weight_maxtrix[nei][t]
                    path_weights.append(math.exp(-1*gama*path_weight))
                total_weight = sum(path_weights)
                ratios = [item/total_weight for item in path_weights]
                alphas[t].append(ratios)
        return alphas

    def get_obj_val(self, action, demrate):
        alphas = self.get_split_ratio(action)

        flowMap = np.zeros((self._nodenum, self._nodenum))
        for t in range(self._nodenum):
            flowMap_t = self.get_flowmap_t(t, demrate[t*(self._nodenum-1):(t+1)*(self._nodenum-1)], alphas[t])
            flowMap = flowMap + flowMap_t

        # print(flowMap)
        flowMap = list(flowMap)
        netutil = []
        for u in range(self._nodenum):
            for v in range(self._nodenum):
                if self._cMatrix[u][v] > 0:
                    netutil.append(round((flowMap[u][v])/self._cMatrix[u][v], 4))
        maxutil = max(netutil)
        return maxutil

    def get_flowmap_t(self, dst, demsizes, alpha):
        flowmap = []
        for _ in range(self._nodenum):
            flowmap.append([0.0]*self._nodenum)
        
        a = [[] for _ in range(self._nodenum)]
        b = []

        offset = 0
        for u in range(self._nodenum):
            if u == dst:
                b.append(0)
                offset = -1
            else:
                b.append(demsizes[u+offset])
            for v in range(self._nodenum):
                if v == u:
                    a[u].append(1)
                    continue
                if v not in self._neighbors[u]:
                    a[u].append(0)
                    continue
                if v == dst:
                    a[u].append(0)
                    continue
                index = self._neighbors[v].index(u)
                a[u].append(-1*alpha[v][index])

        a=np.mat(a)
        b=np.mat(b).T
        #######
        res=solve(a,b)
        x = []
        for i in range(self._nodenum):
            x.append(res[i,0])
        #######

        for u in range(self._nodenum):
            if u == dst:
                continue
            for i in range(len(self._neighbors[u])):
                nei = self._neighbors[u][i]
                flowmap[u][nei] += x[u]*alpha[u][i]
        return np.array(flowmap)

    def reset(self):
        observation = np.copy(self._state)
        return observation

    @property
    def action_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        return Box(low=1, high=10, shape=(self._action_dim,))

    @property
    def observation_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        return Box(low=-np.inf, high=np.inf, shape=(self._state_dim,))

    def render(self):
        pass

