#!/usr/bin/python
#########################################################################
# File Name: main.py
# Author: gn
# Discription: main file of the simulation system
#########################################################################
from __future__ import division
import os
import numpy as np
import time

def get_stamp_type(time_stamp, topo, target, scheme, path_num, synthesis_type, rwd_flag, small_ratio, epochs, block_num, stamp_tail):
    stamp_type = "%s_%s_%s_%s_p%d_%s" % (time_stamp, topo, target, scheme, path_num, synthesis_type)
    if scheme == "MSA" or scheme == "MDA":
        stamp_type += "_rwd%d" % rwd_flag
        stamp_type += "_small%.2f" % small_ratio
    stamp_type += "_epoch%d" % (epochs)
    if topo == "briten12r16grid":
        stamp_type += "_%dblocks" % (block_num)
    if stamp_tail != "":
        stamp_type += '_' + stamp_tail
    return stamp_type

def run(path_pre, scheme, epochs, topo, synthesis_type, rwd_flag, small_ratio, block_num, train_start_index = 0, train_episodes = 40, infer_episodes = 200, failure_episodes = 10):
    path_type = "p%d_%d_%d" % (para_p[0], para_p[1], para_p[2])
    stamp_type = get_stamp_type(time_stamp, topo, target, scheme, path_num, synthesis_type, rwd_flag, small_ratio, epochs, block_num, stamp_tail)
    print(stamp_type)
    if target == "converge":
        cmd = "python3 lib/agent.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=1 --epochs=%d --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d" % (path_pre, stamp_type, scheme, epochs, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num)
        print(cmd)
        os.system(cmd)

    elif target == "coninfer":
        infer_stamp_type = get_stamp_type(time_stamp, topo, "converge", scheme, path_num, synthesis_type, rwd_flag, small_ratio, epochs, block_num, stamp_tail)
        ckpt_path = "%soutputs/ckpoint/%s/ckpt" % (path_pre, infer_stamp_type)
        cmd = "python3 lib/agent.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=1 --epochs=11 --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d --is_train=False --ckpt_path=%s" % (path_pre, stamp_type, scheme, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num, ckpt_path)
        print(cmd)
        os.system(cmd)

    elif target == "train":
        cmd = "python3 lib/agent.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=%d --epochs=%d --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d" % (path_pre, stamp_type, scheme, train_episodes, epochs, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num)
        print(cmd)
        os.system(cmd)

    elif target == "infer":
        infer_stamp_type = get_stamp_type(time_stamp, topo, "train", scheme, path_num, synthesis_type, rwd_flag, small_ratio, epochs, block_num, stamp_tail)
        ckpt_path = "%soutputs/ckpoint/%s/ckpt" % (path_pre, infer_stamp_type)
        cmd = "python3 lib/agent.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=%d --epochs=5 --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d --is_train=False --ckpt_path=%s" % (path_pre, stamp_type, scheme, infer_episodes, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num, ckpt_path)
        print(cmd)
        os.system(cmd)

        util_file = open(path_pre + "outputs/log/%s/util.log" % (stamp_type))
        utils = list(map(float, util_file.readlines()))

        result_file = open(path_pre + "outputs/log/%s/maxutils.result" % (stamp_type), "w")
        for i in range(infer_episodes):
            maxutil = np.mean(np.array(utils[i * 5 : (i + 1) * 5]))
            print(maxutil, file=result_file)
        result_file.close()

    elif target == "failure":
        infer_stamp_type = get_stamp_type(time_stamp, topo, "train", scheme, path_num, synthesis_type, rwd_flag, small_ratio, epochs, block_num, stamp_tail)
        ckpt_path = "%soutputs/ckpoint/%s/ckpt" % (path_pre, infer_stamp_type)
        train_start_index = 40
        epochs = 50
        cmd = "python3 lib/agent.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=%d --epochs=%d --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d --is_train=False --ckpt_path=%s --failure_flag=1" % (path_pre, stamp_type, scheme, failure_episodes, epochs, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num, ckpt_path)
        print(cmd)
        os.system(cmd)

        util_file = open(path_pre + "outputs/log/%s/util.log" % (stamp_type))
        utils = list(map(float, util_file.readlines()))
        result_file = open(path_pre + "outputs/log/%s/maxutils.result" % (stamp_type), "w")
        for i in range(failure_episodes):
            for j in range(10):
                maxutil = np.mean(np.array(utils[i*epochs + j*5 : i*epochs + (j+1)*5]))
                print(maxutil, file=result_file)
        result_file.close()

''' para setting area '''
path_pre = "/home/server/gengnan/NATE_project/"
time_stamp = "0311"
target = "failure"
scheme = "MDA" # MDA MSA ECMP
topo = "google" # 1221c google briten12r16grid
para_p = [3, 3, 1] # intra path num; gate num; gate path num
path_num = para_p[0]*100 + para_p[1]*10 + para_p[2]
synthesis_type = "gravNR250" # gravNR250 gravNR50c
epochs = 3000
rwd_flag = 0
small_ratio = 0.8
if topo == "briten12r16grid":
    block_num = 16
else:
    block_num = 0
stamp_tail = ""

run(path_pre, scheme, epochs, topo, synthesis_type, rwd_flag, small_ratio, block_num)



# scp -r ckpoint/dirname server@128.46.202.245:/home/server/gengnan/NATE_project/outputs/ckpoint/

