#!/usr/bin/python
#########################################################################
# File Name: run.py
# Discription: parameter setting and system running
#########################################################################
from __future__ import division
import os, sys
import numpy as np
import time

def get_stamp_type(time_stamp, topo, target, scheme, path_num, synthesis_type, rwd_flag, small_ratio, epochs, block_num, stamp_tail):
    stamp_type = "%s_%s_%s_%s_p%d_%s" % (time_stamp, topo, target, scheme, path_num, synthesis_type)
    if scheme == "MSA" or scheme == "MDA":
        stamp_type += "_rwd%d" % rwd_flag
        stamp_type += "_small%.2f" % small_ratio
    stamp_type += "_epoch%d" % (epochs)
    if stamp_tail != "":
        stamp_type += '_' + stamp_tail
    return stamp_type

def run(path_pre, scheme, epochs, topo, rwd_flag, small_ratio, block_num = 0, path_num = 331, train_start_index = 0, synthesis_type = "gravNR250", train_episodes = 40, infer_episodes = 200, failure_episodes = 10):
    path_type = "p%d_%d_%d" % (path_num//100, path_num%100//10, path_num%10)
    stamp_type = get_stamp_type(time_stamp, topo, target, scheme, path_num, synthesis_type, rwd_flag, small_ratio, epochs, block_num, stamp_tail)
    print(stamp_type)
    if target == "converge":
        cmd = "python3 lib/main.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=1 --epochs=%d --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d" % (path_pre, stamp_type, scheme, epochs, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num)
        print(cmd)
        os.system(cmd)

    elif target == "train":
        cmd = "python3 lib/main.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=%d --epochs=%d --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d" % (path_pre, stamp_type, scheme, train_episodes, epochs, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num)
        print(cmd)
        os.system(cmd)

    elif target == "infer":
        infer_stamp_type = get_stamp_type(time_stamp, topo, "train", scheme, path_num, synthesis_type, rwd_flag, small_ratio, epochs, block_num, stamp_tail)
        ckpt_path = "%soutputs/ckpoint/%s/ckpt" % (path_pre, infer_stamp_type)
        cmd = "python3 lib/main.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=%d --epochs=5 --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d --is_train=False --ckpt_path=%s" % (path_pre, stamp_type, scheme, infer_episodes, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num, ckpt_path)
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
        cmd = "python3 lib/main.py --path_pre=%s --stamp_type=%s --agent_type=%s --episodes=%d --epochs=%d --topo_name=%s --train_start_index=%d  --synthesis_type=%s --rwd_flag=%d --path_type=%s --small_ratio=%f --block_num=%d --is_train=False --ckpt_path=%s --failure_flag=1" % (path_pre, stamp_type, scheme, failure_episodes, epochs, topo, train_start_index, synthesis_type, rwd_flag, path_type, small_ratio, block_num, ckpt_path)
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
path_pre = os.getcwd().replace("drlte", "")
time_stamp = "test"             # prefix of the log file name
stamp_tail = ""                 # suffix of the log file name
topo = "1221"                   # topology name
target = "converge"             # chioce: converge, train, infer, failure
scheme = "MDA"                  # chioce: MDA (two-agent design), MSA (one-agent design)
epochs = 3000                   # total update steps for each TM
rwd_flag = 0                    # set to 0 for MDA, set to 0 or 1 for MSA (indicates two diff. rwd functions)
small_ratio = 0.8               # small traffic ratio, i.e., \rho

run(path_pre, scheme, epochs, topo, rwd_flag, small_ratio)
