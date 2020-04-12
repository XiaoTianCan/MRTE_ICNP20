from drlAgent import *
from environment import Environment
import sys, os

if not hasattr(sys, 'argv'):
    sys.argv  = ['']

IS_TRAIN = getattr(FLAGS, "is_train")
RWD_FLAG = getattr(FLAGS, "rwd_flag")
ACTOR_LEARNING_RATE = getattr(FLAGS, 'learning_rate_actor')
CRITIC_LEARNING_RATE = getattr(FLAGS, 'learning_rate_critic')

GAMMA = getattr(FLAGS, 'gamma')
TAU = getattr(FLAGS, 'tau')

EP_BEGIN = getattr(FLAGS, 'epsilon_begin')
EP_END = getattr(FLAGS, 'epsilon_end')

BUFFER_SIZE = getattr(FLAGS, 'size_buffer')
MINI_BATCH = getattr(FLAGS, 'mini_batch')

MAX_EPISODES = getattr(FLAGS, 'episodes')
MAX_EP_STEPS = getattr(FLAGS, 'epochs')

if getattr(FLAGS, 'stamp_type') == '':
    REAL_STAMP = str(datetime.datetime.now())
else:
    REAL_STAMP = getattr(FLAGS, 'stamp_type')
PATHPRE = getattr(FLAGS, 'path_pre')

AGENT_TYPE = getattr(FLAGS, "agent_type")

CKPT_PATH = getattr(FLAGS, "ckpt_path")
TOPO_NAME = getattr(FLAGS, "topo_name")
SYNT_TYPE = getattr(FLAGS, "synthesis_type")
PATH_TYPE = getattr(FLAGS, "path_type")
FAILURE_FLAG = getattr(FLAGS, "failure_flag")
START_INDEX = getattr(FLAGS, "train_start_index")
SMALL_RATIO = getattr(FLAGS, "small_ratio")
BLOCK_NUM = getattr(FLAGS, "block_num")


initActions = []
def init_action(regionNum, pathNumList):
    res = []
    for regionId in range(regionNum):
        action = []
        for item in pathNumList[regionId]:
            action += [round(1.0/item, 6) for i in range(item)]
        res.append(action)
    return res


def update_step(maxutil, maxutilList, netutilList, agents, regionNodeNeibor, actionBorderline):
    if AGENT_TYPE == "MDAs":
        actions = []
        agentNum = len(agents)//2
        if RWD_FLAG == 0:
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                if agentid % 2 == 0:
                    reward = -1*maxutilList[agentid//2]
                    result1 = agents[2*agentid].predict(state, reward)
                    result2 = agents[2*agentid+1].predict(state, reward)
                    result = list(result1) + list(result2)
                else:
                    maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid//2]]
                    if len(maxutil_nei) == 0:
                        reward = -0.7*maxutilList[agentid//2]
                    else:
                        reward = -0.7*maxutilList[agentid//2] - 0.3*sum(maxutil_nei)/len(maxutil_nei)
                    result1 = agents[2*agentid].predict(state, reward)
                    result2 = agents[2*agentid+1].predict(state, reward)
                    result = list(result1) + list(result2)
                actions.append(result)
        return actions

    elif AGENT_TYPE == "MDA":
        actions = []
        agentNum = len(agents)
        if RWD_FLAG == 0:
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                if agentid % 2 == 0:
                    reward = -1*maxutilList[agentid//2]
                    result = agents[agentid].predict(state, reward)
                    # result = initActions[agentid]
                else:
                    maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid//2]]
                    if len(maxutil_nei) == 0:
                        reward = -0.7*maxutilList[agentid//2]
                    else:
                        reward = -0.7*maxutilList[agentid//2] - 0.3*sum(maxutil_nei)/len(maxutil_nei)
                    result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c)
        elif RWD_FLAG == 1:
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                reward = -1*maxutilList[agentid//2]
                result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c)
        elif RWD_FLAG == 2:
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid//2]]
                reward = -0.7*maxutilList[agentid//2] - 0.3*sum(maxutil_nei)/len(maxutil_nei)
                result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c)
        elif RWD_FLAG == 3:
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                reward = -1*maxutil
                result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c)
        elif RWD_FLAG == 4:
            beta0 = 0.3
            beta1 = 1 - beta0
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                if agentid % 2 == 0:
                    reward = -1*maxutilList[agentid//2]
                    result = agents[agentid].predict(state, reward)
                else:
                    maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid//2]]
                    reward = -beta0*maxutilList[agentid//2] - beta1*sum(maxutil_nei)/len(maxutil_nei)
                    result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c)
        elif RWD_FLAG == 5:
            beta0 = 0.5
            beta1 = 1 - beta0
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                if agentid % 2 == 0:
                    reward = -1*maxutilList[agentid//2]
                    result = agents[agentid].predict(state, reward)
                else:
                    maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid//2]]
                    reward = -beta0*maxutilList[agentid//2] - beta1*sum(maxutil_nei)/len(maxutil_nei)
                    result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c)
        elif RWD_FLAG == 6:
            beta0 = 0.0
            beta1 = 1 - beta0
            for agentid in range(agentNum):
                state = netutilList[agentid//2]
                if agentid % 2 == 0:
                    reward = -1*maxutilList[agentid//2]
                    result = agents[agentid].predict(state, reward)
                else:
                    maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid//2]]
                    reward = -beta0*maxutilList[agentid//2] - beta1*sum(maxutil_nei)/len(maxutil_nei)
                    result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c)
        else:
            print("reward flag ERROR")
            exit()
        return actions
    
    elif AGENT_TYPE == "MSA":
        actions = []
        agentNum = len(agents)
        if RWD_FLAG == 1:
            for agentid in range(agentNum):
                state = netutilList[agentid]
                reward = -1*maxutilList[agentid]
                result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c[:actionBorderline[agentid]])
                actions.append(ret_c[actionBorderline[agentid]:])
        elif RWD_FLAG == 2:
            for agentid in range(agentNum):
                state = netutilList[agentid]
                maxutil_nei = [maxutilList[nrid] for nrid in regionNodeNeibor[agentid]]
                reward = -0.7*maxutilList[agentid] - 0.3*sum(maxutil_nei)/len(maxutil_nei)
                result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c[:actionBorderline[agentid]])
                actions.append(ret_c[actionBorderline[agentid]:])
        elif RWD_FLAG == 3:
            for agentid in range(agentNum):
                state = netutilList[agentid]
                reward = -1*maxutil
                result = agents[agentid].predict(state, reward)
                ret_c = list(result)
                actions.append(ret_c[:actionBorderline[agentid]])
                actions.append(ret_c[actionBorderline[agentid]:])
        else:
            print("reward flag ERROR")
            exit()
        return actions

    elif AGENT_TYPE == "ECMP":
        return initActions
    else:
        pass

def init_multi_agent(globalSess):
    env = Environment(PATHPRE, TOPO_NAME, MAX_EPISODES, MAX_EP_STEPS, START_INDEX, IS_TRAIN, PATH_TYPE, SYNT_TYPE, SMALL_RATIO, FAILURE_FLAG, BLOCK_NUM)

    regionNum, edgeNumList, pathNumListDual, regionNodeNeibor = env.get_info()
    if AGENT_TYPE == "MDA":
        print("\nConstructing MDA multiple agents ...")
        agents = []
        for regionId in range(regionNum):
            print("Region%d .." % regionId)
            dimState = edgeNumList[regionId]
            dimAction = sum(pathNumListDual[regionId][0])
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListDual[regionId][0], ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)

            dimState = edgeNumList[regionId]
            dimAction = sum(pathNumListDual[regionId][1])
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListDual[regionId][1], ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)
        
        # parameters init  
        print("Running global_variables initializer ...")
        globalSess.run(tf.global_variables_initializer())
        
        # build target actor and critic para
        print("Building target network ...")
        for agentid in range(len(agents)):
            agents[agentid].target_paras_init()
        
        # parameters restore
        mSaver = tf.train.Saver(tf.trainable_variables()) 
        if CKPT_PATH != None and CKPT_PATH != "":
            print("restore paramaters...")
            mSaver.restore(globalSess, CKPT_PATH)
        
        initActions = []
        for regionId in range(regionNum):
            initActions += init_action(2, pathNumListDual[regionId])
        return env, agents, initActions, regionNodeNeibor, []

    elif AGENT_TYPE == "MDAs":
        print("\nConstructing MDAs multiple agents ...")
        agents = []
        for regionId in range(regionNum):
            print("Region%d .." % regionId)
            dimState = edgeNumList[regionId]
            pathNumListTmp = pathNumListDual[regionId][0][:len(pathNumListDual[regionId][0])//2]
            dimAction = sum(pathNumListTmp)
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListTmp, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)

            pathNumListTmp = pathNumListDual[regionId][0][len(pathNumListDual[regionId][0])//2:]
            dimAction = sum(pathNumListTmp)
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListTmp, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)

            pathNumListTmp = pathNumListDual[regionId][1][:len(pathNumListDual[regionId][1])//2]
            dimAction = sum(pathNumListTmp)
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListTmp, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)

            pathNumListTmp = pathNumListDual[regionId][1][len(pathNumListDual[regionId][1])//2:]
            dimAction = sum(pathNumListTmp)
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListTmp, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)
        
        # parameters init  
        print("Running global_variables initializer ...")
        globalSess.run(tf.global_variables_initializer())
        
        # build target actor and critic para
        print("Building target network ...")
        for agentid in range(len(agents)):
            agents[agentid].target_paras_init()
        
        # parameters restore
        mSaver = tf.train.Saver(tf.trainable_variables()) 
        if CKPT_PATH != None and CKPT_PATH != "":
            print("restore paramaters...")
            mSaver.restore(globalSess, CKPT_PATH)
        
        initActions = []
        for regionId in range(regionNum):
            initActions += init_action(2, pathNumListDual[regionId])
        return env, agents, initActions, regionNodeNeibor, []

    elif AGENT_TYPE == "MSA":
        print("\nConstructing MSA agents ...")
        agents = []
        actionBorderline = []
        for regionId in range(regionNum):
            print("Region%d .." % regionId)
            dimState = edgeNumList[regionId]
            dimAction = sum(pathNumListDual[regionId][0]) + sum(pathNumListDual[regionId][1])
            agent = DrlAgent(globalSess, IS_TRAIN, dimState, dimAction, pathNumListDual[regionId][0] + pathNumListDual[regionId][1], ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, BUFFER_SIZE, MINI_BATCH, EP_BEGIN, EP_END, GAMMA, MAX_EP_STEPS)
            agents.append(agent)
            actionBorderline.append(sum(pathNumListDual[regionId][0]))
        
        # parameters init  
        print("Running global_variables initializer ...")
        globalSess.run(tf.global_variables_initializer())
        
        # build target actor and critic para
        print("Building target network ...")
        for agentid in range(len(agents)):
            agents[agentid].target_paras_init()
        
        # parameters restore
        mSaver = tf.train.Saver(tf.trainable_variables()) 
        if CKPT_PATH != None and CKPT_PATH != "":
            print("restore paramaters...")
            mSaver.restore(globalSess, CKPT_PATH)
        
        initActions = []
        for regionId in range(regionNum):
            initActions += init_action(2, pathNumListDual[regionId])
        return env, agents, initActions, regionNodeNeibor, actionBorderline
    elif AGENT_TYPE == "ECMP":
        initActions = []
        for regionId in range(regionNum):
            initActions += init_action(2, pathNumListDual[regionId])
        return env, [], initActions, regionNodeNeibor, []
    else:
        print("Scheme type error")
        exit()

def log_to_file(maxutil, fileUtilOut, netutilList, fileEdgeOut):
    print(maxutil, file=fileUtilOut)
    if fileEdgeOut != None:
        netutils = []
        for item in netutilList:
            netutils += item
        print(netutils, file=fileEdgeOut)

def init_output_file():
    dirLog = PATHPRE + "outputs/log/" + REAL_STAMP
    dirCkpoint = PATHPRE + "outputs/ckpoint/" + REAL_STAMP
    if not os.path.exists(dirLog):
        os.mkdir(dirLog)
    fileUtilOut = open(dirLog + '/util.log', 'w', 1)
    if IS_TRAIN and AGENT_TYPE != "ECMP":
        if not os.path.exists(dirCkpoint):
            os.mkdir(dirCkpoint)
    if not IS_TRAIN and AGENT_TYPE != "ECMP" and MAX_EPISODES == 1:
        fileEdgeOut = open(dirLog + '/edge.log', 'w', 1)
    else:
        fileEdgeOut = None
    return dirLog, dirCkpoint, fileUtilOut, fileEdgeOut

def log_time_file(timeRecord, dirLog):
    print('\n' + REAL_STAMP)
    logfile = open(dirLog + "/runtime.log", 'w')
    runtimeType = ["inital time", "training time", "running time"]
    timeRecordPair = [[timeRecord[0], timeRecord[1]], 
                    [timeRecord[1], timeRecord[2]], 
                    [timeRecord[0], timeRecord[3]]]
    for t in range(len(timeRecordPair)):
        start_time = timeRecordPair[t][0]
        end_time = timeRecordPair[t][1]
        interval = int((end_time-start_time)*1000)
        timeMs = interval%1000
        timeS = int(interval/1000)%60
        timeMin = int((interval/1000-timeS)/60)%60
        timeH = int(interval/1000)/3600
        runtime = "%dh-%dmin-%ds-%dms" % (timeH, timeMin, timeS, timeMs)
        print("%s: %s" % (runtimeType[t], runtime))
        logfile.write("%s: %s\n" % (runtimeType[t], runtime))
    logfile.close()

if __name__ == "__main__":
    '''initial part'''
    print("\n----Information list----")
    print("agent_type: %s" % (AGENT_TYPE))
    print("stamp_type: %s" % (REAL_STAMP))
    timeRecord = []
    timeRecord.append(time.time())
    dirLog, dirCkpoint, fileUtilOut, fileEdgeOut = init_output_file()
    
    config = tf.ConfigProto(intra_op_parallelism_threads = 10)
    globalSess = tf.Session(config = config)
    env, agents, initActions, regionNodeNeibor, actionBorderline = init_multi_agent(globalSess)
    env.show_info()
    timeRecord.append(time.time())

    update_count = 0
    routing = initActions
    for _ in range(MAX_EPISODES * MAX_EP_STEPS):
        maxutil, maxutilList, netutilList = env.update(routing)
        log_to_file(maxutil, fileUtilOut, netutilList, fileEdgeOut)
        routing = update_step(maxutil, maxutilList, netutilList, agents, regionNodeNeibor, actionBorderline)
        if update_count % 1000 == 0:
            print("update_count:", update_count, "  max_util:", maxutilList)
        update_count += 1
    
    timeRecord.append(time.time())
    # store global variables
    if IS_TRAIN and AGENT_TYPE != "ECMP":
        print("saving checkpoint...")
        mSaver = tf.train.Saver(tf.global_variables())        
        mSaver.save(globalSess, dirCkpoint + "/ckpt")
        print("save checkpoint over")
    
    timeRecord.append(time.time())
    fileUtilOut.close()
    if fileEdgeOut != None:
        fileEdgeOut.close()
    log_time_file(timeRecord, dirLog)
