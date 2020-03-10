#coding=utf-8
from __future__ import division
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "/home/server/gengnan/NATE_project/outputs/"

def read_file(filename, start_index = 0, end_index = 999999999):
    infile = open(filename, "r")
    ret = []
    for i in infile.readlines()[start_index:end_index]:
        ret.append(float(i.strip()))
    infile.close()
    return ret

def converge(topo, start, synthesis_type = "", stampList = []):
    length = 40000
    filename = LOG_PATH + "objvals/" + topo + "_mcf_obj_vals%s.txt" % (synthesis_type)
    mcf_ret = read_file(filename, start, start+1)[0]
    # mcf_ret = 0.432

    res = []
    for stamp_type in stampList:
        filename = LOG_PATH + "log/" + stamp_type + "/util.log"
        date_ret = read_file(filename, 0, length)
        res.append([item/mcf_ret for item in date_ret])
        print(date_ret[-1]/mcf_ret)

    for index in range(len(stampList)):
        stamp_type = stampList[index]
        plt.plot([i for i in range(len(res[index]))], res[index], linewidth=1.5, label=stamp_type)
        plt.xlabel("Update Step")
        plt.ylabel("Performance Ratio")
        # plt.ylim(ymin=1.0)
        # plt.ylim(1.0, 3.0)
        plt.savefig(LOG_PATH + "figures/%s.png" % (stamp_type))
        print(LOG_PATH + "figures/%s.png" % (stamp_type))
        plt.clf()

def converge_train(topo, start, synthesis_type = "", stampList = []):
    length = 80000
    filename = LOG_PATH + "objvals/" + topo + "_mcf_obj_vals%s.txt" % (synthesis_type)
    mcf_ret = read_file(filename, start, start+length//10000)

    res = []
    for stamp_type in stampList:
        filename = LOG_PATH + "log/" + stamp_type + "/util.log"
        date_ret = read_file(filename, 0, length)
        res.append([date_ret[i]/mcf_ret[i//10000] for i in range(len(date_ret))])

    for index in range(len(stampList)):
        stamp_type = stampList[index]
        plt.plot([i for i in range(len(res[index]))], res[index], linewidth=1.5, label=stamp_type)
        plt.xlabel("Update Step")
        plt.ylabel("Performance Ratio")
        # plt.ylim(ymin=1.0)
        # plt.ylim(1.0, 3.0)
        plt.savefig(LOG_PATH + "figures/%s.png" % (stamp_type))
        print("%s.png" % (stamp_type))
        plt.clf()

def read_edge_file(filename, regionEdgeNum, lineLen = 5):
    filein = open(filename)
    lines = filein.readlines()
    filein.close()
    res = []
    for i in range(1, lineLen+1):
        line = lines[i].strip()[1:-1]
        # print(line)
        lineList = line.split(',')
        if res == []:
            res = np.array(list(map(float, lineList)))
        else:
            res += np.array(list(map(float, lineList)))
    res2 = list(res/lineLen)
    # length = len(res2)//5
    # tmp = []
    # for i in range(5):
    #     tmp.append(max(res2[i*length:i*length+length]))
    tmp = []
    startIndex = 0
    for i in range(len(regionEdgeNum)):
        tmp.append(max(res2[startIndex:startIndex+regionEdgeNum[i]]))
        startIndex += regionEdgeNum[i]
    return res/lineLen, tmp
def coninfer(topo, schemeList, figurename, synthesis_type = "", stampList = []):
    regionEdgeNumDict = {"google": [68, 35, 57], "briten15r5loopb": [48, 54, 51, 55, 70], "briten15r5line": [43, 53, 50, 54, 58], "1221c": [44, 18, 20, 56, 14]}
    utils = {}
    oldobj = []
    newobjmultiply = []
    newobjsum = []
    for i in range(len(schemeList)):
        stamp_type = stampList[i]
        filename = LOG_PATH + "log/" + stamp_type + "/edge.log"
        ret, ret_max = read_edge_file(filename, regionEdgeNumDict[topo])
        scheme = schemeList[i]
        ret.sort()
        ret = list(ret)
        ret.reverse()
        edgeNum = sum(regionEdgeNumDict[topo])
        # print(scheme, sum(ret_max), ret_max)
        print(scheme, sum(ret_max), max(ret_max), [sum(ret[:int(edgeNum*0.05*k)])/(int(edgeNum*0.05*k)) for k in range(1, 6)])
        tmp = 1
        for item in ret_max:
            tmp *= item
        newobjmultiply.append(tmp)
        newobjsum.append(sum(ret_max))
        oldobj.append(max(ret_max))
        utils[scheme] = [tmp, sum(ret_max), max(ret_max)]
    minnewobjmultiply = min(newobjmultiply)
    minnewobjsum = min(newobjsum)
    minoldobj = min(oldobj)

    filedat = []
    result = [item/minoldobj for item in oldobj]
    filedat.append(result)
    plt.plot([i for i in range(len(schemeList))], result, linewidth=0.5, label="oldobj")

    result = [item/minnewobjmultiply for item in newobjmultiply]
    filedat.append(result)
    plt.plot([i for i in range(len(schemeList))], result, linewidth=0.5, label="newobjm")

    result = [item/minnewobjsum for item in newobjsum]
    filedat.append(result)
    plt.plot([i for i in range(len(schemeList))], result, linewidth=0.5, label="newobjs")
    
    plt.xlabel("XXX")
    plt.ylabel("XXX")
    plt.legend(fontsize=6)
    plt.savefig(LOG_PATH + "figures/%s_edge_line.png" % (figurename))
    print(LOG_PATH + "figures/%s_edge_line.png" % (figurename))
    plt.clf()

    outfile = open(LOG_PATH + "dat/%s_rwd%s_line.dat" % (topo, synthesis_type), 'w')
    length = len(filedat[0])
    for i in range(length):
        outfile.write(str(i) + ' ')
        for j in range(len(filedat)):
            outfile.write(str(filedat[j][i]) + ' ')
        outfile.write('\n')
    outfile.close()

    outfile = open(LOG_PATH + "dat/%s_rwd%s_hist.dat" % (topo, synthesis_type), 'w')
    length = len(filedat[0])
    for i in range(len(filedat)):
        for j in range(length):
            outfile.write(str(filedat[i][j]) + ' ')
        outfile.write('\n')
    outfile.close()

def get_incre_effect(topo, schemeList, start, end, figurename, synthesis_type = "", stampList = [], epoch = 18000, runtime = []):
    utils = {}
    scheme = "MCF"
    filename = LOG_PATH + "objvals/" + topo + "_" + scheme.lower() + "_obj_vals%s.txt" % (synthesis_type)
    ret = read_file(filename, start, end)
    utils[scheme] = ret

    for i in range(len(schemeList)):
        scheme = schemeList[i]
        stamp_type = stampList[i]
        filename = LOG_PATH + "log/" + stamp_type + "/util.log"
        ret = read_file(filename)
        utils[scheme] = []
        for j in range(start, end):
            ret_tmp = ret[j*epoch+epoch-50:j*epoch+epoch]
            utils[scheme].append(sum(ret_tmp)/len(ret_tmp))
    
    filedat = []
    length = len(utils[schemeList[0]])
    for scheme in schemeList:
        result = [utils[scheme][i]/utils["MCF"][i] for i in range(length)]
        result.sort()
        filedat.append(result)
        print(np.mean(result))
        plt.plot(result, [(i+1)/length for i in range(length)], linewidth=1.5, label=scheme)
    plt.xlabel("Performance Ratio", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(LOG_PATH + "figures/%s_cdf.png" % (figurename))
    print(LOG_PATH + "figures/%s_cdf.png\n" % (figurename))
    plt.clf()

    runtime.reverse()
    filedat.reverse()

    outfile = open(LOG_PATH + "dat/%s_incre%s_hist.dat" % (topo, synthesis_type), 'w')
    for i in range(len(schemeList)):
        rowval = [i, np.mean(filedat[i]), np.mean(filedat[i]), np.mean(filedat[i])+np.var(filedat[i]), runtime[i]]
        outfile.write(' '.join(list(map(str, rowval))) + '\n')
    outfile.close()

def infer(topo, schemeList, start, end, figurename, synthesis_type = "", stampList = []):
    utils = {}
    for i in range(len(schemeList)):
        scheme = schemeList[i]
        if scheme == "SP":
            filename = LOG_PATH + "objvals/" + topo + "_sp_obj_vals%s.txt" % synthesis_type
            ret = read_file(filename, start, end)
            utils[scheme] = ret
            continue
        if scheme == "ECMP":
            filename = LOG_PATH + "objvals/" + topo + "_lb_obj_vals%s.txt" % synthesis_type
            ret = read_file(filename, start, end)
            utils[scheme] = ret
            continue
        if scheme == "HP":
            filename = LOG_PATH + "objvals/" + topo + "_hp_obj_vals%s.txt" % synthesis_type
            ret = read_file(filename, start, end)
            utils[scheme] = ret
            continue
        if scheme == "HPMCF":
            filename = LOG_PATH + "objvals/" + topo + "_hpmcf_obj_vals%s.txt" % synthesis_type
            ret = read_file(filename, start, end)
            utils[scheme] = ret
            continue
        if scheme == "DRLTE":
            stamp_type = stampList[-1]
            filename = LOG_PATH + "log/" + stamp_type + "/rwd.log"
            ret_tmp = read_file(filename, start, end)
            ## ret = ret_tmp[190:] + ret_tmp[:190]
            utils[scheme] = ret_tmp
            # print(scheme, len(ret), ret[0:5])
            continue
        stamp_type = stampList[i]
        filename = LOG_PATH + "log/" + stamp_type + "/maxutils.result"
        ret = read_file(filename, start, end)
        utils[scheme] = ret
        # print(scheme, len(ret), ret[0:5])
    
    scheme = "MCF"
    if topo != "briten12r16grid":
        filename = LOG_PATH + "objvals/" + topo + "_" + scheme.lower() + "_obj_vals%s.txt" % (synthesis_type)
    else:
        filename = LOG_PATH + "objvals/" + topo + "_p3_3_1_obj_vals%s.txt" % (synthesis_type)
    ret = read_file(filename, start, end)
    # ret = [min(utils["MMA"])]*len(utils["MMA"])
    utils[scheme] = ret
    # print(scheme, len(ret), ret[0:5])

    length = len(utils["MCF"])
    # for scheme in schemeList:
    #     if scheme == "DRLTE":
    #         result = [utils[scheme][i]*(-1) for i in range(length)]
    #         print(result)
    #     else:
    #         result = [utils[scheme][i]/utils["MCF"][i] for i in range(length)]
    #     plt.plot([i for i in range(length)], result, linewidth=1.5, label=scheme)
    # plt.xlabel("TM Index", fontsize=12)
    # plt.ylabel("Performance Ratio", fontsize=12)
    # plt.legend(fontsize=12)
    # plt.savefig(LOG_PATH + "figures/%s.png" % (figurename))
    # print("%s.png" % (figurename))
    # plt.clf()

    filedat = []
    for scheme in schemeList:
        if scheme == "DRLTE":
            result = [max([1.0, utils[scheme][i]*(-1)]) for i in range(length)]
            # print(result)
        else:
            result = [max([1.0, utils[scheme][i]/utils["MCF"][i]]) for i in range(length)]
        result.sort()
        filedat.append(result)
        plt.plot(result, [(i+1)/length for i in range(length)], linewidth=1.5, label=scheme)
    plt.xlabel("Performance Ratio", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(LOG_PATH + "figures/%s_cdf.png" % (figurename))
    print(LOG_PATH + "figures/%s_cdf.png\n" % (figurename))
    plt.clf()

    outfile = open(LOG_PATH + "dat/%s_eff%s_cdf.dat" % (topo, synthesis_type), 'w')
    for i in range(length):
        outfile.write(str((i+1)/length) + ' ')
        for j in range(len(schemeList)):
            outfile.write(str(filedat[j][i]) + ' ')
        outfile.write('\n')
    outfile.close()

def infer2(topo, schemeList, start, end, figurename, synthesis_type = "", stampList = [], runtime = []):
    utils = {}
    for i in range(len(schemeList)):
        scheme = schemeList[i]
        stamp_type = stampList[i]
        filename = LOG_PATH + "log/" + stamp_type + "/maxutils.result"
        ret = read_file(filename, start, end)
        utils[scheme] = ret
        # print(scheme, len(ret), ret[0:5])
    
    scheme = "MCF"
    filename = LOG_PATH + "objvals/" + topo + "_" + scheme.lower() + "_obj_vals%s.txt" % (synthesis_type)
    ret = read_file(filename, start, end)
    utils[scheme] = ret

    length = len(utils["MCF"])
    filedat = []
    for scheme in schemeList:
        result = [max([1.0, utils[scheme][i]/utils["MCF"][i]]) for i in range(length)]
        filedat.append(result)
        print(np.mean(result), np.var(result), np.std(result, ddof=1))
    plt.boxplot(filedat)
    plt.xlabel("Incre Ratio", fontsize=12)
    plt.ylabel("Performance Ratio", fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.savefig(LOG_PATH + "figures/%s_box.png" % (figurename))
    print(LOG_PATH + "figures/%s_box.png\n" % (figurename))
    plt.clf()

    runtime.reverse()
    filedat.reverse()

    outfile = open(LOG_PATH + "dat/%s_eff%s_hist.dat" % (topo, synthesis_type), 'w')
    for i in range(len(schemeList)):
        rowval = [i, np.mean(filedat[i]), np.mean(filedat[i]), np.mean(filedat[i])+np.var(filedat[i]), runtime[i]]
        outfile.write(' '.join(list(map(str, rowval))) + '\n')
    outfile.close()

def failure(topo, schemeList, start, end, synthesis_types, stampListMMA, stampListECMP, stampListDRLTE = []):
    length = end - start
    filedat = []
    for synid in range(len(synthesis_types)):
        synthesis_type = synthesis_types[synid]
        scheme = "MCF"
        # filename = LOG_PATH + "objvals/" + topo + "_" + scheme.lower() + "_obj_vals%s_failure.txt" % (synthesis_type)
        # mcf_ret = read_file(filename, start, end)
        if topo == "briten12r16grid":
            filename = LOG_PATH + "objvals/" + topo + "_p3_3_1_obj_vals%s.txt" % (synthesis_type)
        else:
            filename = LOG_PATH + "objvals/" + topo + "_" + scheme.lower() + "_obj_vals%s.txt" % (synthesis_type)
        mcf_ret_tmp = read_file(filename, 40, 50)
        mcf_ret = []
        for i in range(10):
            mcf_ret += [mcf_ret_tmp[i]]*10
    
        for i in range(len(schemeList)):
            scheme = schemeList[i]
            if scheme == "SP":
                filename = LOG_PATH + "objvals/" + topo + "_sp_obj_vals%s_failure.txt" % synthesis_type
                ret = read_file(filename, start, end)
                ret_normal = [ret[j]/mcf_ret[j] for j in range(length)]
                filedat.append(ret_normal)
                
            if scheme == "ECMP":
                stamp_type = stampListECMP[synid]
                filename = LOG_PATH + "log/" + stamp_type + "/maxutils.result"
                ret = read_file(filename, start, end)
                ret_normal = [ret[j]/mcf_ret[j] for j in range(length)]
                filedat.append(ret_normal)
                
            if scheme == "HP":
                filename = LOG_PATH + "objvals/" + topo + "_hp_obj_vals%s_failure.txt" % synthesis_type
                ret = read_file(filename, start, end)
                ret_normal = [ret[j]/mcf_ret[j] for j in range(length)]
                filedat.append(ret_normal)
            
            if scheme == "HPMCF":
                filename = LOG_PATH + "objvals/" + topo + "_hpmcf_obj_vals%s_failure.txt" % synthesis_type
                ret = read_file(filename, start, end)
                ret_normal = [ret[j]/mcf_ret[j] for j in range(length)]
                filedat.append(ret_normal)
                
            if scheme == "DRLTE":
                stamp_type = stampListDRLTE[synid]
                filename = LOG_PATH + "log/" + stamp_type + "/maxutil.log"
                ret = read_file(filename, start, end)
                ret_normal = [ret[j]/mcf_ret[j] for j in range(length)]
                filedat.append(ret_normal)
                
            if scheme == "MMA":
                stamp_type = stampListMMA[synid]
                filename = LOG_PATH + "log/" + stamp_type + "/maxutils.result"
                ret = read_file(filename, start, end)
                ret_normal = [ret[j]/mcf_ret[j] for j in range(length)]
                filedat.append(ret_normal)
                # print("median:", scheme, np.median(ret_normal))
        
    outfile = open(LOG_PATH + "dat/%s_failure_box.dat" % (topo), 'w')
    for i in range(length):
        outfile.write(str(i) + ' ')
        for j in range(len(schemeList)*len(synthesis_types)):
            outfile.write(str(filedat[j][i]) + ' ')
        outfile.write('\n')
    outfile.close()

    outfile = open(LOG_PATH + "dat/%s_failure_hist.dat" % (topo), 'w')
    for i in range(len(schemeList)):
        rowval = [i, np.mean(filedat[i]), np.mean(filedat[i]), np.mean(filedat[i])+np.var(filedat[i])]
        outfile.write(' '.join(list(map(str, rowval))) + '\n')
    outfile.close()

def scalability(topo, TMid, synthesis_type = "", stampList = []):
    length = 18000
    # filename = LOG_PATH + "objvals/" + topo + "_mcf_obj_vals%s.txt" % (synthesis_type)
    # mcf_ret = read_file(filename, TMid, TMid+1)[0]
    mcf_ret = 0.3133
    # mcf_ret = 0.413

    res = []
    vals = []
    for stamp_type in stampList:
        filename = LOG_PATH + "log/" + stamp_type + "/util.log"
        date_ret = read_file(filename, 0, length)
        res.append([item/mcf_ret for item in date_ret])
        # print(date_ret[-1]/mcf_ret)
        converge_point = sum([item/mcf_ret for item in date_ret[-100:]])/100.0
        vals.append(converge_point)

    print(vals)
    outfile = open(LOG_PATH + "dat/%s_scale%s_hist.dat" % (topo, synthesis_type), 'w')
    for i in range(len(stampList)):
        outfile.write(str(i) + ' ' + str(vals[i]) + '\n')
    outfile.close()

def alternate(topo, alterNum, start, figurename, synthesis_type, stampList, epoch = 6000):
    filename = LOG_PATH + "objvals/" + topo + "_mcf_obj_vals%s.txt" % (synthesis_type)
    mcf_ret = read_file(filename, start, start+1)[0]

    filename = LOG_PATH + "log/" + stampList[0] + "/util.log"
    date_ret = read_file(filename, 0, epoch*alterNum)
    
    res = []
    vals = []
    for epi in [i for i in range(alterNum+1) if i%2==1]:
        tmp = [item/mcf_ret for item in date_ret[epi*epoch - 500:epi*epoch]]
        res.append(tmp)
        vals.append([np.mean(tmp), max(tmp), min(tmp), np.median(tmp)])
    
    # print(vals)
    for item in vals:
        print(item)

# scp -r nan@128.46.202.143:/home/nan/gengnan/NATE_project/outputs/log/XX ./


################
# effectiveness
################

# google gravNR250
# stamps = ["0302_google_infer_MMA_p331_gravNR250_comm5_incre0.3_alter3_LB_epi40", "0303_google_infer_drlte_gravNR250_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("google", ["MMA", "DRLTE", "ECMP", "HPMCF"], 40, 200, figurename, synthesis_type = "_gravNR250", stampList = stamps)
# exit()

# 1221c gravNR250
# stamps = ["0302_1221c_infer_MMA_p331_gravNR250_comm5_incre0.3_alter3_LB_epi40", "0303_1221c_infer_drlte_gravNR250_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("1221c", ["MMA", "DRLTE", "ECMP", "HPMCF"], 40, 200, figurename, synthesis_type = "_gravNR250", stampList = stamps)
# exit()

# # briten12r16grid gravNR250
# stamps = ["0303_briten12r16grid_infer_MMA_p331_gravNR250_comm5_incre0.3_alter1_LB_16blocks_epi20", "0305_briten12r16grid_infer_drlte_gravNR250_win1_itr5k_b1_pall1"]
# figurename = stamps[0]
# infer("briten12r16grid", ["MMA", "DRLTE", "ECMP", "HPMCF"], 40, 200, figurename, synthesis_type = "_gravNR250", stampList = stamps)
# exit()

################
# scalability
################

# stamps = ["0227_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_1blocks", "0227_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_2blocks", "0306_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_4blocks", "0225_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_8blocks", "0225_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_16blocks"]
# scalability("briten12r16grid", 0, synthesis_type = "_gravNR50c", stampList = stamps)
# exit()


################
# failure
################
# google failure
# stamps1 = ["0302_google_failure_MMA_p331_gravNR250_comm5_incre0.3_alter3_LB_epi40"]
# stamps2 = ["0305_google_failure_ECMP_p331_gravNR250"]
# stamps3 = ["0303_google_failure_drlte_gravNR250_win10_itr5k_b1_pall10"]
# failure("google", ["MMA", "DRLTE", "HPMCF", "ECMP"], 0, 100, ["_gravNR250"], stamps1, stamps2, stamps3)
# exit()

# 1221c failure
# stamps1 = ["0302_1221c_failure_MMA_p331_gravNR250_comm5_incre0.3_alter3_LB_epi40"]
# stamps2 = ["0305_1221c_failure_ECMP_p331_gravNR250"]
# stamps3 = ["0303_1221c_failure_drlte_gravNR250_win10_itr5k_b1_pall10"]
# failure("1221c", ["MMA", "DRLTE", "HPMCF", "ECMP"], 0, 100, ["_gravNR250"], stamps1, stamps2, stamps3)
# # exit()

# briten12r16grid failure
# stamps1 = ["0303_briten12r16grid_failure_MMA_p331_gravNR250_comm5_incre0.3_alter1_LB_16blocks_epi20"]
# stamps2 = ["0305_briten12r16grid_failure_ECMP_p331_gravNR250"]
# stamps3 = ["0305_briten12r16grid_failure_drlte_gravNR250_win1_itr5k_b1_pall1"]
# failure("briten12r16grid", ["MMA", "DRLTE", "HPMCF", "ECMP"], 0, 100, ["_gravNR250"], stamps1, stamps2, stamps3)
# exit()

################
# incre
################

# stamps = ["0305_1221c_train_MMA_p331_gravNR250_comm5_incre0.1_alter3_LB_epi40", "0302_1221c_train_MMA_p331_gravNR250_comm5_incre0.3_alter3_LB_epi40",    "0305_1221c_train_MMA_p331_gravNR250_comm5_incre0.5_alter3_LB_epi40",    "0305_1221c_train_MMA_p331_gravNR250_comm5_incre0.7_alter3_LB_epi40",    "0305_1221c_train_MMA_p331_gravNR250_comm5_incre0.9_alter3_LB_epi40"]
# figurename = stamps[0]
# # runtime = [30, 31.6, 33.33, 35.7, 34.77]
# # runtime = [30, 31.6, 33.33, 35.7, 35.77]
# # runtime = [item*60/40 for item in runtime]
# runtime = [45.33-1, 48.67-1, 50.5-1, 53.5-1, 58.42-1]
# print(runtime)
# get_incre_effect("1221c", ["MMA%d" % i for i in range(len(stamps))], 0, 40, figurename, synthesis_type = "_gravNR250", stampList = stamps, runtime = runtime)
# exit()

################
# reward
################
# stamps = ["0306_1221c_coninfer_MMA_p331_gravNR250_comm2_incre0.1_alter3_LB",
# "0306_1221c_coninfer_MMA_p331_gravNR250_comm3_incre0.1_alter3_LB",
# # "0306_1221c_coninfer_MMA_p331_gravNR250_comm4_incre0.1_alter3_LB",
# '0306_1221c_coninfer_MMA_p331_gravNR250_comm5_incre0.1_alter3_LB', 
# "0306_1221c_coninfer_MMA_p331_gravNR250_comm6_incre0.1_alter3_LB"]
stamps = ["0308_1221c_newconinfer_MMA_p331_gravNR250_comm2_incre0.1_alter1_LB_epo1500_TM1", "0308_1221c_newconinfer_MMA_p331_gravNR250_comm3_incre0.1_alter1_LB_epo1500_TM1", 
"0308_1221c_newconinfer_MMA_p331_gravNR250_comm5_incre0.1_alter1_LB_epo1500_TM1"]
# stamps = ["0220_briten15r5line_coninfer_MMA_p331_bimoSame28_comm3_incre0.7_alter3_LB", "0220_briten15r5line_coninfer_MMA_p331_bimoSame28_comm2_incre0.7_alter3_LB", "0220_briten15r5line_coninfer_MMA_p331_bimoSame28_comm4_incre0.7_alter3_LB", "0220_briten15r5line_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB"]
filename = stamps[0]
coninfer("1221c", ["comm2", "comm3", "comm5"], filename, "_gravNR250", stamps)

exit()

################
# alternate
################
# stamps = ["0306_1221c_converge_MMA_p331_gravNR250_comm5_incre0.1_alter9_LB_test2"]
# figurename = stamps[0]
# alternate("1221c", 9, 0, figurename, synthesis_type = "_gravNR250", stampList = stamps)


exit()
################################################################################################################













################
# scalability
################
# stamps = ["0227_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_1blocks", "0227_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_2blocks", "0226_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_4blocks", "0225_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_8blocks", "0225_briten12r16grid_converge_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_16blocks"]

# scalability("briten12r16grid", 0, synthesis_type = "_gravNR50c", stampList = stamps)

# stamps = ["0227_briten12r16grid_converge_MMA_p331_bimoSame28b_comm5_incre0.7_alter3_LB_1blocks", "0228_briten12r16grid_converge_MMA_p331_bimoSame28b_comm5_incre0.7_alter3_LB_2blocks", '0226_briten12r16grid_converge_MMA_p331_bimoSame28b_comm5_incre0.7_alter3_LB_4blocks', "0226_briten12r16grid_converge_MMA_p331_bimoSame28b_comm5_incre0.7_alter3_LB_8blocks", "0226_briten12r16grid_converge_MMA_p331_bimoSame28b_comm5_incre0.7_alter3_LB_16blocks"]

# scalability("briten12r16grid", 0, synthesis_type = "_bimoSame28b", stampList = stamps)

# exit()

# google failure
# stamps1 = ["0208_google_failure_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_epi40", "0203_google_failure_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_epi40"]
# stamps2 = ["0214_google_failure_ECMP_p331_gravNR50c", "0214_google_failure_ECMP_p331_bimoSame28"]
# stamps3 = ["0209_google_failure_drlte_gravNR50c_win10_itr5k_b1_pall10", "0212_google_failure_drlte_bimoSame28_win10_itr5k_b1_pall10"]
# failure("google", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 0, 100, ["_gravNR50c", "_bimoSame28"], stamps1, stamps2, stamps3)
# exit()

# # briten15r5loopb failure
# stamps1 = ["0210_briten15r5loopb_failure_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_epi40", "0205_briten15r5loopb_failure_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_epi40"]
# stamps2 = ["0214_briten15r5loopb_failure_ECMP_p331_gravNR50c", "0214_briten15r5loopb_failure_ECMP_p331_bimoSame28"]
# stamps3 = ["0211_briten15r5loopb_failure_drlte_gravNR50c_win10_itr5k_b1_pall10", "0210_briten15r5loopb_failure_drlte_bimoSame28_win10_itr5k_b1_pall10"]
# failure("briten15r5loopb", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 0, 100, ["_gravNR50c", "_bimoSame28"], stamps1, stamps2, stamps3)
# exit()

# briten15r5line failure
# stamps1 = ["0208_briten15r5line_failure_MMA_p331_gravNR50c_comm5_incre0.3_alter3_LB_epi40", "0205_briten15r5line_failure_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_epi40"]
# stamps2 = ["0214_briten15r5line_failure_ECMP_p331_gravNR50c", "0214_briten15r5line_failure_ECMP_p331_bimoSame28"]
# stamps3 = ["0213_briten15r5line_failure_drlte_gravNR50c_win10_itr5k_b1_pall10", "0212_briten15r5line_failure_drlte_bimoSame28_win10_itr5k_b1_pall10"]
# failure("briten15r5line", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 0, 100, ["_gravNR50c", "_bimoSame28"], stamps1, stamps2, stamps3)
# exit()

# google gravNR50c
# stamps = ["0208_google_infer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_epi40", "0209_google_infer_drlte_gravNR50c_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("google", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_gravNR50c", stampList = stamps)
# exit()

# google gravNR100
# stamps = ["0229_google_infer_MMA_p331_gravNR100_comm5_incre0.3_alter3_LB_epi40"]
# figurename = stamps[0]
# infer("google", ["MMA", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_gravNR100", stampList = stamps)
# exit()

# google gravNR150
# stamps = ["0302_google_infer_MMA_p331_gravNR150_comm5_incre0.3_alter3_LB_epi40"]
# figurename = stamps[0]
# infer("google", ["MMA", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_gravNR150", stampList = stamps)
# exit()

# google gravNR250
# stamps = ["0302_google_infer_MMA_p331_gravNR250_comm5_incre0.3_alter3_LB_epi40", "0303_google_infer_drlte_gravNR250_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("google", ["MMA", "DRLTE", "ECMP", "HPMCF"], 40, 200, figurename, synthesis_type = "_gravNR250", stampList = stamps)
# exit()


# 1221c gravNR150
# stamps = ["0302_1221c_infer_MMA_p331_gravNR150_comm5_incre0.3_alter3_LB_epi40"]
# figurename = stamps[0]
# infer("1221c", ["MMA", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_gravNR150", stampList = stamps)
# exit()

# 1221c gravNR250
# stamps = ["0302_1221c_infer_MMA_p331_gravNR250_comm5_incre0.3_alter3_LB_epi40", "0303_1221c_infer_drlte_gravNR250_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("1221c", ["MMA", "DRLTE", "ECMP", "HPMCF"], 40, 200, figurename, synthesis_type = "_gravNR250", stampList = stamps)
# exit()

# google bimoSame28b
# stamps = ["0226_google_infer_MMA_p331_bimoSame28b_comm5_incre0.7_alter3_LB_epi40", "0227_google_infer_drlte_bimoSame28b_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("google", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_bimoSame28b", stampList = stamps)
# exit()

# # 1221c bimoSame28b
# stamps = ["0302_1221c_infer_MMA_p331_bimoSame28b_comm5_incre0.7_alter3_LB_epi40", "0302_1221c_infer_drlte_bimoSame28b_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("1221c", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_bimoSame28b", stampList = stamps)
# exit()

# google bimoSame28
# stamps = ["0203_google_infer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_epi40", "0212_google_infer_drlte_bimoSame28_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("google", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_bimoSame28", stampList = stamps)
# exit()

# briten12r16grid bimoSame28b
# stamps = ["0302_briten12r16grid_infer_MMA_p331_bimoSame28b_comm5_incre0.7_alter1_LB_16blocks_epi20"]
# figurename = stamps[0]
# infer("briten12r16grid", ["MMA", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_bimoSame28b", stampList = stamps)
# exit()


# briten15r5loopb gravNR50c
# stamps = ["0210_briten15r5loopb_infer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_epi40", "0211_briten15r5loopb_infer_drlte_gravNR50c_win10_itr5k_b1_pall10"] # "0208_briten15r5loopb_infer_MMA_p331_gravNR50c_comm5_incre0.3_alter3_LB_epi40", 
# figurename = stamps[0]
# infer("briten15r5loopb", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_gravNR50c", stampList = stamps)
# exit()

# briten15r5loopb bimoSame28
# stamps = ["0205_briten15r5loopb_infer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_epi40", "0210_briten15r5loopb_infer_drlte_bimoSame28_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("briten15r5loopb", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_bimoSame28", stampList = stamps)
# exit()

# briten15r5line gravNR50c
#"0208_briten15r5line_infer_MMA_p331_gravNR50c_comm5_incre0.3_alter3_LB_epi40", 
# stamps = ["0216_briten15r5line_infer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_epi40", "0213_briten15r5line_infer_drlte_gravNR50c_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("briten15r5line", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_gravNR50c", stampList = stamps)
# exit()

# briten15r5line bimoSame28
# stamps = ["0205_briten15r5line_infer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_epi40", "0212_briten15r5line_infer_drlte_bimoSame28_win10_itr5k_b1_pall10"]
# figurename = stamps[0]
# infer("briten15r5line", ["MMA", "DRLTE", "SP", "ECMP", "HP"], 40, 200, figurename, synthesis_type = "_bimoSame28", stampList = stamps)
# exit()

# stamps = ["0208_google_train_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_epi40", "0207_google_train_MMA_p331_gravNR50c_comm5_incre0.3_alter3_LB_epi40",  "0207_google_train_MMA_p331_gravNR50c_comm5_incre0.5_alter3_LB_epi40",  "0207_google_train_MMA_p331_gravNR50c_comm5_incre0.7_alter3_LB_epi40",  "0211_google_train_MMA_p331_gravNR50c_comm5_incre0.9_alter3_LB_epi40"]
# figurename = stamps[0] + "_increParaExpl"
# get_incre_effect("google", ["MMA%d" % i for i in range(len(stamps))], 0, 40, figurename, synthesis_type = "_gravNR50c", stampList = stamps)

# stamps = ["0203_google_train_MMA_p331_bimoSame28_comm5_incre0.1_alter3_LB_epi40", "0203_google_train_MMA_p331_bimoSame28_comm5_incre0.3_alter3_LB_epi40", "0203_google_train_MMA_p331_bimoSame28_comm5_incre0.5_alter3_LB_epi40", "0203_google_train_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_epi40", "0203_google_train_MMA_p331_bimoSame28_comm5_incre0.9_alter3_LB_epi40"]
# figurename = stamps[0] + "_increParaExpl"
# get_incre_effect("google", ["MMA%d" % i for i in range(len(stamps))], 0, 40, figurename, synthesis_type = "_bimoSame28", stampList = stamps)
# exit()


# stamps = ["0208_google_infer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_epi40", "0207_google_infer_MMA_p331_gravNR50c_comm5_incre0.3_alter3_LB_epi40",  "0207_google_infer_MMA_p331_gravNR50c_comm5_incre0.5_alter3_LB_epi40",  "0207_google_infer_MMA_p331_gravNR50c_comm5_incre0.7_alter3_LB_epi40",  "0216_google_infer_MMA_p331_gravNR50c_comm5_incre0.9_alter3_LB_epi40"]
# figurename = stamps[0]
# # runtime = [18.5, 25.08, 27.6, 29.6, 32]
# runtime = [20.35, 22.6, 25.1, 29.0, 39.1]
# for i in range(len(runtime)):
#     runtime[i] = (runtime[i] - 1)*40/60
# print(runtime)
# infer2("google", ["MMA%d" % i for i in range(len(stamps))], 40, 200, figurename, synthesis_type = "_gravNR50c", stampList = stamps, runtime = runtime)

# # exit()

# stamps = ["0203_google_infer_MMA_p331_bimoSame28_comm5_incre0.1_alter3_LB_epi40", "0203_google_infer_MMA_p331_bimoSame28_comm5_incre0.3_alter3_LB_epi40", "0203_google_infer_MMA_p331_bimoSame28_comm5_incre0.5_alter3_LB_epi40", "0203_google_infer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_epi40", "0203_google_infer_MMA_p331_bimoSame28_comm5_incre0.9_alter3_LB_epi40"]
# # runtime = [26.6, 29.6, 32.17, 34.2, 36]
# runtime = [25, 28, 31.5, 41.5, 46]
# for i in range(len(runtime)):
#     runtime[i] = (runtime[i] - 1)*40/60
# print(runtime)
# figurename = stamps[0]
# infer2("google", ["MMA%d" % i for i in range(len(stamps))], 40, 200, figurename, synthesis_type = "_bimoSame28", stampList = stamps, runtime = runtime)

# exit()

################
# reward
################

# stamps = ["0220_briten15r5line_coninfer_MMA_p331_bimoSame28_comm3_incre0.7_alter3_LB", "0220_briten15r5line_coninfer_MMA_p331_bimoSame28_comm2_incre0.7_alter3_LB", "0220_briten15r5line_coninfer_MMA_p331_bimoSame28_comm4_incre0.7_alter3_LB", "0220_briten15r5line_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB"]
# filename = "0220_briten15r5line_coninfer_MMA_p331_bimoSame28_incre0.7_alter3_LB_diffrwd_diffobj"
# coninfer("briten15r5line", ["comm3", "comm2", "comm4", "comm5"], filename, "_bimoSame28", stamps)

# exit()

# stamps = ["0220_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm2_incre0.7_alter3_LB", "0220_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm3_incre0.7_alter3_LB", "0220_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB"]
# filename = "0220_briten15r5loopb_coninfer_MMA_p331_bimoSame28_incre0.7_alter3_LB_diffrwd_diffobj"
# coninfer("briten15r5loopb", ["comm2", "comm3", "comm5"], filename, "_bimoSame28", stamps)

# # exit()

# stamps = ["0220_google_coninfer_MMA_p331_bimoSame28_comm2_incre0.7_alter3_LB", "0220_google_coninfer_MMA_p331_bimoSame28_comm3_incre0.7_alter3_LB","0220_google_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB"]
# filename = "0220_google_coninfer_MMA_p331_bimoSame28_incre0.7_alter3_LB_diffrwd_diffobj"
# coninfer("google", ["comm2", "comm3", "comm5"], filename, "_bimoSame28", stamps)

# stamps = ["0220_google_coninfer_MMA_p331_gravNR50c_comm2_incre0.1_alter3_LB", "0220_google_coninfer_MMA_p331_gravNR50c_comm3_incre0.1_alter3_LB", "0220_google_coninfer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB"]
# filename = "0220_google_coninfer_MMA_p331_gravNR50c_incre0.1_alter3_LB_diffrwd_diffobj"
# coninfer("google", ["comm2", "comm3", "comm5"], filename, "_gravNR50c", stamps)

# exit()

# stamps = ["0220_briten15r5line_coninfer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_rwd37", "0220_briten15r5line_coninfer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_rwd55", "0220_briten15r5line_coninfer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_rwd73", "0220_briten15r5line_coninfer_MMA_p331_gravNR50c_comm5_incre0.1_alter3_LB_rwd10"]
# filename = "0220_briten15r5line_coninfer_MMA_p331_gravNR50c_incre0.1_alter3_LB_diffrwd_diffobj"
# coninfer("briten15r5line", ["37", "55", "73", "10"], filename, "_bimoSame28", stamps)

# exit()

# stamps = ["0219_briten15r5line_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd01", "0219_briten15r5line_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd37", "0219_briten15r5line_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd55", "0219_briten15r5line_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd73", "0219_briten15r5line_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd10"]
# filename = "0219_briten15r5line_coninfer_MMA_p331_bimoSame28_incre0.7_alter3_LB_diffrwd_diffobj"
# coninfer("briten15r5line", ["01", "37", "55", "73", "10"], filename, "_bimoSame28", stamps)

# exit()


# stamps = ["0127_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd01", "0127_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd37", "0219_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd55", "0219_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd73", "0219_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd10"]
# # , "0127_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm4_incre0.7_alter3_LB"]
# filename = "0127_briten15r5loopb_coninfer_MMA_p331_bimoSame28_incre0.7_alter3_LB_diffrwd_diffobj_0219"
# coninfer("briten15r5loopb", ["01", "37", "55", "73", "10"], filename, "_bimoSame28", stamps)

# stamps = ["0219b_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd01", "0219b_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd37", "0219b_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd55", "0219b_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd73", "0219b_briten15r5loopb_coninfer_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd10"]
# filename = "0219b_briten15r5loopb_coninfer_MMA_p331_bimoSame28_incre0.7_alter3_LB_diffrwd_diffobj"
# coninfer("briten15r5loopb", ["01", "37", "55", "73", "10"], filename, "_bimoSame28", stamps)

# exit()

# stamps = ["0127_briten15r5loopb_converge_MMA_p331_bimoSame28_comm4_incre0.7_alter3_LB", "0127_briten15r5loopb_converge_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd01", "0127_briten15r5loopb_converge_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd37", "0127_briten15r5loopb_converge_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd55", "0127_briten15r5loopb_converge_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd73", "0127_briten15r5loopb_converge_MMA_p331_bimoSame28_comm5_incre0.7_alter3_LB_rwd10"]
# converge("briten15r5loopb", 0, synthesis_type = "_bimoSame28", stampList = stamps)
# exit()

# stamps = ["0127_google_converge_MMA_p331_gravNR50_comm4_incre0.3_alter3_LB", "0127_google_converge_MMA_p331_gravNR50_comm5_incre0.3_alter3_LB_rwd01", "0127_google_converge_MMA_p331_gravNR50_comm5_incre0.3_alter3_LB_rwd37", "0127_google_converge_MMA_p331_gravNR50_comm5_incre0.3_alter3_LB_rwd55", "0127_google_converge_MMA_p331_gravNR50_comm5_incre0.3_alter3_LB_rwd73", "0127_google_converge_MMA_p331_gravNR50_comm5_incre0.3_alter3_LB_rwd10"]
# converge("google", 0, synthesis_type = "_gravNR50", stampList = stamps)
# exit()