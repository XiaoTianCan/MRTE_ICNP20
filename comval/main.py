#! /usr/bin/env python3
import sys
print(sys.executable)
print(sys.version, sys.platform, sys.executable)
from scheme import *
import time
import random

# Main code
if __name__ == '__main__':
    print("#####Simulations begin!#####")
    m = Model('net') # clear Gurobi output
    del m

    start_time = time.time()
    
    infilePrefix = '../../inputs/'
    outfilePrefix = '../../outputs/'

    totalTMNum = 400
    for topo in ["briten12r16grid"]: #["googlec", "briten15r5loopb", "briten15r5line"]: # briten12r16grid
        topoName = topo
        synthesis_type = "_gravNR250" #_gravNR250 _bimoSame28b
        print(topoName, synthesis_type)

        ''' For failure scenario'''
        ### get_down_links(infilePrefix, topoName, pathType = "p3_3_1", synthesisType = synthesis_type)
        # get_mcf_obj_vals_failure(infilePrefix, outfilePrefix, topoName, totalTMNum = 10, pathType = "p3_3_1", synthesisType = synthesis_type)
        # get_sp_obj_vals_failure(infilePrefix, outfilePrefix, topoName, totalTMNum = 10, pathType = "p3_3_1", synthesisType = synthesis_type)
        # get_hp_obj_vals_failure(infilePrefix, outfilePrefix, topoName, totalTMNum = 10, pathType = "p3_3_1", synthesisType = synthesis_type)
        # get_hpmcf_obj_vals_failure(infilePrefix, outfilePrefix, topoName, totalTMNum = 10, pathType = "p3_3_1", synthesisType = synthesis_type)

        ''' For normal scenario'''
        totalTMNum = 1
        # get_region_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType = "p3_3_1", synthesisType = synthesis_type)
        # continue
        # get_mcf_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType = "p3_3_1", synthesisType = synthesis_type)
        # get_sp_pathset_region(infilePrefix, topoName, pathType = "p3_3_1", synthesisType = synthesis_type)
        get_sp_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType = "sp", synthesisType = synthesis_type)
        # get_lb_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType = "p3_3_1", synthesisType = synthesis_type)
        # get_hp_pathset_region(infilePrefix, topoName, pathType = "p3_3_1", synthesisType = synthesis_type)
        # get_hp_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType = "hp", synthesisType = synthesis_type)

        # get_hpmcf_obj_vals(infilePrefix, outfilePrefix, topoName, totalTMNum, pathType = "hp", synthesisType = synthesis_type)

        # continue

    end_time = time.time()
    interval = int((end_time-start_time)*1000)
    timeMs = interval%1000
    timeS = int(interval/1000)%60
    timeMin = int((interval/1000-timeS)/60)%60
    timeH = int(interval/1000)/3600
    print("Running time: %dh-%dmin-%ds-%dms\n" % (timeH, timeMin, timeS, timeMs))

'''
briten15r5loopb
mcf 
0.102531
0.099488
0.095508
0.095159
0.104633
0.098355
0.096717
0.096297
0.107164
0.095328
0.102649
0.116255
0.093004
0.098164
0.10713
0.099839
0.116277
0.113705
0.097331
0.100438

incre 1.0
0 0.11416438000000005 0.0
1 0.12134930000000009 0.0
2 0.10616638000000005 0.0
3 0.11529676000000004 0.0
4 0.12627800000000003 0.0
5 0.10840142000000001 0.0
6 0.11078912000000005 0.0
7 0.11691259999999999 0.0
8 0.11489138000000004 0.0
9 0.11106456000000003 0.0
10 0.12005748000000008 0.0
11 0.12386866000000002 0.0
12 0.11338240000000001 0.0
13 0.11213876000000007 0.0
14 0.11606500000000002 0.0
15 0.10374544000000005 0.0
16 0.12611190000000008 0.0
17 0.11540424000000003 0.0
18 0.11728588000000004 0.0
19 0.11190064000000002 0.0

incre 0.7
0 0.11444418666666666 0.0
1 0.12189364000000005 0.0
2 0.10643214000000002 0.0
3 0.11586976000000002 0.0
4 0.12726970000000004 0.0
5 0.10988991333333338 0.0
6 0.1107971866666667 0.0
7 0.12000088841368271 0.0
8 0.11520620000000001 0.0
9 0.11207567333333336 0.0
10 0.12115192000000004 0.0
11 0.12612346000000008 0.0
12 0.11810760666666667 0.0
13 0.11307376666666687 0.0
14 0.11606500000000003 0.0
15 0.10468772000000004 0.0
16 0.1269727933333334 0.0
17 0.11627302666666671 0.0
18 0.11815846000000003 0.0
19 0.11261724666666674 0.0
'''