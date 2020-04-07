from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from network_env import SimEnv
from demo.network_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
import time

pathPre = "/home/server/gengnan/NATE_project/"
timestamp = "0305"
target = "failure"
# if infer or failure, remenber to modify the flag in network_env.py
# if n_parallel=1, set the flag to -1; if n_parallel>1, set the flag to 0
# if failure, remember to modify the iter_num to 100 in batch_polopt.py
topo = "briten12r16grid" # google
synthesis_type = "gravNR250" # gravNR50c bimoSame28

# win_size, max_itr_num, batch_size, n_parallel, hidden_layers = (1, 2, 1, 1, (32, 32))
# win_size, max_itr_num, batch_size, n_parallel, hidden_layers = (10, 5*1000, 1, 10, (64, 32, 32))
if topo == "briten12r16grid":
    win_size, max_itr_num, batch_size, n_parallel, hidden_layers = (1, 5*1000, 1, 1, (64, 32, 32))

stamp_tail = "win%d_itr%dk_b%d_pall%d" % (win_size, max_itr_num//1000, batch_size, n_parallel)

if target == "infer" or target == "failure":
    n_parallel = 1

log_name = "_".join([timestamp, topo, target, "drlte", synthesis_type, stamp_tail])
resume_log_name = "_".join([timestamp, topo, "train", "drlte", synthesis_type, stamp_tail])
dnn_para_file = pathPre + "outputs/ckpoint/" + resume_log_name + "/params.pkl"

print('\n' + log_name)
# exit()

def run_task(*_):
    env = normalize(SimEnv(pathPre, topo, synthesis_type, win_size, max_itr_num, log_name))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hidden_layers
    )
    baseline = GaussianMLPBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        n_itr=max_itr_num,
        batch_size=batch_size,
        max_path_length=1,
        discount=0.99,
    )
    algo.train()
start_time = time.time()
if target == "train":
    run_experiment_lite(
        run_task,
        snapshot_mode="last",
        log_dir=pathPre + "outputs/ckpoint/" + log_name,
        exp_name=log_name,
        seed=1,
        n_parallel=n_parallel,
    )
elif target == "infer" or target == "failure":
    run_experiment_lite(
        run_task,
        snapshot_mode="none",
        log_dir=pathPre + "outputs/ckpoint/" + log_name,
        exp_name=log_name,
        resume_from=dnn_para_file,
        seed=1,
        n_parallel=n_parallel,
    )
else:
    pass
end_time = time.time()
print('\n' + log_name)

interval = int((end_time-start_time)*1000)
timeMs = interval%1000
timeS = int(interval/1000)%60
timeMin = int((interval/1000-timeS)/60)%60
timeH = int(interval/1000)/3600
print("Running time: %dh-%dmin-%ds-%dms\n" % (timeH, timeMin, timeS, timeMs))
logfile = open(pathPre + "outputs/log/" + log_name + "/runtime.log", 'w')
logfile.write("Running time: %dh-%dmin-%ds-%dms\n" % (timeH, timeMin, timeS, timeMs))
logfile.close()
