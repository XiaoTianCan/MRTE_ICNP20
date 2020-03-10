from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from network_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite

def run_task(*_):
    env = normalize(PointEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        n_itr=10,
        discount=0.99,
    )
    algo.train()

pathPre = "/home/server/gengnan/NATE_project/outputs/log/"
log_name = "0000_drlte_test2"
log_name1 = "0000_drlte_test"
dnn_para_file = pathPre + log_name1 + "/params.pkl"
run_experiment_lite(
    run_task,
    snapshot_mode="last",
    log_dir=pathPre + log_name,
    exp_name=log_name,
    resume_from=dnn_para_file,
    seed=1,
)
