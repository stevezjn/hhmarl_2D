"""
Main file for training the high-level commander policy.
"""

import os
import time
import shutil
import tqdm
import torch
import logging
import numpy as np
from gymnasium import spaces
from tensorboard import program
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from models.ac_models_hier import CommanderGru
from config import Config
from envs.env_hier import HighLevelEnv

N_OPP_HL = 2 #change for sensing
ACT_DIM = N_OPP_HL+1
OBS_DIM = 14+10*N_OPP_HL

def update_logs(args, log_dir):
    """
    Copy stored checkpoints from Ray log to experiment log directory.
    """
    """
    将Ray训练日志中的检查点和事件文件同步到实验日志目录
    
    Args:
        args (object): 包含配置参数的对象，需要具有log_path属性
        log_dir (str): Ray训练日志的源目录路径
    Returns:
        None: 该函数没有返回值
    """

    # 获取按修改时间排序的日志子目录列表（最新修改的排在最后）
    dirs = sorted(Path(log_dir).glob('*/'), key=os.path.getmtime)
    # 初始化检查点目录和事件文件路径
    check = ''
    event = ''
    # 遍历日志目录，识别检查点和TensorBoard事件文件
    for item in dirs:
        if "checkpoint" in item.name:
            check = str(item)
        if "events" in item.name:
            event = str(item)
    
    # 构建目标检查点目录路径
    result_dir = os.path.join(args.log_path, 'checkpoint')
    
    # 清空目标目录（兼容目录不存在的情况）
    try:
        shutil.rmtree(result_dir)
    except:
        pass

    # 复制最新检查点并保持目录结构
    shutil.copytree(check,result_dir,symlinks=False,dirs_exist_ok=False)
    # 同步TensorBoard事件文件
    shutil.copy(event,result_dir)

def evaluate(args, algo, env, epoch):
    """执行多智能体环境评估过程
    
    Args:
        args: 包含运行时参数的配置对象
            - render: bool 是否渲染环境
            - log_path: Path 日志存储路径
        algo: 强化学习算法实例
            需实现compute_single_action方法
        env: 多智能体环境实例
            需实现reset/step/plot方法
        epoch: int 当前训练轮次编号
    
    Returns:
        无显式返回值，通过env.plot输出评估结果图像
    """

    def cc_obs(obs):
        """构建多智能体观测结构
        Args:
            obs: ndarray 单个智能体的原始观测数据
        Returns:
            包含三个智能体观测和动作占位符的字典结构
        """
        return {
            "obs_1_own": obs,
            "obs_2": np.zeros(OBS_DIM, dtype=np.float32),
            "obs_3": np.zeros(OBS_DIM, dtype=np.float32),
            "act_1_own": np.zeros(1),
            "act_2": np.zeros(1),
            "act_3": np.zeros(1),
        }
    
    # 初始化环境及评估指标
    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    # 执行完整评估循环
    while not done:
        actions = {}
        # LSTM状态初始化
        states = [torch.zeros(200), torch.zeros(200)]

        # 为每个智能体生成动作
        for ag_id, ag_s in state.items():
            a = algo.compute_single_action(observation=cc_obs(ag_s), state=states, policy_id="commander_policy", explore=False)
            actions[ag_id] = a[0]
            # 更新LSTM隐藏状态
            states[0] = a[1][0]
            # 更新LSTM细胞状态
            states[1] = a[1][1]

        # 执行环境步进
        state, rew, hist, trunc, _ = env.step(actions)

        done = hist["__all__"] or trunc["__all__"]
        # 累计全局奖励
        for r in rew.values():
            reward += r

        # 渲染处理
        step += 1
        if args.render:
            env.plot(Path(args.log_path, "current.png"))
            # 控制渲染帧率
            time.sleep(0.18)

    # 保存最终评估结果
    reward = round(reward, 3)
    env.plot(Path(args.log_path, f"Ep_{epoch}_It_{step}_Rew_{reward}.png"))

def make_checkpoint(args, algo, log_dir, epoch, env=None):
    """
    保存算法检查点并执行周期性评估
    
    该函数用于保存当前算法状态、更新训练日志，并在指定周期执行模型评估。
    适用于深度学习/强化学习场景的检查点保存和验证流程。

    Args:
        args (argparse.Namespace): 包含配置参数的对象
        algo (object): 需要保存的算法模型实例
        log_dir (str): 日志文件存储目录路径
        epoch (int): 当前训练周期数/迭代次数
        env (gym.Env, optional): 用于评估的环境对象，默认为None

    Returns:
        None: 本函数不返回任何值
    """
    # 保存模型参数并更新训练日志
    algo.save()
    update_logs(args, log_dir)
    # 周期性执行模型评估（每500个epoch且开启评估时）
    if args.eval and epoch%500==0:
        # 重复评估两次取平均结果
        for _ in range(2):
            evaluate(args, algo, env, epoch)

def get_policy(args):
    """配置并构建强化学习策略算法（PPO）实例
    
    Args:
        args: 包含配置参数的命名空间对象，主要使用以下属性：
            num_workers (int): rollout workers数量
            gpu (int): 使用的GPU数量
            batch_size (int): 训练批次大小
            mini_batch_size (int): SGD小批次大小
            env_config (dict): 环境配置字典

    Returns:
        Algorithm: 配置完成的强化学习算法实例
    """
    class CustomCallback(DefaultCallbacks):
        """
        Here, the opponent's actions will be added to the episode states 
        And the current level will be tracked. 
        """
        """
        自定义回调函数，用于轨迹后处理
        功能：
        1. 将其他智能体的动作添加到当前智能体的观测中
        2. 跟踪当前关卡状态
        """
        def on_postprocess_trajectory(
            self,
            worker,
            episode,
            agent_id,
            policy_id,
            policies,
            postprocessed_batch,
            original_batches,
            **kwargs
        ):
            """轨迹后处理钩子函数，用于修改观测数据
            
            Args:
                postprocessed_batch: 待更新的后处理批次数据
                original_batches: 原始批次数据字典，按agent_id索引
            """
            # 收集所有智能体的动作
            acts = []

            to_update = postprocessed_batch[SampleBatch.CUR_OBS]
            # 获取当前智能体动作
            _, own_batch = original_batches[agent_id]
            own_act = np.squeeze(own_batch[SampleBatch.ACTIONS])
            acts.append(own_act)

            oth_agents = list(range(1,4))
            oth_agents.remove(agent_id)

            # 收集其他智能体动作（编号1-3，排除当前agent）
            for i in oth_agents:
                _, oth_batch = original_batches[i]
                oth_act = np.squeeze(oth_batch[SampleBatch.ACTIONS])
                acts.append(oth_act)
            
            # 将动作数据标准化后写入观测
            for i, act in enumerate(acts):
                to_update[:,i] = act/N_OPP_HL

    def central_critic_observer(agent_obs, **kw):
        """
        Determines which agents will get an observation. 
        In 'on_postprocess_trajectory', the keys will be called lexicographically. 
        """
        """
        集中式批评家观测生成函数
        构建每个智能体的观测字典，包含：
        - 自身观测（obs_1_own）
        - 其他智能体观测（obs_2/obs_3）
        - 动作占位符（act_*）
        
        Returns:
            dict: 按agent_id索引的观测字典，包含跨智能体的观测信息
        """
        new_obs = {
            1: {
                "obs_1_own": agent_obs[1] ,
                "obs_2": agent_obs[2],
                "obs_3": agent_obs[3],
                "act_1_own": np.zeros(1),
                "act_2": np.zeros(1),
                "act_3": np.zeros(1),
            },
            2: {
                "obs_1_own": agent_obs[2] ,
                "obs_2": agent_obs[1],
                "obs_3": agent_obs[3],
                "act_1_own": np.zeros(1),
                "act_2": np.zeros(1),
                "act_3": np.zeros(1),
            },
            3: {
                "obs_1_own": agent_obs[3] ,
                "obs_2": agent_obs[1],
                "obs_3": agent_obs[2],
                "act_1_own": np.zeros(1),
                "act_2": np.zeros(1),
                "act_3": np.zeros(1),
            },
        }
        return new_obs

    # 注册自定义模型
    ModelCatalog.register_custom_model("commander_model",CommanderGru)
    # 定义动作和观测空间
    action_space = spaces.Discrete(ACT_DIM)
    observer_space = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(OBS_DIM,)),
            "obs_2": spaces.Box(low=0, high=1, shape=(OBS_DIM,)),
            "obs_3": spaces.Box(low=0, high=1, shape=(OBS_DIM,)),
            "act_1_own": spaces.Box(low=0, high=ACT_DIM-1, shape=(1,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=ACT_DIM-1, shape=(1,), dtype=np.float32),
            "act_3": spaces.Box(low=0, high=ACT_DIM-1, shape=(1,), dtype=np.float32),
        }
    )

    # 构建PPO算法配置
    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, batch_mode="complete_episodes", enable_connectors=False)
        .resources(num_gpus=args.gpu)
        .evaluation(evaluation_interval=None)
        .environment(env=HighLevelEnv, env_config=args.env_config)
        .training(train_batch_size=args.batch_size, kl_target=0.05, gamma=0.99, clip_param=0.25,lr=1e-4, sgd_minibatch_size=args.mini_batch_size)
        .framework("torch")

        # even though only 'one commander agent' is acting, we use multi-agent to have centalized critic option -> CTDE
        .multi_agent(policies={
                "commander_policy": PolicySpec(
                    None,
                    observer_space,
                    action_space,
                    config={
                        "model": {
                            "custom_model": "commander_model"
                        }
                    }
                )
            },
            policy_mapping_fn= lambda agent_id, episode, worker, **kwargs: "commander_policy",
            observation_fn=central_critic_observer)
        .callbacks(CustomCallback)
        .build()
    )
    return algo

if __name__ == '__main__':
    ##############################################################
    # 主程序入口：强化学习训练流程设置
    # 1. 初始化RLlib日志记录器及配置参数
    # 2. 模型恢复与评估环境准备
    # 3. 启动TensorBoard可视化服务
    # 4. 执行训练循环并定期保存检查点
    ##############################################################
    
    # 初始化RLlib日志记录器（仅记录ERROR级别）
    rllib_logger = logging.getLogger("ray.rllib")
    rllib_logger.setLevel(logging.ERROR)
    # 获取配置参数并初始化算法策略
    args = Config(1).get_arguments
    test_env = None
    algo = get_policy(args)

    # 模型恢复逻辑：优先使用指定路径，否则使用默认日志路径
    if args.restore:
        if args.restore_path:
            algo.restore(args.restore_path)
        else:
            algo.restore(os.path.join(args.log_path, "checkpoint"))
    # 评估模式下的环境初始化
    if args.eval:
        test_env = HighLevelEnv(args.env_config)

    # TensorBoard服务配置与端口绑定（自动递增端口直到成功）
    log_dir = os.path.normpath(algo.logdir)
    tb = program.TensorBoard()
    port = 6006
    started = False
    while not started:
        try:
            tb.configure(argv=[None, '--logdir', log_dir, '--bind_all', f'--port={port}'])
            url = tb.launch()
            started = True
        except:
            port += 1

    print("\n", "--- NO ERRORS FOUND, STARTING TRAINING ---")

    # 训练前准备：清屏操作和时间统计初始化
    time.sleep(2)
    time_acc = 0
    iters = tqdm.trange(0, args.epochs+1,  leave=True)
    os.system('clear') if os.name == 'posix' else os.system('cls')

    # 主训练循环（含进度显示和定期检查点保存）
    for i in iters:
        t = time.time()
        result = algo.train()
        time_acc += time.time()-t
        # 更新进度条显示指标：平均奖励、单步耗时、TensorBoard地址
        iters.set_description(f"{i}) Reward = {result['episode_reward_mean']:.2f} | Avg. Episode Time = {round(time_acc/(i+1), 3)} | TB: {url} | Progress")
        
        # 每50次迭代保存检查点（含评估模式下的测试环境状态）
        if i % 50 == 0:
            make_checkpoint(args, algo, log_dir, i, test_env)

