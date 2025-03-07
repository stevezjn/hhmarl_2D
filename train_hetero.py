"""
Main file for training low-level heterogeneous agents.
HETEROGENEOUS: Agend IDs to AC types: 1->1, 2->2, 3->1, 4->2
"""

import os
import time
import shutil
import tqdm
import torch
import numpy as np
from gymnasium import spaces
from tensorboard import program
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from models.ac_models_hetero import Esc1, Esc2, Fight1, Fight2
from config import Config
from envs.env_hetero import LowLevelEnv

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29
POLICY_DIR = 'policies'

def update_logs(args, log_dir, level, epoch):
    """
    Copy stored checkpoints from Ray log to experiment log directory.
    """
    """
    将Ray训练产生的检查点文件同步到实验日志目录
    
    Args:
        args (argparse.Namespace): 包含运行时参数的对象，需包含log_path属性
        log_dir (str): Ray框架生成的原始日志目录路径
        level: 未使用的日志级别参数（保留参数）
        epoch: 未使用的训练轮次参数（保留参数）
        
    Returns:
        None: 该函数无返回值，仅执行文件操作
    """
    # 按修改时间排序获取日志子目录
    dirs = sorted(Path(log_dir).glob('*/'), key=os.path.getmtime)
    check = ''
    event = ''
    # 遍历目录寻找最新的checkpoint和events文件
    for item in dirs:
        if "checkpoint" in item.name:
            check = str(item)
        if "events" in item.name:
            event = str(item)
    
    # 准备目标检查点目录路径
    result_dir = os.path.join(args.log_path, 'checkpoint')
    
    # 清理旧检查点目录
    try:
        shutil.rmtree(result_dir)
    except:
        pass

    # 复制最新检查点和事件文件到目标目录
    shutil.copytree(check,result_dir,symlinks=False,dirs_exist_ok=False)
    shutil.copy(event,result_dir)

def evaluate(args, algo, env, epoch, level, it):
    """
    Evaluations are stored as pictures of combat scenarios, with rewards in filename.
    """
    """
    执行策略评估并保存战斗场景可视化结果
    
    Args:
        args: 包含运行时参数的配置对象（需包含log_path和render属性）
        algo: 强化学习算法对象（需实现compute_single_action方法）
        env: 对战环境对象（需实现reset/step/plot方法）
        epoch: 当前训练周期数（用于文件名标识）
        level: 难度等级（用于文件名标识）
        it: 当前迭代次数（用于文件名标识）

    功能说明:
        1. 重置环境并初始化评估状态
        2. 循环执行策略直到回合结束
        3. 累计总奖励值
        4. 根据参数决定是否实时渲染战斗过程
        5. 保存最终战斗场景截图到日志路径
    """
    def cc_obs(obs, id):
        """
        智能体观测空间转换器（针对多智能体系统）
        
        Args:
            obs: 原始观测数据
            id: 智能体标识（1或2）
            
        Returns:
            重组后的观测字典，包含：
            - 自身观测（obs_1_own）
            - 对手观测（obs_2）
            - 零初始化的自身动作空间（act_1_own）
            - 零初始化的对手动作空间（act_2）
            
        特殊处理：
            当id=2时交换观测位置，实现对称观测处理
        """
        if id == 1:
            return {
                "obs_1_own": obs[1] ,
                "obs_2": obs[2],
                "act_1_own": np.zeros(ACTION_DIM_AC1),
                "act_2": np.zeros(ACTION_DIM_AC2),
            }
        elif id == 2:
            return {
                "obs_1_own": obs[2] ,
                "obs_2": obs[1],
                "act_1_own": np.zeros(ACTION_DIM_AC2),
                "act_2": np.zeros(ACTION_DIM_AC1),
            }
    
    # 初始化环境状态
    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    # 主评估循环
    while not done:
        actions = {}
        # 为每个智能体生成动作
        for ag_id in state.keys():
            a = algo.compute_single_action(observation=cc_obs(state, ag_id), state=torch.zeros(1), policy_id=f"ac{ag_id}_policy", explore=False)
            actions[ag_id] = a[0]

        # 执行环境步进
        state, rew, term, trunc, _ = env.step(actions)
        done = term["__all__"] or trunc["__all__"]
        # 累计总奖励（所有智能体奖励之和）
        for r in rew.values():
            reward += r

        step += 1
        # 实时渲染处理
        if args.render:
            env.plot(Path(args.log_path, "current.png"))
            time.sleep(0.18)

    # 保存最终评估结果
    reward = round(reward, 3)
    env.plot(Path(args.log_path, f"Ep_{epoch}_It_{step}_Lv{level}_Rew_{reward}.png"))

def make_checkpoint(args, algo, log_dir, epoch, level, env=None):
    """保存算法状态并执行相关评估与日志更新
    
    参数说明:
        args (argparse.Namespace): 包含运行时配置参数的对象
        algo (Algorithm): 需要保存或导出模型的算法实例
        log_dir (str): 日志文件存储目录的路径
        epoch (int): 当前训练周期数(用于日志记录)
        level (int): 当前训练阶段层级标识(用于日志分类)
        env (gym.Env, optional): 用于策略评估的环境实例，默认None表示不评估
    
    返回值:
        None: 本函数无返回值
    """
    
    # 核心操作：保存算法内部状态并更新训练日志
    algo.save()
    update_logs(args, log_dir, level, epoch)
    # 对两个AC模块进行循环处理（通常对应双代理场景）
    for it in range(2):
        # 当达到3级及以上时导出策略模型文件
        if args.level >= 3:
            # 导出当前策略模型并重命名存储文件
            algo.export_policy_model(os.path.join(os.path.dirname(__file__), POLICY_DIR), f'ac{it+1}_policy')
            # 根据参数生成策略文件命名模板
            policy_name = f'L{args.level}_AC{it+1}_{args.agent_mode}'
            os.rename(f'{POLICY_DIR}/model.pt', f'{POLICY_DIR}/{policy_name}.pt')
        # 定期执行策略评估（每500个epoch且评估模式开启时）
        if args.eval and epoch%500==0:
            evaluate(args, algo, env, epoch, level, it)

def get_policy(args):
    """
    Agents get assigned the neural networks Fight1, Fight2 and Esc1, Esc2.
    """
    """
    配置并返回多智能体强化学习策略算法
    
    Args:
        args: 配置参数对象，包含以下属性：
            - agent_mode: 智能体模式（"fight"战斗模式或"escape"逃跑模式）
            - num_workers: 并行工作线程数量
            - gpu: 使用的GPU数量
            - batch_size: 训练批次大小
            - mini_batch_size: 小批次训练尺寸
            - env_config: 环境配置字典
            
    Returns:
        algo: 配置完成的强化学习算法实例
    """
    class CustomCallback(DefaultCallbacks):
        """
        This callback is used to have fully observable critic. Other agent's
        observations and actions will be added to this episode batch.

        ATTENTION: This callback is set up for 2vs2 training.  
        """
        """
        实现完全可观察评论家的轨迹后处理回调（适用于2v2训练场景）
        
        Attributes:
            通过将其他智能体的观察和动作添加到当前批次，增强critic的观察信息
        
        Args:
            worker: 策略执行的工作器实例
            episode: 当前处理的回合数据
            agent_id: 当前智能体ID（1或2）
            policy_id: 当前策略ID
            policies: 所有策略字典
            postprocessed_batch: 后处理批次数据（会被修改）
            original_batches: 原始批次数据字典（按agent_id索引）
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
            # 合并自身和友方智能体的动作到观察空间
            to_update = postprocessed_batch[SampleBatch.CUR_OBS]
            other_id = 2 if agent_id == 1 else 1
            # 获取当前智能体和友方的动作数据
            _, own_batch = original_batches[agent_id]
            own_act = np.squeeze(own_batch[SampleBatch.ACTIONS])
            _, fri_batch = original_batches[other_id]
            fri_act = np.squeeze(fri_batch[SampleBatch.ACTIONS])
            
            acts = [own_act, fri_act]

            # 根据智能体类型进行不同维度的动作归一化处理
            for i, act in enumerate(acts):
                if agent_id == 1 and i == 0 or agent_id==2 and i==1:
                    # 处理主武器动作（4维）
                    if agent_id == 1:
                        # 武器选择归一化
                        to_update[:,i*4] = act[:,0]/12.0
                        # 弹药类型归一化
                        to_update[:,i*4+1] = act[:,1]/8.0
                        # 瞄准状态
                        to_update[:,i*4+2] = act[:,2]
                        # 防御状态
                        to_update[:,i*4+3] = act[:,3]
                    else:
                        to_update[:,i*3] = act[:,0]/12.0
                        to_update[:,i*3+1] = act[:,1]/8.0
                        to_update[:,i*3+2] = act[:,2]
                        to_update[:,i*3+3] = act[:,3]
                elif agent_id == 1 and i == 1 or agent_id==2 and i==0:
                    # 处理副武器动作（3维）
                    if agent_id==1:
                        to_update[:,i*4] = act[:,0]/12.0
                        to_update[:,i*4+1] = act[:,1]/8.0
                        to_update[:,i*4+2] = act[:,2]
                    else:
                        to_update[:,i] = act[:,0]/12.0
                        to_update[:,i+1] = act[:,1]/8.0
                        to_update[:,i+2] = act[:,2]

    def central_critic_observer(agent_obs, **kw):
        """
        Determines which agents will get an observation. 
        In 'on_postprocess_trajectory', the keys will be called lexicographically. 
        """
        """
        构建中心化critic的观察空间
        
        Args:
            agent_obs: 原始观察数据字典（按agent_id索引）
            
        Returns:
            new_obs: 重组后的观察空间字典，包含：
                - obs_1_own: 自身原始观察
                - obs_2: 其他智能体观察
                - act_1_own: 自身动作占位符
                - act_2: 其他智能体动作占位符
        """
        new_obs = {
            1: {
                "obs_1_own": agent_obs[1] ,
                "obs_2": agent_obs[2],
                "act_1_own": np.zeros(ACTION_DIM_AC1),
                "act_2": np.zeros(ACTION_DIM_AC2),
            },
            2: {
                "obs_1_own": agent_obs[2] ,
                "obs_2": agent_obs[1],
                "act_1_own": np.zeros(ACTION_DIM_AC2),
                "act_2": np.zeros(ACTION_DIM_AC1),
            }
        }
        return new_obs

    # 定义智能体1的观察空间（包含自身和友方信息）
    observer_space_ac1 = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(OBS_AC1 if args.agent_mode=="fight" else OBS_ESC_AC1,)),
            "obs_2": spaces.Box(low=0, high=1, shape=(OBS_AC2 if args.agent_mode=="fight" else OBS_ESC_AC2,)),
            "act_1_own": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC1,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC2,), dtype=np.float32),
        }
    )
    # 定义智能体2的观察空间（与智能体1镜像对称）
    observer_space_ac2 = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(OBS_AC2 if args.agent_mode=="fight" else OBS_ESC_AC2,)),
            "obs_2": spaces.Box(low=0, high=1, shape=(OBS_AC1 if args.agent_mode=="fight" else OBS_ESC_AC1,)),
            "act_1_own": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC2,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC1,), dtype=np.float32),
        }
    )

    # 根据模式注册对应的神经网络模型
    if args.agent_mode == "escape":
        ModelCatalog.register_custom_model("ac1_model_esc",Esc1)
        ModelCatalog.register_custom_model("ac2_model_esc",Esc2)
    else:
        ModelCatalog.register_custom_model('ac1_model', Fight1) 
        ModelCatalog.register_custom_model('ac2_model', Fight2)

    # 定义动作空间（多离散空间）
    # [主武器选择, 弹药类型, 瞄准状态, 防御状态]
    action_space_ac1 = spaces.MultiDiscrete([13,9,2,2])
    # [副武器选择, 弹药类型, 闪避状态]
    action_space_ac2 = spaces.MultiDiscrete([13,9,2])

    # 构建PPO算法配置
    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, batch_mode="complete_episodes", enable_connectors=False) #compare with cetralized_critic_2.py
        .resources(num_gpus=args.gpu)
        .evaluation(evaluation_interval=None)
        .environment(env=LowLevelEnv, env_config=args.env_config)
        .training(kl_target=0.025, train_batch_size=args.batch_size, gamma=0.99, clip_param=0.25,lr=1e-4, lambda_=0.95, sgd_minibatch_size=args.mini_batch_size)
        .framework("torch")
        .multi_agent(policies={
                "ac1_policy": PolicySpec(
                    None,
                    observer_space_ac1,
                    action_space_ac1,
                    config={
                        "model": {
                            "custom_model": "ac1_model_esc" if args.agent_mode=="escape" else 'ac1_model'
                        }
                    }
                ),
                "ac2_policy": PolicySpec(
                    None,
                    observer_space_ac2,
                    action_space_ac2,
                    config={
                        "model": {
                            "custom_model": "ac2_model_esc" if args.agent_mode=="escape" else 'ac2_model'
                        }
                    }
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f'ac{agent_id}_policy',
            policies_to_train=["ac1_policy", "ac2_policy"],
            observation_fn=central_critic_observer)
        .callbacks(CustomCallback)
        .build()
    )
    return algo

if __name__ == '__main__':
    # 主程序入口：配置参数初始化、策略算法加载和训练流程控制
    # 参数说明：
    # - Config(0): 创建配置类实例并获取运行参数
    # - test_env: 测试环境实例（仅在评估模式时初始化）
    # - algo: 根据参数生成的强化学习策略算法实例
    args = Config(0).get_arguments
    test_env = None
    algo = get_policy(args)

    # 模型恢复逻辑：根据参数决定从指定路径或默认日志路径恢复训练
    if args.restore:
        if args.restore_path:
            algo.restore(args.restore_path)
        else:
            algo.restore(os.path.join(args.log_path, "checkpoint"))
    # 评估模式初始化：当启用eval参数时创建低级测试环境
    if args.eval:
        test_env = LowLevelEnv(args.env_config)

    # TensorBoard服务配置：自动处理端口冲突问题
    # 尝试从6006端口开始寻找可用端口，直到成功启动TensorBoard服务
    log_dir = os.path.normpath(algo.logdir)
    tb = program.TensorBoard()
    port = 6006
    started = False
    url = None
    while not started:
        try:
            tb.configure(argv=[None, '--logdir', log_dir, '--bind_all', f'--port={port}'])
            url = tb.launch()
            started = True
        except:
            port += 1

    print("\n", "--- NO ERRORS FOUND, STARTING TRAINING ---")

    # 训练前准备：清屏操作和计时器初始化
    time.sleep(2)
    time_acc = 0
    iters = tqdm.trange(0, args.epochs+1,  leave=True)
    os.system('clear') if os.name == 'posix' else os.system('cls')

    # 主训练循环：
    # 1. 调用策略算法进行训练迭代
    # 2. 实时更新进度条显示（平均奖励、关卡等级、耗时等）
    # 3. 每50次迭代执行检查点保存
    for i in iters:
        t = time.time()
        result = algo.train()
        time_acc += time.time()-t
        iters.set_description(f"{i}) Reward = {result['episode_reward_mean']:.2f} | Level = {args.level} | Avg. Episode Time = {round(time_acc/(i+1), 3)} | TB: {url} | Progress")
        
        # 定期保存检查点：包含模型参数和测试环境状态
        if i % 50 == 0:
            make_checkpoint(args, algo, log_dir, i, args.level, test_env)