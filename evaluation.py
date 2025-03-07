from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog
import numpy as np
import torch
import os
import tqdm
from config import Config
import time
import json
from pathlib import Path
from envs.env_hier import HighLevelEnv
from models.ac_models_hier import CommanderGru
ModelCatalog.register_custom_model("commander_model",CommanderGru)

N_OPP_HL = 2 #sensing
OBS_DIM = 14+10*N_OPP_HL

MODEL_NAME = "Commander_3_vs_3" #name of commander model in folder 'results'
N_EVALS = 1000


def cc_obs(obs):
    """将观测数据转换为包含多智能体系统观测和动作空间的字典结构
    
    该函数用于构建符合多智能体系统接口要求的标准化观测字典，包含当前智能体的观测、
    其他智能体的占位观测，以及动作空间的占位数据。

    Args:
        obs (ndarray): 当前智能体的观测数据，维度应与OBS_DIM常量定义一致
    
    Returns:
        dict: 包含以下键值的字典结构：
            - obs_1_own (ndarray): 当前智能体的观测数据
            - obs_2 (ndarray): 其他智能体的观测占位数组（OBS_DIM维零向量）
            - obs_3 (ndarray): 其他智能体的观测占位数组（OBS_DIM维零向量）
            - act_1_own (ndarray): 当前智能体的动作占位数组（1维零向量）
            - act_2 (ndarray): 其他智能体的动作占位数组（1维零向量）
            - act_3 (ndarray): 其他智能体的动作占位数组（1维零向量）
            
    注：OBS_DIM应为预定义的观测维度常量，所有零向量均使用float32数据类型
    """
    return {
        "obs_1_own": obs,
        "obs_2": np.zeros(OBS_DIM, dtype=np.float32),
        "obs_3": np.zeros(OBS_DIM, dtype=np.float32),
        "act_1_own": np.zeros(1),
        "act_2": np.zeros(1),
        "act_3": np.zeros(1),
    }

def evaluate(args, env, algo, epoch, eval_stats, eval_log):
    """评估智能体在环境中的表现并记录统计数据
    
    Args:
        args: 配置参数对象，包含评估所需的各种设置
        env: 环境实例，提供reset/step等交互接口
        algo: 算法实例，包含计算动作的策略
        epoch: 当前训练周期数，用于结果记录
        eval_stats: 评估统计字典，用于累计各项指标
        eval_log: 日志文件存储路径对象
    Returns:
        None: 无返回值，通过eval_stats和文件输出记录结果
    """
    
    # 环境初始化
    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    info = {}

    # 主评估循环
    while not done:
        actions = {}
        # LSTM初始隐藏状态
        states = [torch.zeros(200), torch.zeros(200)]

        # 分层决策模式：使用算法计算每个智能体动作
        if args.eval_hl:
            for ag_id, ag_s in state.items():
                # 计算单个智能体动作及更新LSTM状态
                a = algo.compute_single_action(obs=cc_obs(ag_s), state=states, explore=False)
                # 提取动作值
                actions[ag_id] = a[0]
                # 更新LSTM第一个隐藏状态
                states[0] = a[1][0]
                # 更新LSTM第二个隐藏状态
                states[1] = a[1][1]
        # 默认决策模式：所有智能体执行固定动作
        else:
            # if no commander involved, assign closest opponent for each agent. 
            for n in range(1,args.num_agents+1):
                # 分配默认动作值1
                actions[n] = 1
        # 执行环境步进
        state, rew, hist, trunc, info = env.step(actions)
        # 累加全局奖励
        for r in rew.values(): reward += r
        # 判断终止条件
        done = hist["__all__"] or trunc["__all__"]
        step += 1

        # 更新评估统计数据
        for k, v in info.items(): eval_stats[k] += v
        # 记录总动作次数
        eval_stats["total_n_actions"]+=1

    # 定期保存环境可视化结果
    if epoch %100 ==0:
        env.plot(Path(eval_log, f"Ep_{epoch}_Step_{step}_Rew_{round(reward,2)}.png"))

    return

def postprocess_eval(ev, eval_file):
    """
    处理评估数据并生成统计报告
    
    参数:
    ev (dict): 包含原始评估指标的字典，需要包含以下键：
        agents_win, opps_win, draw, agent_fight, agent_steps, agent_escape,
        opp_fight, opp_steps, opp_escape, opp1, opp2, opp3
    eval_file (str): 输出JSON文件的保存路径
    
    处理流程:
    1. 将原始计数转换为百分比形式
    2. 输出格式化结果到控制台
    3. 将统计结果持久化到JSON文件
    """
    # 计算基础胜负比例
    #calculate fractions
    win = (ev["agents_win"] / N_EVALS) * 100
    lose = (ev["opps_win"] / N_EVALS) * 100
    draw = (ev["draw"] / N_EVALS) * 100
    # 己方行为分析（按步骤次数计算比例）
    fight = (ev["agent_fight"] / ev["agent_steps"]) *100
    esc = (ev["agent_escape"] / ev["agent_steps"]) *100
    # 对手行为分析（按步骤次数计算比例）
    fight_opp = (ev["opp_fight"] / ev["opp_steps"]) *100
    esc_opp = (ev["opp_escape"] / ev["opp_steps"]) *100
    # 对手类型分布（按战斗次数计算比例）
    opp1 = (ev["opp1"] / ev["agent_fight"]) *100
    opp2 = (ev["opp2"] / ev["agent_fight"]) *100
    opp3 = (ev["opp3"] / ev["agent_fight"]) *100
    evals = {"win":win, "lose":lose, "draw":draw, "fight":fight, "esc":esc, "fight_opp":fight_opp, "esc_opp":esc_opp, "opp1":opp1, "opp2":opp2, "opp3":opp3}
    # 控制台输出格式化结果
    for k,v in evals.items():
        print(f"{k}: {round(v,2)}")
    # 结果持久化到JSON文件
    with open(eval_file, 'w') as file:
        json.dump(evals, file, indent=3)

if __name__ == "__main__":
    """
    主程序入口：策略模型评估脚本
    功能：执行多轮策略评估，收集战斗统计数据并保存结果
    流程：
        1. 初始化评估日志路径
        2. 创建强化学习环境
        3. 加载预训练策略模型
        4. 执行多轮对抗评估
        5. 统计并保存评估结果
    """
    # 计时器初始化
    t1 = time.time()
    # 路径配置模块
    # 获取实验配置参数
    args = Config(2).get_arguments

    # 基础日志目录
    log_base = os.path.join(os.getcwd(),'results')
    # 模型检查点路径
    check = os.path.join(log_base, MODEL_NAME, 'checkpoint')
    # 根据评估模式生成配置前缀
    config = "Commander_" if args.eval_hl else "Low-Level_"
    # 构建完整配置标识
    config = config + f"{args.num_agents}-vs-{args.num_opps}"
    # 评估日志初始化
    # 评估日志目录
    eval_log = os.path.join(log_base, "EVAL_"+config)
    # 指标存储文件
    eval_file = os.path.join(eval_log, f"Metrics_{config}.json")
    # 创建日志目录
    if not os.path.exists(eval_log): os.makedirs(eval_log)

    # 环境初始化模块
    # 创建高层决策环境实例
    env = HighLevelEnv(args.env_config)

    # if evaluating purely low-level policies, we don't need commander.
    # 策略加载模块（根据评估模式决定是否加载指挥官策略）
    policy = Policy.from_checkpoint(check, ["commander_policy"])["commander_policy"] if args.eval_hl else None
    # 评估统计指标初始化
    eval_stats = {"agents_win": 0, "opps_win": 0, "draw": 0, "agent_fight": 0, "agent_escape":0, "opp_fight":0, "opp_escape":0, \
                  "agent_steps":0, "opp_steps":0, "total_n_actions":0 ,\
                    "opp1":0, "opp2":0, "opp3":0}
    # 评估循环模块
    # 带进度条的多轮评估迭代器
    iters = tqdm.trange(0, N_EVALS,  leave=True)
    for n in iters:
        # 单轮评估执行
        evaluate(args, env, policy, n, eval_stats, eval_log)

    # 结果处理模块
    print("------RESULTS:")
    # 统计结果后处理及保存
    postprocess_eval(eval_stats, eval_file)
    # 时间统计输出
    # 输出总耗时
    print(f"------TIME: {round(time.time()-t1, 3)} sec.")
