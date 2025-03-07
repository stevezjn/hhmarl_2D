"""
High-Level Environment for HHMARL 2D Aircombat.
Low Level agent policies are included in this env.
"""

import os
import random
import numpy as np
from fractions import Fraction
from gymnasium import spaces
from .env_base import HHMARLBaseEnv

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

N_OPP_HL = 2 #change for sensing
OBS_OPP_HL = 10
OPP_SIZE = N_OPP_HL*OBS_OPP_HL
OBS_FRI_HL = 5
FRI_SIZE = 2 * OBS_FRI_HL
OBS_HL = 14 + N_OPP_HL*OBS_OPP_HL

class HighLevelEnv(HHMARLBaseEnv):
    """
    High-Level Environment for Aircombat Maneuvering.
    """
    def __init__(self, env_config):
        """
        初始化高级别环境配置
        
        参数:
            env_config (dict): 包含环境配置参数的字典，应包含：
                - args: 包含核心参数的命名空间对象
                - 其他可能的环境配置参数
        
        属性初始化:
            - 基础参数配置
            - 环境观测/动作空间定义
            - 智能体标识集合
            - 父类环境初始化
        """
        # 参数处理与默认值设置
        self.args = env_config.get("args", None)
        # 子步骤基准值
        self.n_sub_steps = 15
        # 最小保证子步骤数
        self.min_sub_steps = 10

        # 定义环境空间（观测空间与动作空间）
        self.observation_space = spaces.Box(low=np.zeros(OBS_HL), high=np.ones(OBS_HL), dtype=np.float32)
        self.action_space = spaces.Discrete(N_OPP_HL+1)
        # 智能体配置
        # 生成从1开始的智能体ID集合
        self._agent_ids = set(range(1,self.args.num_agents+1))
        # 指挥官指令存储
        self.commander_actions = None

        # 父类环境初始化
        super().__init__(self.args.map_size)
        # 策略加载方法调用
        self._get_policies("HighLevel")

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态
        Args:
            seed (int, optional): 随机数生成种子，当前未实际使用但保留参数接口
            options (dict, optional): 环境重置配置选项，用于传递额外参数
            
        Returns:
            tuple: 包含两个元素的元组
                - state: 环境重置后的初始状态，由self.state()返回具体内容
                - dict: 预留的信息字典，当前固定返回空字典"""
        
        # 调用父类reset方法并强制设置运行模式为HighLevel
        super().reset(options={"mode":"HighLevel"})
        # 重置指挥官指令状态
        self.commander_actions = None
        return self.state(), {}

    def state(self):
        """
        High Level state for commander.
        self.opp_to_attack[agent_id] = [[opp_id1, opp_dist2], [opp_id2, opp_dist2], ...]
        """
        """
        生成指挥官使用的高级状态信息
        
        功能描述:
            遍历所有单位(包括己方和敌方)，收集每个己方智能体的高级状态信息：
            1. 基础状态：归一化后的坐标、速度、航向角
            2. 敌方状态：附近敌方单位的特征值(距离、方位等)
            3. 友军状态：附近友军的特征值
            
        返回值:
            dict: 状态字典，键为智能体ID，值为由以下部分组成的numpy数组:
                [基础状态(4维), 敌方特征(OPP_SIZE维), 友军特征(FRI_SIZE维)]
        """
        state_dict = {}

        # 遍历所有单位（包含敌方单位）
        for ag_id in range(1, self.args.total_num+1):
            self.opp_to_attack[ag_id] = []
            # 处理己方智能体
            if ag_id <= self.args.num_agents:
                # 获取单位存在性检查
                if self.sim.unit_exists(ag_id):
                    # 初始化状态容器
                    state = []
                    # 获取附近敌方单位
                    opps = self._nearby_object(ag_id)
                    if opps:
                        # 提取基础状态: 坐标、速度、航向
                        unit = self.sim.get_unit(ag_id)
                        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                        state.append(x)
                        state.append(y)
                        state.append(np.clip(unit.speed/unit.max_speed,0,1))
                        state.append(np.clip((unit.heading%359)/359, 0, 1))
                        # 处理敌方特征
                        opp_state = []
                        for opp in opps:
                            opp_state.extend(self.opp_ac_values("HighLevel", opp[0], ag_id, opp[1]))
                            self.opp_to_attack[ag_id].append(opp) #store opponents, Commander decides which to attack
                            if len(opp_state) == OPP_SIZE:
                                break
                        if len(opp_state) < OPP_SIZE:
                            # 填充不足部分
                            opp_state.extend(np.zeros(OPP_SIZE-len(opp_state)))
                        # 处理友军特征
                        fri_state = []
                        fri = self._nearby_object(ag_id, True)
                        for f in fri:
                            fri_state.extend(self.friendly_ac_values(ag_id, f[0]))
                            if len(fri_state) == FRI_SIZE:
                                break
                        if len(fri_state) < FRI_SIZE:
                            # 填充不足部分
                            fri_state.extend(np.zeros(FRI_SIZE-len(fri_state)))
                        # 合并状态
                        state.extend(opp_state)
                        state.extend(fri_state)
                        # 状态维度验证
                        assert len(state) == OBS_HL, f"Hierarchy state len {len(state)} is not as required ({OBS_HL}) for agent {ag_id}"
                    else:
                        state = np.zeros(OBS_HL, dtype=np.float32)
                else:
                    state = np.zeros(OBS_HL, dtype=np.float32)
                state_dict[ag_id] = np.array(state, dtype=np.float32)
            # 处理敌方单位
            else:
                if self.sim.unit_exists(ag_id):
                    # 仅记录最近的对抗目标
                    # for opponents, only need to store the closest agent
                    # explicitly exclude from the loop above to not fill values in state_dict
                    self.opp_to_attack[ag_id] = self._nearby_object(ag_id)
        return state_dict

    def lowlevel_state(self, mode, agent_id, unit):
        """
        Lowlevel state for fight or escape.
        the opponent to attack is fixed for the whole low-level cycle, 
        but the closest friendly aircraft will be determined new every round.
        """
        """
       生成战术级智能体的实时观测状态，支持战斗/逃脱双模式策略
        
        参数:
            mode (str): 战术行为模式标识
                - "fight": 攻击模式，锁定预设攻击目标
                - 其他值: 逃脱模式，维持默认对抗目标
            agent_id (int): 智能体唯一标识符，用于跨单位状态追踪
            unit (Unit): 承载战术单元状态的对象，需包含至少：
                - position: 含lon(经度)、lat(纬度)属性的位置对象
                - heading: 当前航向角（度）
                - speed: 当前速度（单位：节）
                
        返回:
            dict: 标准化状态空间字典，结构为 {agent_id: np.ndarray}
            状态向量组成（示例）：
                [相对经度, 相对纬度, 敌方航向, 友方距离, 自身燃油状态, ...]
        """
        # 获取最近友方单位ID（每回合动态计算）
        fri_id = self._nearby_object(agent_id, friendly=True)
        fri_id = fri_id[0][0] if fri_id else None
        # 根据模式选择状态生成策略
        if mode == "fight":
            # 战斗模式下使用预设的攻击目标和友方单位计算状态值
            state = self.fight_state_values(agent_id, unit, self.opp_to_attack[agent_id][self.commander_actions[agent_id]-1], fri_id)
        else:
            # 逃脱模式下使用默认攻击目标和友方单位计算状态值
            state = self.esc_state_values(agent_id, unit, self.opp_to_attack[agent_id], fri_id)
        # 返回标准化状态向量字典
        return {agent_id:np.array(state, dtype=np.float32)}

    def _take_action(self, commander_actions):
        """执行指挥官指令并计算环境反馈
        
        Args:
            commander_actions (dict): 指挥官为每个单位分配的动作指令
                key为单位ID，value为动作编号（0表示逃跑，非0表示战斗）
        
        Returns:
            无返回值，结果通过类属性self.rewards和内部状态更新体现
        """
        # 初始化环境步数计数器和事件标志
        s = 0
        # 击杀事件触发标志
        kill_event = False
        # 特殊环境事件标志
        situation_event = False
        rewards = {}
        self.commander_actions = commander_actions
        # 环境交互最小步数阈值
        self.min_sub_steps = 10 #random.randint(10,16)

        # 评估初始动作并初始化奖励
        # Select opp to attack and assess action
        rewards = self._action_assess(rewards)

        # 主环境交互循环
        while s <= self.n_sub_steps and not kill_event and not situation_event:
            # 遍历所有作战单位
            for i in range(1, self.args.total_num+1):
                if self.sim.unit_exists(i):
                    u = self.sim.get_unit(i)
                    # 根据指挥官指令选择行为策略（逃跑/战斗）
                    actions = self._policy_actions(policy_type="escape" if self.commander_actions[i]==0 else "fight", agent_id=i, unit=u)
                    # 执行高层决策动作
                    self._take_base_action("HighLevel", u, i, self.opp_to_attack[i][self.commander_actions[i]-1][0], actions)

            # 计算环境反馈和事件状态
            rewards, kill_event = self._get_rewards(rewards, self.sim.do_tick())
            # 达到最小步数后开始检测环境事件
            if s > self.min_sub_steps:
                #take at least min_sub_steps until checking for surrounding event
                situation_event = self._surrounding_event()

            # 更新步数计数器
            s += 1
            self.steps += 1
        # 保存最终奖励数据
        self.rewards = rewards
        return

    def _action_assess(self, rewards):
        """
        Select the opponent to attack based on commander action.
        Punish for chosing not existing opponent, reward for choosing to fight in favourable situation.
        If an opponent is chosen that does not exist (observation filled with zeros), 
        we assign the closest opponent to attack -> self.commander_actions[i] = 1 (because 0 is escape)

        self.commander_actions will be expanded to include closest agents to attack w.r.t opponents
        """
        """
        评估并更新智能体的战斗行为奖励
        
        参数:
        - rewards: list[float] 包含所有智能体奖励值的列表，将被直接修改
        
        返回:
        - list[float] 更新后的奖励值列表
        
        功能说明:
        - 根据指挥官动作选择攻击目标，对无效选择施加惩罚，对有利战斗给予奖励
        - 处理智能体和敌方单位的不同行为逻辑，包含：
            1. 有效攻击目标的验证与调整
            2. 距离和视角角度的战斗条件评估
            3. 敌方单位的随机战斗行为生成
        """
        # 遍历所有单位（包含己方智能体和敌方单位）
        for i in range(1,self.args.total_num+1):
            if self.sim.unit_exists(i):
                # 处理己方智能体逻辑
                if i <= self.args.num_agents:
                    rewards[i] = 0
                    # 当选择攻击动作时
                    if self.commander_actions[i] > 0:
                        # 验证目标对手的有效性
                        try:
                            opp_id = self.opp_to_attack[i][self.commander_actions[i]-1][0]
                        except:
                            opp_id = None
                            # 默认选择最近目标
                            self.commander_actions[i] = 1

                        # 无效目标惩罚
                        if not opp_id: rewards[i] = -0.1
                        # 战斗条件评估（距离和角度）
                        if self.args.hier_action_assess and opp_id:
                            if self._distance(i, opp_id) < 0.1 and self._focus_angle(i, opp_id) < 15 and self._focus_angle(opp_id, i) > 40:
                                rewards[i] = 0.1
                            else:
                                rewards[i] = 0
                    # 未选择攻击动作时的评估
                    else:
                        if self.args.hier_action_assess:
                            cl_opp = self.opp_to_attack[i][0][0]
                            if self._distance(cl_opp, i) < 0.1 and self._focus_angle(cl_opp, i) < 15 and self._focus_angle(i, cl_opp) > 40:
                                rewards[i] = 0.1
                # 处理敌方单位逻辑
                else:
                    # 生成随机战斗决策
                    # determine if opponents select pi_fight with fight probability
                    p_of = Fraction(self.args.hier_opp_fight_ratio, 100).limit_denominator().as_integer_ratio()
                    fight = bool(random.choices([0, 1], weights=[p_of[1]-p_of[0], p_of[0]], k=1)[0])
                    # 选择攻击目标
                    if fight:
                        possible_agent_ids = len(self.opp_to_attack[i])
                        if possible_agent_ids > 1 and bool(random.choices([0, 1], weights=[1, 3], k=1)[0]):
                            # randomly select another agent to attack
                            ag_id = random.randint(2,possible_agent_ids)
                        else:
                            ag_id = 1
                            # 逃跑动作
                    else: ag_id = 0 #escape
                    # define commander actions for opponents
                    self.commander_actions[i] = ag_id
            # 处理不存在单位的情况
            else:
                if i <= self.args.num_agents: rewards[i] = 0
                self.commander_actions[i] = None
        return rewards

    def _surrounding_event(self):
        """
        检测是否存在符合近距离包围条件的敌对单位事件
        
        通过遍历所有己方代理与敌方单位的组合，检查是否存在满足以下条件的情况：
        1. 两者距离小于0.1单位
        2. 至少有一方的注视角度小于15度
        
        Returns:
            bool: 是否检测到包围事件，True表示存在符合条件的敌对单位
        """
        def eval_event(ag_id, opp_id):
            """
            评估指定单位对是否触发包围条件
            
            Args:
                ag_id: 己方单位标识符
                opp_id: 敌方单位标识符
                
            Returns:
                bool: 是否满足包围条件
            """
            # 判断距离和注视角度的组合条件
            if self._distance(ag_id, opp_id) < 0.1:
                # 检查任意一方的注视角度是否满足条件
                if self._focus_angle(ag_id, opp_id) < 15 or self._focus_angle(opp_id, ag_id) < 15:
                    return True
            return False
        
        event = False
        # 遍历所有己方代理与敌方单位的组合
        for i in range(1, self.args.num_agents+1):
            for j in range(self.args.num_agents+1, self.args.total_num+1):
                # 检查单位实体是否存在后进行事件评估
                if self.sim.unit_exists(i) and self.sim.unit_exists(j):
                    event = eval_event(i, j)
                # 发现事件立即终止检测
                if event:
                    break
            if event:
                break
        return event

    def _get_rewards(self, rewards, events):
        """
        计算并分配多智能体系统的战斗奖励
        
        参数:
            rewards (dict/list): 存储各智能体当前奖励值的容器，键/索引为智能体ID
            events (list): 包含战斗事件信息的列表，用于计算即时战斗奖励
            
        返回:
            tuple: 包含两个元素的元组
                - 更新后的奖励容器 (与输入rewards同类型)
                - 击杀事件标识 (kill_event): 表示是否发生关键击杀事件的布尔值
        """
        rews, destroyed_ids, kill_event = self._combat_rewards(events, mode="HighLevel")

        # 汇总所有收集的奖励
        #sum all collected rewards together
        for i in range(1, self.args.num_agents+1):
            # 仅处理现存或被摧毁的智能体
            if self.sim.unit_exists(i) or i in destroyed_ids:
                # 当存在全局奖励分配系数时
                if self.args.glob_frac > 0:
                    # 合并其他智能体的奖励
                    #incorporate rewards of other aircraft
                    ids = list(range(1,self.args.num_agents+1))
                    ids.remove(i)
                    # 计算自身奖励 + 全局奖励分配部分
                    rewards[i] += sum(rews[i]) + self.args.glob_frac*sum(sum(rews[j]) for j in ids)
                else:
                    # 仅累加自身奖励
                    rewards[i] += sum(rews[i])

        return rewards, kill_event

    def _sample_state(self, agent, i, r):
        """
        生成智能体的随机初始状态
        
        参数：
        agent (str): 智能体类型标识，"agent"表示己方智能体，其他值表示敌方
        i (int): 智能体索引编号，用于计算纵向偏移
        r (int): 区域标识，1或2表示不同的横向生成区域
        
        返回值：
        (x, y, a): 生成的三元组，包含横坐标、纵坐标和朝向角度(0-359度)
        """
        x = 0
        y = 0
        a = 0
        # 己方智能体生成逻辑
        if agent == "agent":
            # 区域1：左半区生成，区域2：右半区生成
            if r == 1:
                # 左半区x坐标范围
                x = random.uniform(7.07, 7.22)
                # 纵向基础范围5.07-5.12，叠加索引偏移量
                y = random.uniform(5.07 + i*(0.4/self.args.num_agents), 5.12 + i*(0.4/self.args.num_agents))
                a = random.randint(0, 359)
            elif r == 2:
                # 右半区x坐标范围
                x = random.uniform(7.28, 7.43)
                y = random.uniform(5.07 + i*(0.4/self.args.num_agents), 5.12 + i*(0.4/self.args.num_agents))
                a = random.randint(0, 359)

        # 敌方智能体生成逻辑
        else:
            # 区域标识与己方智能体区域相反
            if r == 1:
                # 右半区x坐标范围
                x = random.uniform(7.28, 7.43)
                # 使用敌方单位数量计算纵向偏移
                y = random.uniform(5.07 + i*(0.4/self.args.num_opps), 5.12 + i*(0.4/self.args.num_opps))
                a = random.randint(0, 359)
            elif r == 2:
                # 左半区x坐标范围
                x = random.uniform(7.07, 7.22)
                y = random.uniform(5.07 + i*(0.4/self.args.num_opps), 5.12 + i*(0.4/self.args.num_opps))
                a = random.randint(0, 359)
        
        return x,y,a
