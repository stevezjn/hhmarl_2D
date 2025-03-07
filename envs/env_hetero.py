"""
Low-Level Environment for HHMARL 2D Aircombat.
"""
import random
import numpy as np
from gymnasium import spaces
from .env_base import HHMARLBaseEnv

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

class LowLevelEnv(HHMARLBaseEnv):
    """
    Low-Level Environment for Aircombat Maneuvering.
    """
    def __init__(self, env_config):
        """初始化多智能体环境配置

        Args:
            env_config (dict): 环境配置字典，需包含：
                - args: 参数对象，需包含以下属性：
                    * agent_mode (str): 智能体行为模式["fight"战斗/"escape"逃亡]
                    * num_agents (int): 环境中的智能体数量
                    * map_size (int): 地图尺寸，用于父类初始化
                    * level (int): 训练课程等级（4/5级启用策略初始化）
                - 其他隐式环境参数（通过代码使用推断）
        """
        # 核心参数初始化
        self.args = env_config.get("args", None)
        self.agent_mode = self.args.agent_mode
        # 默认对手行为模式
        self.opp_mode = "fight"

        # 观察空间维度配置
        # 作战模式观察维度映射
        self.obs_fight = {1: OBS_AC1, 2:OBS_AC2, 3:OBS_AC1, 4:OBS_AC2}
        # 逃亡模式观察维度映射
        self.obs_esc = {1: OBS_ESC_AC1, 2:OBS_ESC_AC2, 3:OBS_ESC_AC1, 4:OBS_ESC_AC2}
        # 根据模式选择观察空间
        self.obs_dim_map = self.obs_fight if self.agent_mode == "fight" else self.obs_esc

        # 配置标准化观察空间（每个智能体独立Box空间）
        self._obs_space_in_preferred_format = True
        self.observation_space = spaces.Dict({
            1: spaces.Box(low=np.zeros(self.obs_dim_map[1]), high=np.ones(self.obs_dim_map[1]), dtype=np.float32),
            2: spaces.Box(low=np.zeros(self.obs_dim_map[2]), high=np.ones(self.obs_dim_map[2]), dtype=np.float32),
            3: spaces.Box(low=np.zeros(self.obs_dim_map[3]), high=np.ones(self.obs_dim_map[3]), dtype=np.float32),
            4: spaces.Box(low=np.zeros(self.obs_dim_map[4]), high=np.ones(self.obs_dim_map[4]), dtype=np.float32)
            }
        )

        # 配置多维度离散动作空间
        self._action_space_in_preferred_format = True
        self.action_space = spaces.Dict({
            1: spaces.MultiDiscrete([13,9,2,2]), 
            2: spaces.MultiDiscrete([13,9,2]), 
            3: spaces.MultiDiscrete([13,9,2,2]), 
            4: spaces.MultiDiscrete([13,9,2])
            }
        )

        # 动态生成智能体ID集合
        self._agent_ids = set(range(1,self.args.num_agents+1))

        # 初始化父类环境（传递地图尺寸参数）
        super().__init__(self.args.map_size)

        # fictitious Self-Play (sp), starting from L4 on
        # 策略初始化（虚构自我对抗机制）
        # L4级开始启用策略共享
        if self.args.level >= 4:
            # 加载基线策略
           self._get_policies("LowLevel")
        # 特殊对手模式切换（L5级逃亡训练）
        if self.args.level == 5 and self.agent_mode == "escape": self.opp_mode = "fight"

    def reset(self, *, seed=None, options=None):
        """
        重置环境并初始化新一局游戏的参数
        
        参数:
            self: 类实例的引用
            seed (int, optional): 随机种子（当前代码未实际使用该参数）
            options (dict, optional): 环境配置参数（会被强制覆盖为低阶模式）
            
        返回值:
            tuple: 包含当前环境状态和空字典的元组（符合gym.Env标准返回格式）
        """
        # 强制调用父类reset方法并锁定为低阶模式
        super().reset(options={"mode":"LowLevel"})
        # 在第五关的战斗模式下随机初始化对手行为
        if self.args.level == 5 and self.args.agent_mode=="fight":
            #randomly select opponent behavior every new episode in L5.
            # 随机选择策略编号并同步对手模式
            k = random.randint(3,5)
            self.policy = self.policies[k]
            self.opp_mode = "escape" if k == 5 else "fight"
        return self.state(), {}
    
    def state(self):
        return self.lowlevel_state(self.agent_mode)

    def lowlevel_state(self, mode, agent_id=None, **kwargs):
        """
        Current observation in fight / esc mode, stored in state_dict. 
        Destroyed agent's observation is filled with zeros, needed for Ray callback (centralized critic).
        """
        """
        获取智能体在战斗/逃跑模式下的底层状态观测值
        
        参数：
            mode (str): 模式选择，"fight"表示战斗模式，"esc"表示逃跑模式
            agent_id (int, optional): 指定单个智能体ID，默认处理全部智能体
            **kwargs: 保留扩展参数
            
        返回：
            dict: 包含各智能体观测状态的字典，键为智能体ID，值为numpy数组
                 被摧毁的智能体状态用零填充，保证固定维度用于强化学习框架
        
        功能说明：
            - 构建符合强化学习框架要求的状态观测数据
            - 自动处理智能体存在性检查及异常状态填充
            - 包含观测维度验证机制确保数据一致性
        """

        def fri_ac_id(agent_id):
            """辅助函数：根据智能体ID映射友军标识"""
            if agent_id<=self.args.num_agents:
                return 1 if agent_id == 2 else 2
            else:
                return 3 if agent_id == 4 else 4

        # 动态确定观测维度：战斗模式与逃跑模式使用不同维度配置
        # 需重新定义obs_dim以应对模式切换时维度变化的情况
        #need to define obs_dim again, because in L5, agent is in fight mode and opp may switch to esc mode.
        obs_dim = self.obs_fight if mode == "fight" else self.obs_esc
        state_dict = {}

        # 确定处理范围：单个智能体或全部智能体
        if agent_id:
            start = agent_id
            end = agent_id +1
        else:
            start = 1
            end = self.args.num_agents+1

        # 核心处理循环：遍历指定范围内的智能体
        for ag_id in range(start, end):
            # 重置攻击目标
            self.opp_to_attack[ag_id] = None
            # 智能体存在性检查
            if self.sim.unit_exists(ag_id):
                opps = self._nearby_object(ag_id)
                if opps:
                    # 状态生成分支：存在可交互对象时
                    unit = self.sim.get_unit(ag_id)
                    state = self.fight_state_values(ag_id, unit, opps[0], fri_ac_id(ag_id)) if mode == "fight" else self.esc_state_values(ag_id, unit, opps, fri_ac_id(ag_id))
                    # 记录攻击目标
                    self.opp_to_attack[ag_id] = opps[0][0]
                    # 维度验证：确保生成状态符合预期维度
                    assert len(state) == obs_dim[ag_id], f"{mode} state len {len(state)} is not as required ({obs_dim[ag_id]}) for agent {ag_id}"
                else:
                    # 无交互对象时填充零状态
                    state = np.zeros(obs_dim[ag_id], dtype=np.float32)
            else:
                # 智能体不存在时填充零状态
                state = np.zeros(obs_dim[ag_id], dtype=np.float32)

            # 状态字典更新：确保数据类型一致性
            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _take_action(self, action):
        """
        Apply actions to agents and opponents and get rewards.
        Opponent behavior: 
            -L1 and L2 random
            -L3 engage nearby agent (method _hardcoded_opp())
            -L4 previous policy (L3)
            -L5 previous policies + escape.
        """
        """
        参数:
            action (object): 代理执行的动作集合，具体类型取决于调用环境
            
        返回值:
            None: 结果通过更新self.rewards属性返回
            
        Opponent behavior logic:
            - L1 and L2: 随机行为模式
            - L3: 使用_hardcoded_opp()方法与附近代理交战
            - L4: 继承L3行为模式
            - L5: 继承之前策略并增加逃跑逻辑
        """
        self.steps += 1
        rewards = {}
        opp_stats = {}

        def __opp_level1(unit, unit_id):
            """处理L1级别对手行为：随机发射导弹"""
            # 满足多重随机条件时发射导弹
            if not unit.actual_missile and self.steps % 40 in range(3) and bool(random.randint(0,1)) and self.missile_wait[i] == 0 and unit.ac_type == 1:
                d_ag = self._nearby_object(unit_id)
                if d_ag:
                    unit.fire_missile(unit, self.sim.get_unit(d_ag[0][0]), self.sim)
                    self.missile_wait[unit_id] = 5

        def __opp_level2(unit, unit_id):
            """处理L2级别对手行为：随机移动和开火"""
            # 基础射击行为
            unit.fire_cannon()
            # 周期性随机改变运动状态
            if self.steps<=5 or self.steps % random.randint(35,45) <= 5:
                r = random.randint(0,1)
                unit.set_heading((unit.heading + ((-1)**r)*90)%360) #+90 = max turn right, -90 = max turn left
                s = 100 + random.randint(0,4)*75
                unit.set_speed(s)
            # 继承L1的导弹发射逻辑
            if not unit.actual_missile and self.steps % 40 in range(3) and bool(random.randint(0,1)) and self.missile_wait[unit_id] == 0 and unit.ac_type == 1:
                d_ag = self._nearby_object(unit_id)
                if d_ag:
                    unit.fire_missile(unit, self.sim.get_unit(d_ag[0][0]), self.sim)
                    self.missile_wait[unit_id] = 5

        def __opp_level3(unit, unit_id):
            """处理L3及以上级别对手行为：包含逃跑逻辑的智能行为"""
            # 初始化逃跑状态
            if self.steps % 60 == 0 and not self.hardcoded_opps_escaping:
                self.hardcoded_opps_escaping = bool(random.randint(0, 1))
                if self.hardcoded_opps_escaping:
                    self.opps_escaping_time = int(random.uniform(20, 30))

            # 执行逃跑或常规策略
            if self.hardcoded_opps_escaping:
                #in order to prevent from keeping circulating to tail of agent, we randomly set opponents to escape.
                opp, heading, speed, fire, fire_missile, _ = self._escaping_opp(unit)
                self.opps_escaping_time -= 1
                if self.opps_escaping_time <= 0:
                    self.hardcoded_opps_escaping = False
            else:
                opp, heading, speed, fire, fire_missile, _ = self._hardcoded_opp(unit, unit_id)
            # 应用行为决策
            unit.set_heading(heading)
            unit.set_speed(speed)
            if fire:
                unit.fire_cannon()
            if fire_missile and opp and not unit.actual_missile and self.missile_wait[unit_id] == 0 and unit.ac_type == 1:
                unit.fire_missile(unit, self.sim.get_unit(opp), self.sim)
                self.missile_wait[unit_id] = 10

        # 主循环处理所有单位
        for i in range(1, self.args.total_num+1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                # 处理代理和高级别对手
                if i <= self.args.num_agents or self.args.level >= 4:
                    if i >= self.args.num_agents+1:
                        actions = self._policy_actions(policy_type=self.opp_mode, agent_id=i, unit=u)
                    else:
                        actions = action
                        rewards[i] = 0
                        # 收集对手状态数据
                        if self.sim.unit_exists(self.opp_to_attack[i]):
                            opp_stats[i] = [self._focus_angle(self.opp_to_attack[i], i, True), self._distance(i, self.opp_to_attack[i])]

                    rewards = self._take_base_action("LowLevel",u,i,self.opp_to_attack[i],actions,rewards)

                # 处理不同级别的对手行为
                else:
                    if self.args.level == 1:
                        __opp_level1(u, i)

                    elif self.args.level == 2:
                        __opp_level2(u, i)

                    elif self.args.level == 3:
                        __opp_level3(u, i)

        # 执行物理模拟并计算最终奖励
        # self.sim.do_tick simulates the dynamics
        self.rewards = self._get_rewards(rewards, self.sim.do_tick(), opp_stats)
        return

    def _get_rewards(self, rewards, events, opp_stats):
        """
        Calculating Rewards. 
        First check for out-of-boundary, then killing rewards.
        rewards are collected in dict 'rews' in order to sum them together and 
        possibly add a global fraction 'glob_frac' (to incorporate rewads of cooperating agents).
        """
        """
        Args:
            rewards (dict): 用于累积各智能体奖励的字典，键为智能体ID，值为数值
            events (list): 战斗事件列表，记录击杀、受伤等事件
            opp_stats (dict): 对手状态信息，用于计算战斗奖励
            
        Returns:
            dict: 更新后的奖励字典，包含各智能体累计奖励
            
        处理流程：
        1. 通过战斗事件计算基础战斗奖励
        2. 在逃脱模式下处理距离相关的动态奖励
        3. 汇总所有奖励项并应用全局奖励共享机制
        """
        rews, destroyed_ids, _ = self._combat_rewards(events, opp_stats)

        # 处理逃脱模式的距离奖励机制
        #per-time-step escape reward
        if self.agent_mode == "escape" and self.args.esc_dist_rew:
            for i in range(1, self.args.num_agents+1):
                if self.sim.unit_exists(i):
                    u = self.sim.get_unit(i)
                    # considering all opps
                    opps = self._nearby_object(i)
                    # 根据与最近对手的距离调整奖励：距离过近惩罚，距离过远奖励
                    # 奖励值与距离排序成反比（最近对手j=1的系数最大）
                    for j, o in enumerate(opps, start=1):
                        #o[2] is unnormalized distance
                        #opps is ordered by distance, so we scale rewards by closest opponent j
                        if o[2] < 0.06:
                            rews[i].append(-0.02/j)
                            if u.speed < 200:
                                rews[i].append(-0.02/j)
                        elif o[2] > 0.13:
                            rews[i].append(0.02/j)
                            if u.speed > 500:
                                rews[i].append(0.02/j)
                
        # 奖励汇总与共享机制
        #sum all collected rewards together
        for i in range(1, self.args.num_agents+1):
            if self.sim.unit_exists(i) or i in destroyed_ids:
                # 在战斗模式下应用全局奖励共享（适用于2v2训练场景）
                if self.args.glob_frac > 0 and self.agent_mode == "fight":
                    #shared reward function, defined for 2-vs-2 training in fight mode
                    rewards[i] += sum(rews[i]) + self.args.glob_frac*sum(rews[i%2+1])
                else:
                    rewards[i] += sum(rews[i])

        return rewards
    
    def _escaping_opp(self, unit):
        """
        This ensures that the hardcoded opponents don't stuck in rotating around agents. 
        So, opponents shortly escape in the diagonal direction.
        """
        """
        使硬编码的对手单位避免绕智能体旋转，通过向对角线方向短暂移动
        
        Args:
            unit: Unit 对象，表示需要进行逃逸操作的对手单位
            
        Returns:
            tuple: 包含六个元素的元组
                - 第一个元素保留位(始终为None)
                - heading: int 型，新的目标航向角度(0-359度)
                - speed: int 型，新的移动速度(300-600单位)
                - 随机布尔值，用于控制状态切换
                - 固定False值保留位
                - 航向调整角度，限制在[-180,180]范围的整数值
        """
        # 获取单位在二维坐标系中的归一化相对位置
        y, x = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        # 根据相对位置选择四个对角区域生成航向角
        if y < 0.5:
            # 左下或右下区域
            if x < 0.5:
                # 东北偏东方向
                heading = int(random.uniform(30, 60))
            else:
                # 西北偏西方向
                heading = int(random.uniform(300, 330))
        else:
            # 左上或右上区域
            if x < 0.5:
                # 东南偏东方向
                heading = int(random.uniform(120, 150))
            else:
                # 西南偏西方向
                heading = int(random.uniform(210, 240))
        # 生成随机速度值
        speed= int(random.uniform(300, 600))
        # 计算航向调整方向符号
        sign = np.sign(heading-unit.heading+1e-9)
        # 返回包含运动参数的元组
        return None, heading, speed, bool(random.randint(0,1)), False, np.clip(((heading-unit.heading)%360)*sign, -180, 180)

    def _hardcoded_opp(self, opp_unit, opp_id):
        """
        Deterministic opponents to fight against agents. They have slight randomness in heading and speed. 
        """
        """
        处理确定性对手的行为逻辑，包含航向和速度的随机性控制

        参数:
            opp_unit: 敌方单位对象，包含当前状态信息（如航向、飞机类型等）
            opp_id: 敌方单位的唯一标识符

        返回值:
            tuple: 包含六个元素的元组(
                最近友方单位ID: str | None,  # 最近可交互友方单位的标识
                新航向角度: float,          # 计算后的新航向角度（0-360度）
                调整后速度: int,            # 经过随机处理和限制后的移动速度
                机炮开火标记: bool,         # 是否满足机炮开火条件
                导弹发射标记: bool,         # 是否满足导弹发射条件
                相对航向调整量: float       # 本次计算的航向调整量（带方向）
            )
        """
        # 获取最近的友方单位信息（距离，单位ID）
        d_agt = self._nearby_object(opp_id)
        heading = opp_unit.heading
        fire = False
        fire_missile = False
        # 基础速度随机值
        speed = int(random.uniform(100, 400))
        # 航向调整量初始化
        head_rel = 0 #heading degree towards opp
        if d_agt:
            # 计算角度修正方向符号（正负号）
            sign = self._correct_angle_sign(opp_unit, self.sim.get_unit(d_agt[0][0]))
            # 随机系数增强行为不可预测性
            r = random.uniform(0.7, 1.3)
            focus = self._focus_angle(opp_id, d_agt[0][0])
            # 动态调整航向逻辑（仅在有效距离和聚焦角度时触发）
            if d_agt[0][1] > 0.008 and focus > 4:
                # 带随机性的相对航向调整量
                head_rel = r*sign*focus
                # 计算新航向
                heading = (heading + r*sign*focus)%360
            # 距离相关速度调节逻辑
            if d_agt[0][1] > 0.05:
                speed = int(random.uniform(500, 800)) if focus < 30 else int(random.uniform(100, 500))
            # 武器系统触发条件判断
            # 近距离小角度触发机炮
            fire = d_agt[0][1] < 0.03 and focus < 10
            # 中距离更严格角度触发导弹
            fire_missile = d_agt[0][1] < 0.09 and focus < 5 # ROCKET
            # 特定机型速度限制
            speed = np.clip(speed, 0, 600) if opp_unit.ac_type == 2 else speed
            return d_agt[0][0], heading, speed, fire, fire_missile, head_rel
        # 无有效目标时的默认处理
        speed = np.clip(speed, 0, 600) if opp_unit.ac_type == 2 else speed
        return None, heading, speed, fire, fire_missile, head_rel