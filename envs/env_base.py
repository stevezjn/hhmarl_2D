
import torch
import os
import random
import numpy as np
from pathlib import Path
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from math import sin, cos, acos, pi, hypot, radians, exp, sqrt
from warsim.scenplotter.scenario_plotter import PlotConfig, ColorRGBA, StatusMessage, TopLeftMessage, \
    Airplane, PolyLine, Drawable, Waypoint, Missile, ScenarioPlotter
from warsim.simulator.cmano_simulator import Position, CmanoSimulator, UnitDestroyedEvent
from warsim.simulator.ac1 import Rafale
from warsim.simulator.ac2 import RafaleLong
from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits
from utils.angles import sum_angles

colors = {
    'red_outline': ColorRGBA(0.8, 0.2, 0.2, 1),
    'red_fill': ColorRGBA(0.8, 0.2, 0.2, 0.2),
    'blue_outline': ColorRGBA(0.3, 0.6, 0.9, 1),
    'blue_fill': ColorRGBA(0.3, 0.6, 0.9, 0.2),
    'waypoint_outline': ColorRGBA(0.8, 0.8, 0.2, 1),
    'waypoint_fill': ColorRGBA(0.8, 0.8, 0.2, 0.2)
}

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

class HHMARLBaseEnv(MultiAgentEnv):
    """
    Base class for HHMARL 2D with core functionalities. 
    """
    """
    HHMARL 2D 环境的基类，实现多智能体协同作战的核心功能。
    管理智能体与对手的交互、战斗逻辑、奖励计算及环境状态更新。
    """
    def __init__(self, map_size):
        """初始化环境模拟器的实例
        
        Args:
            map_size (float): 地图的基础尺寸参数，与map_limits共同决定
                模拟环境的边界范围。单位通常为米或标准单位长度
        """
        # 环境控制参数初始化
        # 禁用环境检查标志
        self._skip_env_checking = True
        # 步数计数器
        self.steps = 0
        # 仿真引擎占位符
        self.sim = None
        # 地图边界配置
        self.map_size = map_size
        self.map_limits = MapLimits(7.0, 5.0, 7.0+map_size, 5.0+map_size)

        # 单位状态管理
        # 存活友方单位计数器
        self.alive_agents = 0
        # 存活敌方单位计数器
        self.alive_opps = 0
        # 奖励值存储字典
        self.rewards = {}

        # 战斗行为控制参数
        # needed for combat behavior
        # 敌方攻击目标映射表
        self.opp_to_attack = {}
        # 导弹发射冷却计时器
        self.missile_wait = {}
        # 敌方强制撤退标志
        self.hardcoded_opps_escaping = False
        # 敌方撤退持续时间
        self.opps_escaping_time = 0

        # 可视化配置
        # Plotting
        self.plt_cfg = PlotConfig()
        # 单位绘制缩放比例
        self.plt_cfg.units_scale = 20.0
        self.plotter = ScenarioPlotter(self.map_limits, dpi=200, config=self.plt_cfg)

        super().__init__()

    def reset(self, *, seed=None, options=None):
        """
        Reset scenario to a new random configuration.
        """
        """
        重置场景到新的随机配置

        Args:
            seed (int, optional): 随机种子（当前实现未使用）
            options (dict, optional): 包含重置参数的配置字典
                - mode: 场景模式配置（默认None）

        Returns:
            None

        Notes:
            - 重置所有场景计数器、智能体状态标志
            - 初始化导弹冷却时间和攻击目标字典
            - 创建新的仿真器实例并重置场景配置
        """
        # 重置基础计数器和状态标志
        self.steps = 0
        self.alive_agents = 0
        self.alive_opps = 0

        # 重置对手行为追踪状态
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        # 初始化武器冷却和攻击目标系统
        self.missile_wait = {i:0 for i in range(1, self.args.total_num+1)}
        self.opp_to_attack = {i:None for i in range(1, self.args.total_num+1)}

        # 创建新的仿真环境实例
        self.sim = CmanoSimulator(num_units=self.args.num_agents, num_opp_units=self.args.num_opps)
        self._reset_scenario(options.get("mode", None))
        return
    
    def step(self, action):
        """
        Take one step for all agents in env.

        In High-Level mode, actions will be modified in take_action() to include actions for opponents.
        Since actions is a dict, the reference will be passed and modified. 
        """
        """
        执行环境中所有智能体的单步动作

        Args:
            action (dict): 包含各智能体动作指令的字典。在高层控制模式下，
                该参数会在_take_action()方法中被修改以包含对手动作
                
        Returns:
            tuple: 包含五个元素的元组：
                - 当前环境状态
                - 奖励值字典
                - 终止状态字典
                - 截断状态字典
                - 包含评估信息的字典（仅在eval_info启用时包含详细统计）

        在High-Level模式下，actions会通过引用传递并在take_action()中被修改
        """
        # 初始化奖励字典
        self.rewards = {}
        if action:
            # 执行实际动作处理逻辑
            self._take_action(action)
        # 初始化终止/截断状态字典
        terminateds = truncateds = {}
        # 设置全局终止条件：存活单位归零或达到步数限制
        truncateds["__all__"] = terminateds["__all__"] = self.alive_agents <= 0 or self.alive_opps <= 0 or self.steps >= self.args.horizon
        # 收集评估信息（仅在eval_info启用时）
        if self.args.eval_info:
            # 战斗行为统计计数器初始化
            af = ae = of = oe = ast = ost = 0
            # 对手选择统计字典
            opps_selection = {"opp1":0, "opp2":0, "opp3":0}
            # 遍历所有动作进行统计
            for k, v in action.items():
                if self.sim.unit_exists(k):
                    if v:
                        # 统计战斗/逃跑行为
                        if k <= self.args.num_agents: 
                            af+=1; ast+=1; opps_selection[f"opp{v}"]+=1
                        else: of+=1; ost+=1
                    else:
                        # 统计静默行为
                        if k <= self.args.num_agents: ae+=1; ast+=1
                        else: oe += 1; ost+=1

            # 构建评估信息字典
            info = {"agents_win": int(self.alive_opps<=0 and self.steps<self.args.horizon), "opps_win": int(self.alive_agents<=0 and self.steps<self.args.horizon), "draw": int(self.steps>=self.args.horizon and self.alive_agents>0 and self.alive_opps>0), \
             "agent_fight": af, "agent_escape":ae, "opp_fight":of, "opp_escape":oe, "agent_steps":ast, "opp_steps":ost}
            info.update(opps_selection)
        else: info = {}

        return self.state(), self.rewards, terminateds, truncateds, info
    
    def fight_state_values(self, agent_id, unit, opp, fri_id=None):
        """
        Fill the observation values for fight mode in low-level scene.
        opp = [opp_id, opp_dist]
        """
        """
        生成战斗模式下低级场景的观测状态向量

        参数:
            agent_id (int): 当前智能体的唯一标识符
            unit (Unit): 当前智能体对应的作战单位对象
            opp (list): 对手信息列表 [对手单位ID, 与对手的距离]
            fri_id (int, optional): 友方单位标识符，默认为None

        返回值:
            list: 包含战斗状态特征值的归一化向量，元素范围在[0,1]或标准化数值
        
        观测特征包含：
        - 单位位置坐标
        - 运动状态（速度、航向）
        - 武器系统状态
        - 与对手单位的相对状态
        - 友方单位状态
        """
        state = []
        # 位置特征处理
        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.append(x)
        state.append(y)
        # 运动状态特征归一化
        state.append(np.clip(unit.speed/unit.max_speed,0,1))
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        # 对抗关系特征计算
        # 获取聚焦角度
        state.append(self._focus_angle(agent_id, opp[0],True))
        # 计算方位角
        state.append(self._aspect_angle(opp[0], agent_id))
        # 航向差计算
        state.append(self._heading_diff(agent_id, opp[0]))
        # 对手距离
        state.append(opp[1])
        # 武器系统状态处理
        state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
        if unit.ac_type == 1:
            # 导弹相关状态
            state.append(np.clip(unit.missile_remain/unit.rocket_max, 0, 1))
            # 导弹发射就绪状态
            state.append(int(self.missile_wait[agent_id]==0))
            state.append(int(bool(unit.actual_missile)) or int(unit.cannon_current_burst_secs > 0))
        else:
            # 机炮状态
            state.append(int(unit.cannon_current_burst_secs > 0))
        # 添加对抗单位和友方单位的扩展特征
        state.extend(self.opp_ac_values("fight", opp[0], agent_id, opp[1]))
        state.extend(self.friendly_ac_values(agent_id, fri_id))
        return state

    def esc_state_values(self, agent_id, unit, opps, fri_id=None):
        """
        Fill the observation values for escape mode in low-level scene.
        opps = [[opp_id1, opp_dist1], [opp_id2, opp_dist2], ...]
        """
        """
        生成逃生模式下的智能体状态观测向量
        
        Args:
            agent_id (int): 当前代理的唯一标识符
            unit (Unit): 当前代理控制的单位对象
            opps (list): 敌对单位信息列表，格式为[[敌方ID1, 距离1], [敌方ID2, 距离2], ...]
            fri_id (int, optional): 关联友军单位ID，默认为None
            
        Returns:
            list: 包含43个元素的归一化状态向量，结构为：
                [自身状态(5-6元素), 射击状态(1), 敌方状态(18), 友方状态(19)]
        """
        # 初始化基础状态容器
        state = []
        # 添加单位空间状态信息
        x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        # 归一化横向坐标
        state.append(x)
        # 归一化纵向坐标
        state.append(y)
        # 添加运动及武器系统状态
        # 速度百分比
        state.append(np.clip(unit.speed/unit.max_speed,0,1))
        # 航向归一化
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        # 主炮充能
        state.append(np.clip(unit.cannon_remain_secs/unit.cannon_max, 0, 1))
        # 特殊机型携带导弹时的状态
        if unit.ac_type==1:
            # 导弹剩余量
            state.append(np.clip(unit.missile_remain/unit.rocket_max, 0, 1))
        # 武器发射状态检测（主炮/导弹）
        shot = unit.cannon_current_burst_secs > 0
        if unit.ac_type==1:
            shot = shot or bool(unit.actual_missile)
        state.append(int(shot))
        # 处理敌对单位状态信息（最多处理到填满18个元素）
        opp_state = []
        for opp in opps:
            opp_state.extend(self.opp_ac_values("esc", opp[0], agent_id, opp[1]))
            # 达到状态容量上限时停止处理
            if len(opp_state) == 18:
                break
        # 敌对单位信息不足时填充零值
        if len(opp_state) < 18:
            opp_state.extend(np.zeros(18-len(opp_state)))
        # 整合友军单位状态信息
        state.extend(opp_state)
        state.extend(self.friendly_ac_values(agent_id, fri_id))
        return state

    def friendly_ac_values(self, agent_id, fri_id=None):
        """
        State of friendly aircraft w.r.t. agent or opp.
        """
        """        
        获取友好航空器相对于当前智能体或对手的状态信息

        参数:
        agent_id (int): 当前智能体单位的唯一标识符
        fri_id (int, optional): 目标友好单位的标识符，默认为None表示无指定目标

        返回:
        np.ndarray: 包含5个元素的状态向量，依次表示：
            [x坐标偏移量, y坐标偏移量, 
             当前智能体指向友好单位的角度, 
             友好单位指向当前智能体的角度, 
             两者间距离]
            若目标不存在则返回零向量
        """
        if not fri_id:
            return np.zeros(5)
        elif self.sim.unit_exists(fri_id):
            # 获取友好单位实例并构建状态向量
            unit = self.sim.get_unit(fri_id)
            state = []
            # 计算相对坐标偏移量
            x,y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            state.append(x)
            state.append(y)
            # 计算双向视角关系及距离
            state.append(self._focus_angle(agent_id, fri_id, True))
            state.append(self._focus_angle(fri_id, agent_id, True))
            state.append(self._distance(agent_id, fri_id, True))
            return state
        else:
            return np.zeros(5)
        
    def opp_ac_values(self, mode, opp_id, agent_id, dist):
        """
        State of opponent aircraft w.r.t. agent or opp.
        """
        """
        获取对手飞机相对于指定智能体/对手的状态特征向量
        
        参数:
            mode (str): 模式标识，可选"fight"/"HighLevel"等，决定特征组合方式
            opp_id (int): 对手飞机的唯一标识符
            agent_id (int): 主智能体飞机的唯一标识符  
            dist (float): 两机之间的当前距离
        
        返回值:
            list: 包含14-16个特征值的状态向量，具体元素根据模式不同而变化：
                  [x坐标, y坐标, 归一化速度, 归一化航向, 航向差, 焦点角...]
        """
        state = []
        # 获取对手单位对象并计算相对坐标
        unit = self.sim.get_unit(opp_id)
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        # 添加基础空间特征
        # 经度方向的归一化相对位置
        state.append(x)
        # 纬度方向的归一化相对位置
        state.append(y)
        # 速度占最大速度的比例
        state.append(np.clip(unit.speed/unit.max_speed,0,1))
        # 航向角归一化
        state.append(np.clip((unit.heading%359)/359, 0, 1))
        # 两机航向差
        state.append(self._heading_diff(opp_id, agent_id))
        # 根据模式添加角度特征
        if mode == "fight":
            # 战斗模式下的焦点角和方位角
            # 对手对主机的焦点角
            state.append(self._focus_angle(opp_id, agent_id,True))
            # 主机对对手的方位角
            state.append(self._aspect_angle(agent_id, opp_id))
        else:
            # 非战斗模式的双向焦点角
            # 主机对对手的焦点角
            state.append(self._focus_angle(agent_id, opp_id, True))
            # 对手对主机的焦点角
            state.append(self._focus_angle(opp_id, agent_id, True))
        # HighLevel模式特有特征
        if mode == "HighLevel":
            # 主机对对手的方位角
            state.append(self._aspect_angle(agent_id, opp_id))
            # 对手对主机的方位角
            state.append(self._aspect_angle(opp_id, agent_id))
        # 添加原始距离值
        state.append(dist)
        # 武器状态检测（非HighLevel模式）
        if mode != "HighLevel":
            # 判断是否正在使用机炮
            shot = unit.cannon_current_burst_secs > 0
            # 如果是战斗机类型，检测导弹发射状态
            if unit.ac_type==1:
                shot = shot or bool(unit.actual_missile)
            # 武器使用状态(0/1)
            state.append(int(shot))
        return state

    def _take_base_action(self, mode, unit, unit_id, opp_id, actions, rewards=None):
        """
        Take basic combat step in env (and assign some rewards for escape).
        """
        """
        执行基本战斗动作并处理相关奖励（特别是逃跑行为的奖励分配）
        
        Args:
            mode (str): 当前环境模式，可能值为"LowLevel"或其他模式标识
            unit (Unit): 当前操作的单位对象实例
            unit_id (int): 当前单位的唯一标识符
            opp_id (int): 敌对单位的唯一标识符
            actions (dict): 动作指令字典，包含各单位的动作参数
            rewards (list/dict, optional): 奖励值容器，默认为None时可能初始化新容器

        Returns:
            list/dict: 更新后的奖励值容器，包含当前单位获得的奖励调整值

        Note:
            - 同时处理航向/速度调整、火炮射击、导弹发射三种基础战斗行为
            - 在LowLevel模式下对特定单位的奖励机制进行特殊处理
        """
        # 航向和速度调整模块
        unit.set_heading((unit.heading + (actions[unit_id][0]-6)*15)%360) #relative heading
        unit.set_speed(100+((unit.max_speed-100)/8)*actions[unit_id][1]) #absolute speed

        # 火炮射击逻辑处理
        if bool(actions[unit_id][2]) and unit.cannon_remain_secs > 0:
            unit.fire_cannon()
            if mode=="LowLevel" and unit_id <= self.args.num_agents:
                if self.agent_mode == "escape" and unit.cannon_remain_secs < 90:
                    rewards[unit_id] -= 0.1

        # 导弹发射逻辑处理（仅限特定单位类型）
        if unit.ac_type == 1 and bool(actions[unit_id][3]):
            if opp_id and unit.missile_remain > 0 and not unit.actual_missile and self.missile_wait[unit_id] == 0:
                unit.fire_missile(unit, self.sim.get_unit(opp_id), self.sim)
                self.missile_wait[unit_id] = random.randint(7,17) if mode == "LowLevel" else random.randint(8, 12)
                if mode=="LowLevel" and unit_id <= self.args.num_agents:
                    if self.agent_mode == "escape" and unit.missile_remain < 3:
                        rewards[unit_id] -= 0.1

        # 导弹冷却时间更新
        if self.missile_wait[unit_id] > 0 and not bool(unit.actual_missile):
            self.missile_wait[unit_id] = self.missile_wait[unit_id] -1

        return rewards

    def _combat_rewards(self, events, opp_stats=None, mode="LowLevel"):
        """"
        Calculating Rewards. 
        First check for out-of-boundary, then killing rewards.
        """
        """
        计算战斗奖励系统

        参数:
            events (list): 当前时间步发生的战斗事件列表
            opp_stats (dict, optional): 对手状态信息字典，包含各敌方单位的属性统计
            mode (str): 训练模式，可选"LowLevel"或"High-Level"

        返回值:
            tuple: 包含三个元素的元组
                - rews (dict): 智能体ID到奖励值列表的映射字典
                - destroyed_ids (list): 本回合被摧毁的单位ID列表
                - kill_event (bool): 是否发生有效击杀事件的标志

        处理逻辑：
        1. 边界越界惩罚阶段
        2. 战斗事件奖励计算阶段
        """
        rews = {a:[] for a in range(1,self.args.num_agents+1)}
        destroyed_ids = []
        s=self.args.rew_scale
        kill_event = False

        # 处理越界单位惩罚
        # 检查所有单位位置，移除越界单位并施加相应惩罚
        #out-of-boundary punishment
        for i in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if not self.map_limits.in_boundary(u.position.lat, u.position.lon):
                    self.sim.remove_unit(i)
                    kill_event = True
                    if i <= self.args.num_agents:
                        p = -5 if mode=="LowLevel" else -2
                        rews[i].append(p*s)
                        destroyed_ids.append(i)
                        self.alive_agents -= 1
                    else:
                        self.alive_opps -= 1

        # 处理战斗事件奖励
        #event rewards
        for ev in events:

            # 处理己方单位击杀行为
            # agent kill
            if ev.unit_killer.id <= self.args.num_agents:

                # 敌方单位被摧毁的奖励计算
                #killed opp
                if ev.unit_destroyed.id in range(self.args.num_agents+1, self.args.total_num+1):
                    if mode=="LowLevel":
                        if self.agent_mode == "fight":
                            # 基于剩余弹药量的动态奖励计算
                            if ev.origin.id >= self.args.total_num+1: #killed by rocket
                                rews[ev.unit_killer.id].append(self._shifted_range(ev.unit_killer.missile_remain/ev.unit_killer.rocket_max, 0,1, 1,1.5)*s)
                            else:
                                rews[ev.unit_killer.id].append((self._shifted_range(ev.unit_killer.cannon_remain_secs/ev.unit_killer.cannon_max, 0,1, 0.5,1) + self._shifted_range(opp_stats[ev.unit_killer.id][0], 0,1, 0.5,1))*s)
                        else:
                            # no reward for killing in escape mode
                            pass
                            #rews[ev.unit_killer.id].append(1)
                    else:
                        #constant reward for killing in High-Level Env
                        rews[ev.unit_killer.id].append(1)

                    self.alive_opps -= 1

                # 友军误伤惩罚处理
                #friendly kill
                elif ev.unit_destroyed.id <= self.args.num_agents:
                    if mode=="LowLevel":
                        rews[ev.unit_killer.id].append(-2*s)
                        if self.args.friendly_punish:
                            rews[ev.unit_destroyed.id].append(-2*s)
                            destroyed_ids.append(ev.unit_destroyed.id)
                    self.alive_agents -= 1

            # 处理敌方单位击杀行为
            # opp kill
            elif ev.unit_killer.id in range(self.args.num_agents+1, self.args.total_num+1):
                # 己方单位被摧毁的惩罚
                if ev.unit_destroyed.id <= self.args.num_agents:
                    p = -2 if mode=="LowLevel" else -1
                    rews[ev.unit_destroyed.id].append(p*s)
                    destroyed_ids.append(ev.unit_destroyed.id)
                    self.alive_agents -= 1
                # 敌方内部击杀处理
                elif ev.unit_destroyed.id in range(self.args.num_agents+1, self.args.total_num+1):
                    self.alive_opps -= 1

            kill_event = True

        return rews, destroyed_ids, kill_event

    def _get_policies(self, mode):
        """
        Restore torch policies for fictitious self-play.
        """
        """
        加载不同层级的策略模型用于虚构自对弈训练/评估
        
        参数:
            mode (str): 策略加载模式
                "LowLevel" - 仅加载底层动作策略（战斗/逃跑）
                其他模式 - 加载包含对手策略的完整策略集
            
        功能逻辑:
            1. 根据运行模式和环境参数，从预训练模型目录加载对应的策略文件
            2. 处理不同对抗层级的策略组合：
                - 战斗策略（fight）根据关卡等级加载
                - 逃跑策略（escape）默认加载L3版本，存在高阶对抗时加载L5版本
            3. 异常处理确保策略文件缺失时的回退机制
            
        策略文件命名规则:
            L{level}_AC{agent_num}_{policy_type}.pt
            示例: L3_AC1_fight.pt 表示3级关卡1号智能体的战斗策略
            
        注意:
            - policy_dir 路径为项目根目录下的policies文件夹
            - 当eval_hl=False时需额外加载对手策略形成对抗
        """
        # 构建策略文件路径（项目根目录/policies/）
        policy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'policies')
        # 初始化策略容器
        self.policy = {}
        # 底层策略加载模式
        if mode == "LowLevel":
            # 战斗模式策略加载
            if self.args.agent_mode == "fight":
                # 特殊处理4级关卡策略
                if self.args.level == 4:
                    self.policy = {"fight_1": torch.load(os.path.join(policy_dir, 'L3_AC1_fight.pt')), "fight_2": torch.load(os.path.join(policy_dir, 'L3_AC2_fight.pt'))}
                # 3-5级关卡策略加载
                else:
                    self.policies = {}
                    for i in range(3,6):
                        # 3-4级加载战斗策略
                        if i <= 4:
                            self.policies[i] = {"fight_1": torch.load(os.path.join(policy_dir, f'L{i}_AC1_fight.pt')), "fight_2": torch.load(os.path.join(policy_dir, f'L{i}_AC2_fight.pt'))}
                        # 5级加载逃跑策略
                        else:
                            self.policies[i] = {"escape_1": torch.load(os.path.join(policy_dir, 'L3_AC1_escape.pt')), "escape_2": torch.load(os.path.join(policy_dir, 'L3_AC2_escape.pt'))}
            else: #escape-vs-L5_fight
                # 5级对手战斗策略
                if self.args.level==5:
                    self.policy = {"fight_1": torch.load(os.path.join(policy_dir, 'L5_AC1_fight.pt')), "fight_2": torch.load(os.path.join(policy_dir, 'L5_AC2_fight.pt'))}
        # 高层策略加载模式
        else:
            # 基础战斗策略加载
            self.policy["fight_1"] = torch.load(os.path.join(policy_dir, f'L{self.args.eval_level_ag}_AC1_fight.pt'))
            self.policy["fight_2"] = torch.load(os.path.join(policy_dir, f'L{self.args.eval_level_ag}_AC2_fight.pt'))
            # 逃跑策略加载（优先尝试L5版本）
            try:
                self.policy["escape_1"] = torch.load(os.path.join(policy_dir, f'L5_AC1_escape.pt'))
                self.policy["escape_2"] = torch.load(os.path.join(policy_dir, f'L5_AC2_escape.pt'))
            except:
                # 回退机制：当缺少L5对抗策略时使用L3版本
                # if escape was not trained against L5_fight
                self.policy["escape_1"] = torch.load(os.path.join(policy_dir, f'L3_AC1_escape.pt'))
                self.policy["escape_2"] = torch.load(os.path.join(policy_dir, f'L3_AC2_escape.pt'))

            # 非高层评估时加载对手策略
            if not self.args.eval_hl:
                # if evaluating purely low-level policies without commander
                self.policy["fight_1_opp"] = torch.load(os.path.join(policy_dir, f'L{self.args.eval_level_opp}_AC1_fight.pt'))
                self.policy["fight_2_opp"] = torch.load(os.path.join(policy_dir, f'L{self.args.eval_level_opp}_AC2_fight.pt'))
        return

    def _policy_actions(self, policy_type, agent_id, unit):
        """
        Apply self-play actions from previous, learned policy.
        We call method get_torch_action() to compute actions manually, 
        because there is an inconsistent behaviour when calling Policy.compute_single_action() from Ray (version 2.4)
        """
        """
        应用自对弈策略生成智能体动作
        
        Args:
            policy_type (str): 策略类型，可选战斗(fight)或逃跑(escape)
            agent_id (int): 智能体标识符
            unit (object): 作战单元对象，需包含ac_type属性标识单元类型
            
        Returns:
            dict: 包含智能体ID到动作映射的字典

        Note:
            使用手动计算的torch action替代Ray 2.4版本Policy.compute_single_action()方法，
            因原方法存在不一致行为问题
        """
        actions = {}

        # 定义观测张量生成函数（根据单元类型构建不同的观测结构）
        def obs_tens(obs):
            if unit.ac_type == 1:
                return {
                    "obs_1_own": torch.tensor(obs),
                    "obs_2": torch.zeros((1,OBS_AC2 if policy_type=="fight" else OBS_ESC_AC2)),
                    "act_1_own": torch.zeros((1,ACTION_DIM_AC1)),
                    "act_2": torch.zeros((1,ACTION_DIM_AC2)),
                }
            else:
                return {
                    "obs_1_own": torch.tensor(obs),
                    "obs_2": torch.zeros((1,OBS_AC1 if policy_type=="fight" else OBS_ESC_AC1)),
                    "act_1_own": torch.zeros((1,ACTION_DIM_AC2)),
                    "act_2": torch.zeros((1,ACTION_DIM_AC1)),
                }

        # 基于分类分布的手动动作计算方法
        def get_torch_action(inp):
            """
            Manual computation of action based on Categorical distribution.
            """
            """
            通过Categorical分布手动计算动作
            
            Args:
                inp (Tensor): 策略网络输出张量
                
            Returns:
                ndarray: 计算得到的动作数组
            """
            in_lens = np.array([13,9,2,2]) if unit.ac_type == 1 else np.array([13,9,2])
            inputs_split = inp.split(tuple(in_lens), dim=1)
            cats = [torch.distributions.categorical.Categorical(logits=input_) for input_ in inputs_split]
            arr = [torch.argmax(cat.probs, -1) for cat in cats]
            action = torch.stack(arr, dim=1)
            return np.array(action[0])

        # 生成底层状态观测
        state = self.lowlevel_state(policy_type, agent_id, unit=unit)

        # 策略网络推理流程
        with torch.no_grad(): 
            # 构建策略标识符（区分主方和对手策略）
            if self.args.eval_hl or policy_type == "escape":
                policy_str = f"{policy_type}_{unit.ac_type}"
            else:
                policy_str = f"{policy_type}_{unit.ac_type}" if agent_id <= self.args.num_agents else f"{policy_type}_{unit.ac_type}_opp"
            
            # 执行策略网络前向计算
            out = self.policy[policy_str](
                input_dict = {"obs": obs_tens(np.expand_dims(state[agent_id], axis=0))},
                state=[torch.tensor(0)],
                seq_lens=torch.tensor([1])
            )
        # 存储并返回动作结果
        actions[agent_id] = get_torch_action(out[0])
        return actions

    def _nearby_object(self, agent_id, friendly=False):
        """
        Return a sorted list with id's and distances to opponents/friendly aircraft. 
        """
        """
        获取指定智能体附近的友方/敌方单位列表，按距离排序

        Args:
            agent_id (int): 当前智能体的唯一标识符
            friendly (bool, optional): 是否查找友方单位，默认为False查找敌方单位

        Returns:
            list: 排序后的单位列表，每个元素为[单位ID, 距离1, 距离2]或[单位ID, 距离]
                  当查找敌方时返回两个距离值，查找友方时返回一个距离值
                  列表按距离1（第二个元素）升序排列
        """
        order = []
        # 处理友方单位查找
        if friendly:
            # 确定友方单位ID范围：根据当前单位所属阵营（红/蓝方）
            f = list(range(1,self.args.num_agents+1)) if agent_id <= self.args.num_agents else list(range(self.args.num_agents+1, self.args.total_num+1))
            # 排除自身ID
            f.remove(agent_id)
            # 遍历有效友方单位并计算距离
            for i in f:
                if self.sim.unit_exists(i):
                    # 使用简化距离计算
                    order.append([i, self._distance(agent_id, i, True)])
        # 处理敌方单位查找
        else:
            # 确定敌方单位ID范围：根据当前单位所属阵营（红/蓝方）
            if agent_id <= self.args.num_agents:
                start = self.args.num_agents + 1
                end = self.args.total_num + 1
            else:
                start = 1
                end = self.args.num_agents + 1
            # 遍历有效敌方单位并计算两种距离
            for i in range(start, end):
                if self.sim.unit_exists(i):
                    # 两种距离计算方式
                    order.append([i, self._distance(agent_id, i, True), self._distance(agent_id, i)])
        # 按主要距离指标（列表第二个元素）进行升序排序
        order.sort(key=lambda x:x[1])
        return order

    def _focus_angle(self, agent_id, opp_id, norm=False):
        """
        Compute ATA angle based on vector angles of current heading direction and position of the two airplanes. 
        """
        """
        计算两架飞机之间的航向夹角（ATA，Aspect Angle）

        参数:
            agent_id (int): 主飞机的唯一标识符
            opp_id (int): 目标飞机的唯一标识符
            norm (bool): 是否对角度进行归一化处理，默认False返回实际角度值

        返回:
            float: 角度值（当norm=True时返回归一化到[0,1]区间的值，否则返回实际角度值0-180）

        实现逻辑:
        1. 通过向量点积计算两飞行器航向与相对位置之间的夹角余弦值
        2. 将数学角度转换为实际航向角（坐标系转换）
        3. 包含数值稳定性处理（防止除以零）
        """
        x = np.clip((np.dot(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ),sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]), np.array([self.sim.get_unit(opp_id).position.lon-self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat-self.sim.get_unit(agent_id).position.lat])))/(np.linalg.norm(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ),sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]))*np.linalg.norm(np.array([self.sim.get_unit(opp_id).position.lon-self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat-self.sim.get_unit(agent_id).position.lat]))+1e-10), -1, 1)
        if norm:
            return np.clip( (acos(x) * (180 / pi))/180, 0, 1)
        else:
            return acos(x) * (180 / pi)

    def _distance(self, agent_id, opp_id, norm=False):
        """
        Euclidian Distance between two aircrafts.
        """
        """
        计算两个飞行器之间的欧氏距离，并可选择进行归一化处理

        Args:
            agent_id (int/str): 主体飞行器的唯一标识符
            opp_id (int/str): 对方飞行器的唯一标识符
            norm (bool, optional): 是否返回归一化后的结果，默认False返回原始距离值

        Returns:
            float: 当norm=True时返回[0,1]区间的归一化距离值，
                   当norm=False时返回原始欧氏距离值
        
        实现说明：
        - 通过飞行器对象的经度(lon)和纬度(lat)坐标计算平面距离
        - 使用hypot函数计算直角三角形的斜边长度
        - 归一化时将原始距离从[0, sqrt(2*map_size²)]线性映射到[0,1]区间
        """
        d = hypot(self.sim.get_unit(opp_id).position.lon - self.sim.get_unit(agent_id).position.lon, self.sim.get_unit(opp_id).position.lat - self.sim.get_unit(agent_id).position.lat)
        return self._shifted_range(d, 0, sqrt(2*self.map_size**2), 0, 1) if norm else d

    def _aspect_angle(self, agent_id, opp_id, norm=True):
        """
        Aspect angle: angle from agent_id tail to opp_id, regardless of heading of opp_id.
        """
        """
        计算从指定智能体尾部到目标智能体的视角角（Aspect Angle）
        
        参数:
            agent_id (int): 观察主体的智能体标识符
            opp_id (int): 被观察目标的智能体标识符
            norm (bool, optional): 是否对结果进行归一化处理，默认为True
        
        返回:
            float: 当norm=True时返回[0,1]范围的归一化角度值
                   norm=False时返回[0,180]范围的实际角度值
        
        说明:
        该角度表示从主体智能体尾部方向到目标智能体位置的视线夹角，
        与目标智能体的朝向无关。通过焦点角计算后取补角得到。
        """
        focus = self._focus_angle(agent_id, opp_id)
        return np.clip((180 - focus)/180,0,1) if norm else np.clip(180-focus,0,180)

    def _heading_diff(self, agent_id, opp_id, norm=True):
        """
        Angle between heading vectors.
        """
        """
        计算两个单位航向向量之间的夹角。

        参数:
            agent_id (int/str): 主单位标识符
            opp_id (int/str): 对比单位标识符
            norm (bool, 可选): 是否对角度进行归一化处理，默认为True

        返回:
            float: 夹角值，当norm为True时返回[0,1]区间的归一化值，否则返回角度值(0-180)

        实现逻辑:
            1. 将航向角转换为二维单位向量
            2. 计算向量点积并做数值裁剪
            3. 通过反余弦计算原始角度
            4. 根据norm参数决定是否归一化
        """
        # 将两个单位的航向角转换为笛卡尔坐标系单位向量，并计算余弦相似度
        x = np.clip((np.dot(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ), sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]), np.array([cos( ((90-self.sim.get_unit(opp_id).heading)%360)*(pi/180) ), sin( ((90-self.sim.get_unit(opp_id).heading)%360)*(pi/180) )])))/(np.linalg.norm(np.array([cos( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) ),sin( ((90-self.sim.get_unit(agent_id).heading)%360)*(pi/180) )]))*np.linalg.norm(np.array([cos( ((90-self.sim.get_unit(opp_id).heading)%360)*(pi/180) ), sin( ((90-self.sim.get_unit(opp_id).heading)%360)*(pi/180) )]))+1e-10), -1, 1)
        # 将弧度转换为角度后，根据norm参数决定是否进行归一化处理
        if norm:
            return np.clip( (acos(x) * (180 / pi))/180, 0, 1)
        else:
            return acos(x) * (180 / pi)

    def _shifted_range(self, x, a,b, c,d):
        """
        find value in new range from [a,b] to [c,d]
        """
        """
        将原始范围[a, b]中的值线性映射到新范围[c, d]
        
        实现公式：c + ((d - c) / (b - a)) * (x - a)
        当x等于a时对应c，x等于b时对应d，中间值按线性比例映射

        :param x: 需要进行范围映射的原始数值
        :param a: 原始范围的下限值
        :param b: 原始范围的上限值
        :param c: 目标范围的下限值
        :param d: 目标范围的上限值
        :return: 映射到新范围[c, d]后的数值
        """
        return c + ((d-c)/(b-a))*(x-a)

    def _correct_angle_sign(self, opp_unit, ag_unit):
        """
        The correct heading is computed in 'hardcoded_opps'. 
        Here the correct direction is determined, if to turn right or left. 
        """
        """
        通过向量叉积判断目标方位，确定最优转向方向（顺时针/逆时针）
        
        参数:
            opp_unit (Unit): 对抗单位对象，包含位置(lon,lat)和航向(heading)
            ag_unit (Unit/None): 待修正方向的智能体单位对象，None表示无目标时的默认转向
            
        返回:
            int: 转向指令 
                1  - 向右/顺时针方向修正
                -1 - 向左/逆时针方向修正

        核心算法:
            1. 构建对抗单位航向向量：
               - 基于opp_unit当前位置(x,y)和航向角a，计算单位方向向量终点(x1,y1)
               - 公式推导：x1 = x + sin(a%360), y1 = y + cos(a%360)
               
            2. 三点定位判断：
               使用向量叉积公式判断ag_unit位于对抗单位航向向量的哪一侧：
               line_val = (x1−x0)(y2−y0) − (x2−x0)(y1−y0)
               其中(x0,y0)=opp位置，(x1,y1)=航向终点，(x2,y2)=ag_unit位置
               
            3. 方向决策：
               - 当line_val < 0 时，目标点位于向量右侧，返回右转指令
               - 当line_val > 0 时，目标点位于向量左侧，返回左转指令
               - 无ag_unit时默认右转（战场安全撤退策略）

        几何原理:
            基于右手定则的向量叉积判断，当line_val的符号决定点在向量方向的左右关系：
            - 正值为逆时针侧（左侧）
            - 负值为顺时针侧（右侧）
        """
        def line(x0, y0, x1, y1, x2, y2):
            """计算点(x2,y2)相对于向量(x0,y0)->(x1,y1)的位置关系

            Args:
                x0/y0: 起点坐标
                x1/y1: 航向向量终点坐标
                x2/y2: 待判断点坐标

            Returns:
                float: 正值表示点在向量左侧，负值表示右侧，0表示共线
            """
            return  (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)

        # 获取敌方单位坐标和航向角度
        x = opp_unit.position.lon
        y = opp_unit.position.lat
        a = opp_unit.heading
        # 计算航向延长线参考点（将角度转换为三角函数值）
        x1 = x + round(sin(radians(a%360)),3)
        y1 = y + round(cos(radians(a%360)),3)

        # 当存在友方单位时进行位置关系判断
        if ag_unit:
            xc = ag_unit.position.lon
            yc = ag_unit.position.lat
            # 通过向量叉积判断友方单位相对航向的位置
            val = line(x,y,x1,y1,xc,yc)
            # 根据叉积结果返回转向标记（负值表示友方在航向右侧）
            if val < 0:
                return 1
            else:
                return -1
        else:
            return 1

    def _sample_state(self, agent, i, r):
        """
        根据智能体类型、索引和区域编号生成随机状态坐标
        
        参数:
            agent (str): 智能体类型，"agent"表示己方，"opp"表示对方
            i (int): 索引值，影响y坐标的偏移量
            r (int): 区域编号(1或2)，决定坐标生成范围
            
        返回:
            tuple: 包含x坐标(float)、y坐标(float)、角度(int)的三元组
        """
        x = 0
        y = 0
        a = 0
        # 处理己方智能体状态生成
        if agent == "agent":
            # Level 1的基础生成规则
            if self.args.level == 1:
                if r == 1:
                    x = random.uniform(7.12, 7.14)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                    a = random.randint(30, 150)
                elif r == 2:
                    x = random.uniform(7.16, 7.17)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                    a = random.randint(200, 330)
            # Level 2的扩展生成规则
            elif self.args.level == 2:
                if r == 1:
                    x = random.uniform(7.08, 7.13)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 180)
                elif r == 2:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(180, 359)
            # Level 3及以上的高级生成规则
            elif self.args.level >= 3:
                if r == 1:
                    x = random.uniform(7.07, 7.12)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(0, 270)
                elif r == 2:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(90, 359)

        # 处理对方智能体状态生成        
        elif agent == "opp":
            # Level 1的基础生成规则
            if self.args.level == 1:
                if r == 1:
                    x = random.uniform(7.16, 7.17)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
                elif r == 2:
                    x = random.uniform(7.12, 7.14)
                    y = random.uniform(5.1 + i*0.1, 5.11 + i*0.1)
            # Level 2的扩展生成规则
            elif self.args.level == 2:
                if r == 1:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 359)
                elif r == 2:
                    x = random.uniform(7.08, 7.13)
                    y = random.uniform(5.08 + i*0.1, 5.13 + i*0.1)
                    a = random.randint(0, 359)
            # Level 3及以上的高级生成规则
            elif self.args.level >= 3:
                if r == 1:
                    x = random.uniform(7.18, 7.23)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(0, 359)
                elif r == 2:
                    x = random.uniform(7.07, 7.12)
                    y = random.uniform(5.09 + i*0.1, 5.12 + i*0.1)
                    a = random.randint(0, 359)
        
        return x,y,a

    def _reset_scenario(self, mode):
        """
        create aircraft units (Rafale and RafaleLong).
        """
        """
        重置战斗场景并初始化作战单位
        
        Args:
            mode (str): 场景模式，接受"LowLevel"或其他模式值。控制敌方单位的武器配置策略
                        "LowLevel"模式会根据关卡等级调整敌方武器容量
                        非"LowLevel"模式使用默认武器配置
        """
        # 随机选择初始阵营位置（1或2）
        r = random.randint(1,2) #chose sides
        # 迭代创建己方（agent）和敌方（opp）单位
        for group, count in [("agent", self.args.num_agents), ("opp", self.args.num_opps)]:
            # 按配置数量创建作战单位
            for i in range(count):
                # 获取单位初始状态（坐标和朝向）
                x, y, a = self._sample_state(group, i, r)
                # 确保前两个单位使用不同机型，后续随机选择机型
                #at least one aircraft type (ac) per group
                ac = i+1 if i <=1 else random.randint(1,2)
                # 初始化单位基础属性
                if ac == 1:
                    unit = Rafale(Position(y, x, 10_000), heading=a, speed=0 if self.args.level<=2 and group=="opp" else 100, group=group, friendly_check=self.args.friendly_kill)
                else:
                    unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=0 if self.args.level<=2 and group=="opp" else 100, group=group, friendly_check=self.args.friendly_kill)

                # 配置武器系统参数
                if mode == "LowLevel":
                    # 低难度模式武器配置逻辑
                    if self.args.level <= 4 and group == "opp":
                        unit.cannon_max = unit.cannon_remain_secs = 400
                        if ac == 1:
                            unit.missile_remain = unit.rocket_max = 8
                    elif self.args.level == 5:
                        unit.cannon_max = unit.cannon_remain_secs = 300
                        if ac == 1:
                            unit.missile_remain = unit.rocket_max = 6
                else:
                    # 标准模式武器配置
                    unit.cannon_max = unit.cannon_remain_secs = 300
                    if ac == 1:
                        unit.missile_remain = unit.rocket_max = 8

                # 将单位加入模拟系统
                self.sim.add_unit(unit)
                self.sim.record_unit_trace(unit.id)
                # 更新存活单位计数器
                if group == "agent":
                    self.alive_agents += 1
                else:
                    self.alive_opps += 1

    def _plot_airplane(self, a: Rafale, side: str, path=True, use_backup=False, u_id=0):
        """
        绘制飞机及其轨迹的可视化对象
        
        参数:
        a (Rafale): 要绘制的飞机对象
        side (str): 所属阵营标识，'red'表示红方，'blue'表示蓝方
        path (bool): 是否绘制飞行路径，默认为True
        use_backup (bool): 是否使用备份数据绘制，默认为False
        u_id (int): 单位ID，当使用备份数据时指定的单位标识，默认为0
        
        返回:
        list: 包含可视化图形对象的列表
        """
        objects = []
        # 使用备份数据绘制历史轨迹
        if use_backup:
            # 从模拟记录中提取单位轨迹坐标
            trace = [(position.lat, position.lon) for t, position, heading, speed in self.sim.trace_record_units[u_id]]
            # 创建虚线轨迹线
            objects.append(PolyLine(trace, line_width=1, dash=(2, 2),
                                    edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                    zorder=0))
            # 添加终点标记点
            objects.append(Waypoint(trace[-1][0], trace[-1][1],
                                edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                fill_color=colors['red_fill'] if side == 'red' else colors['blue_fill'],
                                info_text=f"r_{u_id}", zorder=0))
        else:
            # 创建当前单位图形对象
            objects = [Airplane(a.position.lat, a.position.lon, a.heading,
                                edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                fill_color=colors['red_fill'] if side == 'red' else colors['blue_fill'],
                                info_text=f"r_{a.id}", zorder=0)]
            # 绘制实时飞行路径
            if path:
                trace = [(position.lat, position.lon) for t, position, heading, speed in self.sim.trace_record_units[a.id]]
                objects.append(PolyLine(trace, line_width=1, dash=(2, 2),
                                        edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                        zorder=0))
            # 绘制机炮攻击范围扇形
            if a.cannon_current_burst_secs > 0:  # noqa
                # 计算扇形边界坐标
                d1 = geodetic_direct(a.position.lat, a.position.lon,
                                    sum_angles(a.heading, a.cannon_width_deg / 2.0),
                                    a.cannon_range_km * 1000)
                d2 = geodetic_direct(a.position.lat, a.position.lon,
                                    sum_angles(a.heading, - a.cannon_width_deg / 2.0),
                                    a.cannon_range_km * 1000)
                # 创建扇形多边形
                objects.append(PolyLine([(a.position.lat, a.position.lon),
                                        (d1[0], d1[1]), (d2[0], d2[1]),
                                        (a.position.lat, a.position.lon)], line_width=1, dash=(1, 1),
                                        edge_color=colors['red_outline'] if side == 'red' else colors['blue_outline'],
                                        zorder=0))
        return objects

    def plot(self, out_file: Path, paths=True):
        """
        Draw current scenario.
        """
        """
        绘制当前场景并输出为图像文件
        
        参数:
            out_file (Path): 输出图像文件的保存路径
            paths (bool): 是否绘制飞机飞行路径，默认为True
        
        返回值:
            None: 无直接返回值，结果输出到指定文件
        """
        # 初始化基础绘图元素（状态信息和时间显示）
        objects = [
            StatusMessage(self.sim.status_text),
            TopLeftMessage(self.sim.utc_time.strftime("%Y %b %d %H:%M:%S"))
        ]
        # 绘制所有飞机单位（蓝方agents和红方opponents）
        # 根据单位存在状态生成实际飞机或占位标记
        for i in range(1, self.args.num_agents+self.args.num_opps+1):
            # 颜色区分阵营
            col = 'blue' if i<=self.args.num_agents else 'red'
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                objects.extend(self._plot_airplane(unit, col, paths))
            else:
                objects.extend(self._plot_airplane(None, col, paths, True, i))
        # 绘制导弹单位（根据发射源区分阵营颜色）
        for i in range(self.args.total_num+1, self.args.total_num*5+2):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                col = "blue" if unit.source.id <= self.args.num_agents else "red"
                objects.append(
                    Missile(unit.position.lat, unit.position.lon, unit.heading, edge_color=colors[f'{col}_outline'], fill_color=colors[f'{col}_fill'],
                    info_text=f"m_{i}", zorder=0),
                )
        # 将所有绘图对象渲染为PNG图像
        self.plotter.to_png(str(out_file), objects)

