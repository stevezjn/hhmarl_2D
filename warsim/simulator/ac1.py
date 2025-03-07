"""
    A Rafale airplane unit with Rockets.
"""

from typing import List
import numpy as np
import random

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator, units_bearing, UnitDestroyedEvent, \
    units_distance_km
from simulator.rocket_unit import Rocket
from utils.angles import signed_heading_diff, sum_angles

class UnitDetected(Event):
    """单位检测事件类，当某个单位被检测到时触发
    
    Attributes:
        detected_unit (Unit): 被检测到的目标单位实例
        
    Inherits:
        Event: 继承自基础事件类
    """
    def __init__(self, origin: Unit, detected_unit: Unit):
        """初始化单位检测事件实例
        
        Args:
            origin (Unit): 触发检测事件的来源单位
            detected_unit (Unit): 被检测到的目标单位实例
        """
        super().__init__("UnitDetected", origin)
        self.detected_unit = detected_unit

    def __str__(self):
        """生成事件描述字符串
        
        Returns:
            str: 组合父事件描述与目标单位信息的格式化字符串
            格式示例：EventName(OriginUnit)[TargetType[TargetID]]
        """
        # 组合基础事件描述与目标单位详细信息
        return super().__str__() + f"({self.detected_unit.type}{[self.detected_unit.id]})"


class Rafale(Unit):
    """代表阵风战斗机的仿真模型类

    类属性:
        max_deg_sec (int): 最大转向速率(度/秒)
        min_speed_knots (int): 最小速度(节)
        max_speed_knots (int): 最大速度(节)
        max_knots_sec (int): 最大加速度(节/秒)
        cannon_range_km (float): 机炮有效射程(公里)
        cannon_width_deg (int): 机炮有效攻击角度范围(度)
        cannon_max_time_sec (int): 机炮持续射击总时长(秒)
        cannon_burst_time_sec (int): 机炮单次连发持续时间(秒)
        cannon_hit_prob (float): 机炮单发命中概率
        max_missiles (int): 最大导弹携带量
        missile_range_km (int): 导弹射程(公里)
        missile_width_deg (int): 导弹雷达扫描范围(度)
        aircraft_type (int): 飞机类型标识
    """
    max_deg_sec = 5
    min_speed_knots = 0
    max_speed_knots = 900
    max_knots_sec = 35
    cannon_range_km = 2.0
    cannon_width_deg = 10
    cannon_max_time_sec = 200
    cannon_burst_time_sec = 5
    cannon_hit_prob = 0.75
    max_missiles = 5
    missile_range_km = 111
    missile_width_deg = 120 # valid ?
    aircraft_type = 1

    def __init__(self, position: Position, heading: float, speed: float, group:str, friendly_check: bool = True):
        """
        初始化 Rafale 战机实例
        
        参数:
        position (Position): 战机的初始位置坐标
        heading (float): 初始航向角度（单位：度）
        speed (float): 初始飞行速度（单位：节）
        group (str): 战机所属的编队/分组标识
        friendly_check (bool): 是否考虑友方飞机（默认开启）
        """
        super().__init__("Rafale", position, heading, speed)
        self.new_heading = heading
        self.new_speed = speed
        # 最大飞行速度（节）
        self.max_speed = Rafale.max_speed_knots

        # Cannon
        # 机炮系统状态管理
        # 机炮剩余可用时间（秒）
        self.cannon_remain_secs = Rafale.cannon_max_time_sec
        # 当前连射持续时间
        self.cannon_current_burst_secs = 0
        # 机炮最大持续使用时间
        self.cannon_max = Rafale.cannon_max_time_sec

        # 导弹系统状态管理
        # 当前挂载的激活状态导弹
        self.actual_missile = None
        # 剩余导弹数量
        self.missile_remain = Rafale.max_missiles
        # 最大导弹携带量
        self.rocket_max = Rafale.max_missiles

        # group type and if friendly aircrafts considered
        # 编队及敌我识别配置
        # 友军识别开关
        self.friendly_check = friendly_check
        # 所属战斗群组
        self.group = group
        # 战机型号标识
        self.ac_type = Rafale.aircraft_type

    def set_heading(self, new_heading: float):
        """
        设置对象的新航向角

        Args:
            new_heading (float): 目标航向角（单位：度），取值范围[0, 360)
        
        Raises:
            Exception: 当输入航向角超出合法范围时抛出异常

        Returns:
            None
        """
        # 参数有效性校验：确保航向角在合法范围内
        if new_heading >= 360 or new_heading < 0:
            # 抛出数值越界异常，附带错误信息
            raise Exception(f"Rafale.set_heading Heading must be in [0, 360), got {new_heading}")
        # 更新对象航向状态
        self.new_heading = new_heading

    def set_speed(self, new_speed: float):
        """
        设置战机飞行速度并进行有效性校验

        验证输入速度是否在战机定义的合法操作速度范围内（最小/最大速度），
        通过校验后更新实例的速度属性

        Args:
            new_speed (float): 目标速度（单位：节）。必须介于类常量 Rafale.min_speed_knots
                和 Rafale.max_speed_knots 之间（含边界值）

        Raises:
            Exception: 当输入速度超出战机操作速度范围时抛出异常
        """
        # 校验速度是否在类定义的操作限制范围内
        if new_speed > Rafale.max_speed_knots or new_speed < Rafale.min_speed_knots:
            raise Exception(f"Rafale.set_speed Speed must be in [{Rafale.min_speed_knots}, {Rafale.max_speed_knots}] "
                            f"knots, got {new_speed}")
        # 通过校验后更新实例速度属性
        self.new_speed = new_speed

    def fire_cannon(self):
        """
        执行火炮发射动作，计算当前爆发持续时间。

        该方法根据火炮剩余可用时间和预设单次爆发时间，取较小值作为本次爆发时间，
        并更新实例的cannon_current_burst_secs属性。

        属性说明：
        - cannon_remain_secs: 火炮剩余可用时间（秒）
        - Rafale.cannon_burst_time_sec: 类常量，单次爆发最大持续时间（秒）
        """
        self.cannon_current_burst_secs = min(self.cannon_remain_secs, Rafale.cannon_burst_time_sec)

    def fire_missile(self, ag_unit:Unit, opp_unit:Unit, sim: CmanoSimulator):
        """
        发射导弹攻击目标单位
        
        参数:
        self -- 当前战机实例（Unit类型）
        ag_unit -- 攻击方单位（Unit类型）
        opp_unit -- 被攻击的敌方单位（Unit类型）
        sim -- 模拟器实例（CmanoSimulator类型）
        
        返回值:
        无
        """
        # 检查导弹状态和剩余数量
        if not self.actual_missile and self.missile_remain > 0:
            # 计算目标距离并验证攻击条件
            unit_distance = units_distance_km(self, opp_unit)
            if unit_distance <= Rafale.missile_range_km and self._angle_in_radar_range(units_bearing(self, opp_unit)):
                # 创建导弹实例并加入模拟器
                missile = Rocket(self.position.copy(), self.heading, sim.utc_time, opp_unit, ag_unit, self.friendly_check)
                sim.add_unit(missile)
                # 更新导弹状态和剩余数量
                self.actual_missile = missile
                self.missile_remain = max(0, self.missile_remain - 1)

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        """更新实体状态并生成仿真事件
        
        Args:
            tick_secs: 仿真时间步长（秒）
            sim: 仿真器实例，用于访问战场环境数据
            
        Returns:
            List[Event]: 本时间步内触发的事件列表
        """
        # 航向更新逻辑：以最大转向速率平滑转向目标航向
        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = Rafale.max_deg_sec * tick_secs
            if abs(delta) <= max_deg:
                self.heading = self.new_heading
            else:
                self.heading += max_deg if delta >= 0 else -max_deg
                self.heading %= 360

        # 速度更新逻辑：以最大加速度逼近目标速度
        # Update speed
        if self.speed != self.new_speed:
            delta = self.new_speed - self.speed
            max_delta = Rafale.max_knots_sec * tick_secs
            if abs(delta) <= max_delta:
                self.speed = self.new_speed
            else:
                self.speed += max_delta if delta >= 0 else -max_delta

        # 机炮系统逻辑：处理射击冷却，检测并攻击有效目标
        # Update cannon
        events = []
        if self.cannon_current_burst_secs > 0:
            self.cannon_current_burst_secs = max(self.cannon_current_burst_secs - tick_secs, 0.0)
            self.cannon_remain_secs = max(self.cannon_remain_secs - tick_secs, 0.0)
            # 目标筛选条件：非自身、飞机单位、符合阵营规则、在射程内
            for unit in list(sim.active_units.values()):
                if unit.id != self.id:
                    if unit.id <= sim.num_units + sim.num_opp_units: # consider only aircrafts, no rockets
                        if self.friendly_check or (self.group == "agent" and unit.id >= sim.num_units+1) or (self.group == "opp" and unit.id <= sim.num_units):
                            if unit.type in ["RafaleLong", "Rafale"]:
                                if self._unit_in_cannon_range(unit):
                                    # 根据时间步长调整命中概率计算
                                    if sim.rnd_gen.random() < \
                                            (Rafale.cannon_hit_prob / (Rafale.cannon_burst_time_sec / tick_secs)):
                                        sim.remove_unit(unit.id)
                                        events.append(UnitDestroyedEvent(self, self, unit))

        # 导弹系统逻辑：为现役导弹添加航向扰动
        # Update missile
        if self.actual_missile:
            if not sim.unit_exists(self.actual_missile.id):
                self.actual_missile = None
            else:
                # 注入5%的航向噪声模拟制导误差
                # if FIRE_OBJ == "missile":
                #     heading = units_bearing(self.actual_missile, self.actual_missile.target)
                #     heading = np.clip(heading * random.uniform(0.8, 1.2), 0, 359) #inject some noise in missile
                #     self.actual_missile.set_heading(heading)

                heading = np.clip(self.actual_missile.heading * random.uniform(0.95, 1.05), 0, 359)
                self.actual_missile.set_heading(heading)
        
        # 调用父类位置更新逻辑并合并事件
        # Update position
        events.extend(super().update(tick_secs, sim))

        return events

    def _unit_in_cannon_range(self, u: Unit) -> bool:
        """判断目标单位是否处于当前单位的炮火有效范围内

        Args:
            u (Unit): 需要检查的目标单位对象

        Returns:
            bool: 当同时满足射程要求和角度要求时返回True，否则返回False
        """
        # 计算与目标单位的实际距离（公里）
        distance = units_distance_km(self, u)
        # 检查单位是否在炮火有效射程内
        if distance < Rafale.cannon_range_km:
            # 计算目标相对于当前单位的方位角
            bearing = units_bearing(self, u)
            # 计算航向角与方位角的绝对差值
            delta = abs(signed_heading_diff(self.heading, bearing))
            # 检查方位角是否在炮火有效角度范围内
            return delta <= Rafale.cannon_width_deg / 2.0
        else:
            return False
        
    def _angle_in_radar_range(self, angle: float) -> bool:
        """
        检查目标角度是否处于导弹雷达的有效扫描范围内
        
        :param angle: 待检测的目标角度（单位：度）
        :return: 目标角度在雷达扫描范围内返回True，否则返回False
        """
        # 计算目标角度与雷达扫描半宽边界的绝对角度差
        delta = abs(signed_heading_diff(sum_angles(self.heading, Rafale.missile_width_deg / 2), angle))
        # 判断差值是否小于等于雷达扫描半宽（取整后比较）
        return int(delta) <= int(Rafale.missile_width_deg / 2.0)
