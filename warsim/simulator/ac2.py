"""
    A modified Rafale airplane unit
"""

from typing import List
import numpy as np
import random

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator, units_bearing, UnitDestroyedEvent, \
    units_distance_km
from utils.angles import signed_heading_diff, sum_angles

class UnitDetected(Event):
    """表示单位检测到其他单位的事件
    
    Attributes:
        detected_unit (Unit): 被检测到的单位实例
    """
    def __init__(self, origin: Unit, detected_unit: Unit):
        """初始化单位检测事件
        
        Args:
            origin (Unit): 触发检测事件的来源单位实例
            detected_unit (Unit): 被检测到的目标单位实例
        """
        super().__init__("UnitDetected", origin)
        self.detected_unit = detected_unit

    def __str__(self):
        """生成事件的字符串表示
        
        Returns:
            str: 事件名称(单位类型[单位ID]) 的格式字符串
        """
        # 调用父类字符串表示并追加检测单位的类型和ID信息
        return super().__str__() + f"({self.detected_unit.type}{[self.detected_unit.id]})"


class RafaleLong(Unit):
    """
    阵风战斗机(长机型)性能参数配置类
    
    属性说明：
        max_deg_sec (float): 最大转向速率(度/秒)
        min_speed_knots (int): 最低飞行速度(节)
        max_speed_knots (int): 最高飞行速度(节)
        max_knots_sec (int): 最大加速度(节/秒)
        cannon_range_km (float): 机炮有效射程(千米)
        cannon_width_deg (int): 机炮射击角度范围(度)
        cannon_max_time_sec (int): 机炮最大持续射击时间(秒)
        cannon_burst_time_sec (int): 机炮连发持续时间(秒)
        cannon_hit_prob (float): 机炮命中概率(0.0~1.0)
        aircraft_type (int): 机型分类标识(2表示长程机型)
    """
    max_deg_sec = 3.5
    min_speed_knots = 0
    max_speed_knots = 600
    max_knots_sec = 28
    cannon_range_km = 4.5
    cannon_width_deg = 7
    cannon_max_time_sec = 200
    cannon_burst_time_sec = 3
    cannon_hit_prob = 0.9
    aircraft_type = 2

    def __init__(self, position: Position, heading: float, speed: float, group:str, friendly_check: bool = True):
        """初始化阵风长程战斗机实例

         Args:
             position (Position): 初始地理坐标位置对象
             heading (float): 初始航向角度（0-360度）
             speed (float): 初始飞行速度（单位：节）
             group (str): 所属作战群组标识（"agent"或"opp"）
             friendly_check (bool, optional): 是否开启友军识别检查（默认开启）
         """
        # 调用父类Unit初始化基础航空器属性
        super().__init__("RafaleLong", position, heading, speed)
        # 飞行控制系统参数
        # 目标航向角度
        self.new_heading = heading
        # 目标飞行速度
        self.new_speed = speed
        # 最大飞行速度限制
        self.max_speed = RafaleLong.max_speed_knots

        # Cannon
        # 机炮系统状态管理
        # 剩余可用射击时间
        self.cannon_remain_secs = RafaleLong.cannon_max_time_sec
        # 当前连射持续时间计数器
        self.cannon_current_burst_secs = 0
        # 最大持续射击时间配置
        self.cannon_max = RafaleLong.cannon_max_time_sec

        # 武器挂载状态
        # 当前激活的导弹对象
        self.actual_missile = None
        # 剩余导弹库存量
        self.missile_remain = 0
        # 火箭弹最大载弹量（当前型号未启用）
        self.rocket_max = 0

        # group type and if friendly aircrafts considered
        # 作战识别系统配置
        # 友军识别开关
        self.friendly_check = friendly_check
        # 所属作战阵营标识
        self.group = group
        # 机型分类编码（固定值2）
        self.ac_type = RafaleLong.aircraft_type

    def set_heading(self, new_heading: float):
        """
        设置对象的新航向
        
        参数:
        new_heading (float): 目标航向角度值，必须满足 0 <= new_heading < 360，
            以度为单位表示方向
        
        异常:
        Exception: 当输入值超出合法范围时抛出
        
        返回值:
        None
        """
        # 验证航向角度值在合法区间[0, 360)
        if new_heading >= 360 or new_heading < 0:
            raise Exception(f"RafaleLong.set_heading Heading must be in [0, 360), got {new_heading}")
        # 设置新的航向属性
        self.new_heading = new_heading

    def set_speed(self, new_speed: float):
        """
        设置当前对象的速度值

        参数:
        new_speed (float): 要设置的新速度值（单位：节）
        
        异常:
        Exception: 当速度超过允许范围时抛出，包含详细错误信息
        
        返回值:
        None
        """
        # 验证输入速度是否在合法速度范围内
        if new_speed > RafaleLong.max_speed_knots or new_speed < RafaleLong.min_speed_knots:
            raise Exception(f"RafaleLong.set_speed Speed must be in [{RafaleLong.min_speed_knots}, {RafaleLong.max_speed_knots}] "
                            f"knots, got {new_speed}")
        # 通过验证后更新速度属性
        self.new_speed = new_speed

    def fire_cannon(self):
        """
        控制火炮单次爆发持续时间
        
        计算并设置当前火炮爆发持续时间，该时间不超过火炮剩余可用时间
        和机型预设的单次爆发时间中的较小值
        
        Args:
            self: 表示当前战斗机实例，包含火炮相关状态属性
            
        Returns:
            None: 直接修改实例的cannon_current_burst_secs属性
        """
        # 计算并设置当前火炮爆发持续时间
        self.cannon_current_burst_secs = min(self.cannon_remain_secs, RafaleLong.cannon_burst_time_sec)

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        """更新飞行器状态并返回模拟事件
        
        Args:
            tick_secs: 时间间隔（秒），表示自上次更新经过的时间
            sim: 模拟器实例，用于访问其他单位信息和执行模拟操作
            
        Returns:
            List[Event]: 本次更新周期内产生的事件列表
        """
        
        # 调整飞行器朝向到新目标航向
        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = RafaleLong.max_deg_sec * tick_secs
            if abs(delta) <= max_deg:
                self.heading = self.new_heading
            else:
                self.heading += max_deg if delta >= 0 else -max_deg
                self.heading %= 360

        # 调整飞行速度到新目标速度
        # Update speed
        if self.speed != self.new_speed:
            delta = self.new_speed - self.speed
            max_delta = RafaleLong.max_knots_sec * tick_secs
            if abs(delta) <= max_delta:
                self.speed = self.new_speed
            else:
                self.speed += max_delta if delta >= 0 else -max_delta

        # 处理机炮射击逻辑和命中判定
        # Update cannon
        events = []
        if self.cannon_current_burst_secs > 0:
            self.cannon_current_burst_secs = max(self.cannon_current_burst_secs - tick_secs, 0.0)
            self.cannon_remain_secs = max(self.cannon_remain_secs - tick_secs, 0.0)
            # 遍历所有活动单位进行射击检测
            for unit in list(sim.active_units.values()):
                if unit.id != self.id:
                    # 仅处理飞机单位（排除导弹等）
                    if unit.id <= sim.num_units + sim.num_opp_units: # consider only aircrafts, no rockets
                        # 检查敌我识别条件
                        if self.friendly_check or (self.group == "agent" and unit.id >= sim.num_units+1) or (self.group == "opp" and unit.id <= sim.num_units):
                            if unit.type in ["RafaleLong", "Rafale"]:
                                # 命中判定和单位摧毁处理
                                if self._unit_in_cannon_range(unit):
                                    if sim.rnd_gen.random() < \
                                            (RafaleLong.cannon_hit_prob / (RafaleLong.cannon_burst_time_sec / tick_secs)):
                                        sim.remove_unit(unit.id)
                                        events.append(UnitDestroyedEvent(self, self, unit))
        
        # 调用父类位置更新并收集事件
        # Update position
        events.extend(super().update(tick_secs, sim))

        return events

    def _unit_in_cannon_range(self, u: Unit) -> bool:
        """
        检查目标单位是否处于当前单位的机炮有效射程范围内。

        参数:
            self: 当前作战单位实例
            u (Unit): 待检测的目标单位

        返回:
            bool: 目标同时满足射程和角度条件时返回True，否则返回False
        """
        distance = units_distance_km(self, u)
        # 距离有效性检查：小于机炮射程时才进行角度判断
        if distance < RafaleLong.cannon_range_km:
            # 计算目标相对方位角及航向偏差角度
            bearing = units_bearing(self, u)
            delta = abs(signed_heading_diff(self.heading, bearing))
            # 判断目标是否处于机炮有效射击角度范围内
            return delta <= RafaleLong.cannon_width_deg / 2.0
        else:
            return False
