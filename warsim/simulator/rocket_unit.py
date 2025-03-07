"""
    A PAC-3 Missile Unit
"""

from datetime import datetime
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator, units_distance_km, UnitDestroyedEvent
from utils.angles import signed_heading_diff

class Rocket(Unit):
    """
    火箭作战单位类，继承自基础Unit类，模拟导弹的飞行轨迹和作战行为

    类属性:
        max_deg_sec: 最大转向速度（度/秒）
        speed_profile: 时间-速度插值函数，根据存活时间计算当前速度值
    """
    max_deg_sec = 10
    speed_profile_time = np.array([0, 10, 20, 30])
    #speed_profile_time = np.array([0, 10, 20, 30, 40])
    speed_profile_knots = np.array([500, 2000, 1400, 600])
    #speed_profile_knots = np.array([400, 2000, 1800, 1200, 500]) #fill_value = (400,500)
    speed_profile = interp1d(speed_profile_time, speed_profile_knots, kind='quadratic', assume_sorted=True,
                             bounds_error=False, fill_value=(500, 600))

    def __init__(self, position: Position, heading: float, firing_time: datetime, target: Unit, source: Unit, friendly_check: bool = True):
        """
        初始化火箭实例

        Args:
            position: 初始位置坐标
            heading: 初始航向角度（0-360度）
            firing_time: 导弹发射时间（UTC时间）
            target: 攻击目标单位对象
            source: 导弹发射源单位对象
            friendly_check: 是否开启友军碰撞检查（默认开启）
        """
        self.speed = Rocket.speed_profile(0)
        super().__init__("Rocket", position, heading, self.speed)
        self.new_heading = heading
        self.firing_time = firing_time
        self.target = target
        self.source = source
        self.friendly_check = friendly_check

    def set_heading(self, new_heading: float):
        """
        设置新的目标航向

        Args:
            new_heading: 新航向角度（0-360度）

        Raises:
            Exception: 当输入角度超出有效范围时抛出
        """
        if new_heading >= 360 or new_heading < 0:
            raise Exception(f"Rocket.set_heading Heading must be in [0, 360), got {new_heading}")
        self.new_heading = new_heading

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        """
        更新火箭状态（每帧调用）

        Args:
            tick_secs: 时间步长（秒）
            sim: 模拟器实例

        Returns:
            返回本帧产生的事件列表，可能包含单位摧毁事件
        """
        """目标命中检测：当与目标距离<1km时触发摧毁事件"""
        # Check if the target has been hit
        if units_distance_km(self, self.target) < 1 and sim.unit_exists(self.target.id):
            sim.remove_unit(self.id)
            sim.remove_unit(self.target.id)
            return [UnitDestroyedEvent(self, self.source, self.target)]
        
        """友军碰撞检测：当开启检查且与友军距离<1km时触发摧毁事件"""
        if self.friendly_check:
            # check if friendly aircraft has been hit
            friendly_id = 1 if self.source.id == 2 else 2
            if sim.unit_exists(friendly_id):
                friendly_unit = sim.get_unit(friendly_id)
                if units_distance_km(self, friendly_unit) < 1:
                    sim.remove_unit(self.id)
                    sim.remove_unit(friendly_id)
                    return [UnitDestroyedEvent(self, self.source, friendly_unit)]

        """生命周期检测：超过速度曲线定义的有效时间后自动销毁"""
        # Check if eol is arrived
        life_time = (sim.utc_time - self.firing_time).seconds
        if life_time > Rocket.speed_profile_time[1]:
            sim.remove_unit(self.id)
            return []

        """航向调整：根据最大转向速度逐步转向新航向"""
        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = Rocket.max_deg_sec * tick_secs
            if abs(delta) <= max_deg:
                self.heading = self.new_heading
            else:
                self.heading += max_deg if delta >= 0 else -max_deg

        """速度更新：根据存活时间从预定义速度曲线获取当前速度"""
        # Update speed
        self.speed = Rocket.speed_profile(life_time)

        """位置更新：调用父类方法处理基础运动学计算"""
        # Update position
        return super().update(tick_secs, sim)
