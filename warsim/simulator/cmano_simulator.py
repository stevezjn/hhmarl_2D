"""
    CmanoSimulator is a simulator of the essential characteristic of CMANO.
    It is used to pre-train AI agent with basic capabilities that will be
    further refined with a real CMANO simulation.
"""

from __future__ import annotations

import random
from abc import ABC
from datetime import datetime, timedelta
from typing import Callable, List, Dict

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname((__file__))))

from utils.geodesics import geodetic_direct, geodetic_distance_km, geodetic_bearing_deg

# --- Constants
knots_to_ms = 0.514444


# --- Classes
class Position:
    """
    表示三维地理坐标位置

    属性:
        lat (float): 纬度值，范围[-90, +90]
        lon (float): 经度值，范围[0, 180]
        alt (float): 海拔高度值，单位为米
    """

    def __init__(self, lat: float, lon: float, alt: float):
        """
        初始化地理坐标位置

        参数:
            lat (float): 纬度，有效范围[-90, +90]
            lon (float): 经度，有效范围[0, 180]
            alt (float): 海拔高度，单位为米
        """
        self.lat = lat  # Latitude [-90; +90]
        self.lon = lon  # Longitude [0; 180]
        self.alt = alt  # Altitude (meters)

    def copy(self) -> Position:
        """
        创建当前坐标位置的深拷贝对象

        返回:
            Position: 包含相同经纬度和海拔的新位置对象实例
        """
        return Position(self.lat, self.lon, self.alt)


class Event(ABC):
    """表示一个抽象事件基类，用于派生具体事件类型
    
    属性：
        name (str): 事件名称标识
        origin (Unit): 事件来源的单位实例
    """
    def __init__(self, name, origin: Unit):
        """初始化事件实例
        
        Args:
            name: 事件名称，建议使用全大写字母命名
            origin: 事件来源的单位对象，包含单位类型和ID信息
        """
        self.name = name
        self.origin = origin

    def __str__(self):
        """生成事件的标准字符串表示
        
        返回格式示例：
            UnitType[123].EVENT_NAME
        
        Returns:
            str: 包含来源单位类型、ID和事件名称的格式化字符串
        """
        return f"{self.origin.type}[{self.origin.id}].{self.name}"


class UnitDestroyedEvent(Event):
    """表示单位被销毁事件的类，继承自基础事件类
    
    Attributes:
        unit_killer (Unit): 执行击杀动作的单位实例
        unit_destroyed (Unit): 被销毁的单位实例
    """
    def __init__(self, origin: Unit, unit_killer: Unit, unit_destroyed: Unit):
        """初始化单位销毁事件实例
        
        Args:
            origin: 事件发起源单位
            unit_killer: 造成单位销毁的施动单位
            unit_destroyed: 被销毁的目标单位
        """
        super().__init__("UnitDestroyedEvent", origin)
        self.unit_killer = unit_killer
        self.unit_destroyed = unit_destroyed

    def __str__(self):
        """生成事件的标准字符串表示
        
        格式说明：
        - 继承父类的基础事件字符串表示
        - 追加击杀单位类型及ID信息
        - 追加被销毁单位类型及ID信息
        
        Returns:
            str: 完整的事件状态描述字符串
        """
        return super().__str__() + f"({self.unit_killer.type}{[self.unit_killer.id]} ->" \
                                   f"{self.unit_destroyed.type}{[self.unit_destroyed.id]})"


class Unit(ABC):
    """
    表示模拟环境中的基本单位抽象基类

    Attributes:
        type: 单位类型标识字符串
        position: 包含经纬度、高度的位置对象
        heading: 单位航向角（单位：度），范围[0,360)
        speed: 单位移动速率（单位：节），标量值
        id: 单位唯一标识符
    """
    def __init__(self, type: str, position: Position, heading: float, speed_knots: float):
        """
        初始化军事单位实例

        Args:
            type: 单位类型描述字符串
            position: 包含初始经纬度的高度位置对象
            heading: 初始航向角度（0度表示正北，顺时针方向）
            speed_knots: 初始移动速度（单位：节）

        Raises:
            Exception: 当航向角不在[0,360)范围时抛出
        """
        if heading >= 360 or heading < 0:
            raise Exception(f"Unit.__init__: bad heading {heading}")
        self.type = type
        self.position = position
        self.heading = heading  # bearing, in degrees [0, 360)
        self.speed = speed_knots  # scalar, in knots
        self.id = None

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        """
        更新单位状态（每帧调用）

        Args:
            tick_secs: 自上次更新以来的时间间隔（秒）
            sim: 模拟器实例引用

        Returns:
            本次更新产生的事件列表

        Implementation:
            根据航向和速度计算新位置，当速度大于0时执行位置更新
        """
        # Basic position update based on speed
        if self.speed > 0:
            # 使用地理空间计算库进行航位推算
            d = geodetic_direct(self.position.lat, self.position.lon, self.heading,
                                self.speed * knots_to_ms * tick_secs)
            self.position.lat = d[0]
            self.position.lon = d[1]
        return []

    def to_string(self) -> str:
        """
        生成单位状态的格式化字符串表示

        Returns:
            包含单位类型、ID、位置、航向和速度的格式化字符串
            格式示例：Type[id]: p=(lat,lon,alt) h=(heading) s=(speed)
        """
        return f"{self.type}[{self.id}]: p=({self.position.lat:.4f}, {self.position.lon:.4f}, " \
               f"{self.position.alt:.4f}) h=({self.heading:.4f}) s=({self.speed:.4f})"


class CmanoSimulator:
    """CMANO 仿真模拟器核心类

    管理模拟器状态、单位实体、时间推进和事件回调

    Args:
        utc_time: 模拟器初始UTC时间，默认为当前时间
        tick_secs: 单次推进的时间步长（秒）
        random_seed: 随机数生成器种子
        num_units: 初始友方单位数量
        num_opp_units: 初始敌方单位数量
    """
    def __init__(self, utc_time=datetime.now(), tick_secs=1, random_seed=None, num_units=0, num_opp_units=0):
        """
        初始化模拟环境实例

        参数：
            utc_time (datetime, optional): 初始UTC时间，默认为当前时间
            tick_secs (int, optional): 每次时间推进的秒数，默认为1秒
            random_seed (int, optional): 随机数生成器种子，未设置时使用系统随机源
            num_units (int, optional): 初始友方单位数量，默认为0
            num_opp_units (int, optional): 初始敌方单位数量，默认为0
        
        属性：
            active_units (Dict[int, Unit]): 活跃单位字典，键为ID，值为Unit对象
            trace_record_units (Dict[int, list]): 单位轨迹记录，键为ID，值为时间序列数据
            utc_time (datetime): 当前模拟时间
            utc_time_initial (datetime): 初始基准时间
            tick_secs (int): 时间推进步长(秒)
            _tick_callbacks (List[Callable]): 注册的时间推进回调函数列表
            rnd_gen (Random): 随机数生成器实例
            _next_unit_id (int): 下一个生成单位的ID序号
            status_text (str): 系统状态描述文本
            num_units (int): 当前友方单位总数
            num_opp_units (int): 当前敌方单位总数
        """
        self.active_units: Dict[int, Unit] = {}  # id -> Unit
        self.trace_record_units = {}  # id ->[(time, position, heading, speed), ...]
        self.utc_time = utc_time
        self.utc_time_initial = utc_time
        self.tick_secs = tick_secs
        self._tick_callbacks: List[Callable[[datetime], None]] = []  # registered tick callbacks fn: datetime -> ()
        self.random_seed = random_seed
        self.rnd_gen = random.Random(random_seed)
        self._next_unit_id = 1
        self.status_text = None

        self.num_units = num_units
        self.num_opp_units = num_opp_units

    def reset_sim(self, units):
        """重置模拟器到初始状态
        
        参数:
            self: 表示类的实例对象
            units (dict): 包含所有活动单元的字典，键为单元标识符，值为单元配置数据
        
        返回值:
            无返回值
        """
        # 重置内部状态变量
        self.utc_time = datetime.now()
        self._tick_callbacks = []
        self.status_text = None
        
        # 设置活动单元并初始化追踪
        self.active_units = units
        # 为每个单元创建追踪记录
        for i in units.keys():
            self.record_unit_trace(i)

    def add_unit(self, unit: Unit) -> int:
        """
        将新的单位对象添加到活动单位字典中，并分配唯一ID

        Args:
            unit (Unit): 需要添加的单位对象，必须包含id属性

        Returns:
            int: 返回分配给该单位的唯一标识ID
        """
        # 将单位对象存入活动单位字典，使用_next_unit_id作为键
        # 同时将ID绑定到单位对象上
        self.active_units[self._next_unit_id] = unit
        unit.id = self._next_unit_id

        # 递增ID生成器并返回已分配的ID值
        # (注意这里返回的是递增前的值，保证ID唯一性)
        self._next_unit_id += 1
        return self._next_unit_id - 1

    def remove_unit(self, unit_id: int):
        """
        从管理单元中移除指定ID的单位

        该方法会从活动单位集合中删除目标单位，后续将不再追踪该单位的状态变化。
        若需要同步移除追踪记录中的单位，需取消下方相关代码注释。

        Args:
            unit_id: 要移除的单位唯一标识符，要求为整数类型
        """
        # self.active_units[unit_id].id = None  # Remove this line after some tests

        # 直接删除活动单位字典中的对应条目
        # 当前实现假设调用方已确保unit_id存在
        #if self.unit_exists(unit_id):
        del self.active_units[unit_id]

        # 以下为预留的追踪记录清理逻辑
        # 若需要同步清理追踪记录中的单位信息，可取消下方注释
        # if unit_id in self.trace_record_units:
        #     del self.trace_record_units[unit_id]

    def get_unit(self, unit_id: int) -> Unit:
        """
        根据单位ID获取对应的单位对象
        
        :param unit_id: 要获取的单位的唯一标识符（整数类型）
        :return: 从活动单位字典中查找到的Unit实例
        """
        return self.active_units[unit_id]

    def unit_exists(self, unit_id: int) -> bool:
        """
        检查指定单位ID是否存在于活动单位集合中
        
        参数:
        unit_id (int): 需要验证的单位唯一标识符
        
        返回值:
        bool: 当单位存在时返回True，否则返回False
        """
        return unit_id in self.active_units

    def record_unit_trace(self, unit_id: int):
        """
        开始记录指定单位的状态追踪轨迹
        
        参数:
            unit_id: 需要记录轨迹的单位标识符
            
        异常:
            如果单位不存在于活动单位列表中则抛出异常
        """
        if unit_id not in self.active_units:
            raise Exception(f"Unit.record_unit_trace(): unknown unit {unit_id}")
        if unit_id not in self.trace_record_units:
            self.trace_record_units[unit_id] = []
            self._store_unit_state(unit_id)

    def set_status_text(self, text: str):
        """
        设置当前状态显示文本
        
        参数:
            text: 需要显示的状态文本内容
        """
        self.status_text = text

    def add_tick_callback(self, cb_fn: Callable[[datetime], None]):
        """
        添加时间刻度的回调函数
        
        参数:
            cb_fn: 接受UTC时间参数的函数，将在每个时间刻度触发
        """
        self._tick_callbacks.append(cb_fn)

    def do_tick(self) -> List[Event]:
        """
        执行一个时间刻度的仿真更新
        
        返回:
            返回本刻度内产生的所有事件列表
            
        处理流程:
            1. 更新所有活动单位的状态
            2. 保存当前追踪单位的状态快照
            3. 触发注册的回调函数
        """
        # 更新所有单位状态并收集事件
        # Update the state of all units
        events = []
        #kills = {}
        for unit in list(self.active_units.values()):
            event = unit.update(self.tick_secs, self)
            events.extend(event)
            #kills.update(kill)
        self.utc_time += timedelta(seconds=self.tick_secs)

        # 保存追踪单位的状态记录
        # Save trace
        for _ in self.trace_record_units.keys():
            self._store_unit_state(_)

        # 触发时间刻度回调
        # Notify clients
        for fn in self._tick_callbacks:
            fn(self.utc_time)

        #return events, kills
        return events

    def _store_unit_state(self, unit_id):
        """
        内部方法：存储指定单位的当前状态快照
        
        参数:
            unit_id: 需要记录状态的单位标识符
            
        存储内容包含：
            - 记录时间戳
            - 单位位置（存储副本）
            - 航向角度
            - 当前速度
        """
        if self.unit_exists(unit_id):
            unit = self.active_units[unit_id]
            self.trace_record_units[unit_id].append((self.utc_time, unit.position.copy(), unit.heading, unit.speed))


# --- General purpose utilities

def units_distance_km(unit_a: Unit, unit_b: Unit) -> float:
    """计算两个单位之间的测地线距离（单位：千米）

    基于WGS84椭球体模型计算两个地理坐标点之间的最短地表距离

    参数:
        unit_a: 包含地理坐标（经纬度）的单位实例
        unit_b: 包含地理坐标（经纬度）的单位实例

    返回:
        float: 两个单位之间的地表距离（千米）
    """
    return geodetic_distance_km(unit_a.position.lat, unit_a.position.lon,
                                unit_b.position.lat, unit_b.position.lon)


def units_bearing(unit_from: Unit, unit_to: Unit) -> float:
    """计算从起始单位到目标单位的初始方位角（单位：度）

    基于测地线计算起始点指向目标点的初始罗盘方位角（真北基准）

    参数:
        unit_from: 起始单位（包含经纬度坐标）
        unit_to: 目标单位（包含经纬度坐标）

    返回:
        float: 方位角度数（0-360度，0度表示正北方向）
    """
    return geodetic_bearing_deg(unit_from.position.lat, unit_from.position.lon,
                                unit_to.position.lat, unit_to.position.lon)
