"""
    A static Waypoint unit
"""

from typing import List

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator


class Waypoint(Unit):
    """表示仿真环境中的路径点/导航点单元
    
    继承自Unit基类，用于标记或导航到特定位置点，可包含描述性文本
    
    Args:
        position: 路径点的三维坐标位置
        heading: 路径点的朝向角度（单位：度）
        text: 与路径点关联的可选描述文本，默认为空
    """

    def __init__(self, position: Position, heading: float, text=None):
        # 调用父类构造器初始化基础单元属性
        # 固定单元类型为"Waypoint"，速度为0表示静止不动
        super().__init__("Waypoint", position, heading, 0)
        # 存储路径点的附加描述信息
        self.text = text

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        """执行路径点单元的仿真更新逻辑
        
        Args:
            tick_secs: 距上次更新的时间间隔（秒）
            sim: 所属的仿真器实例引用
            
        Returns:
            当前更新周期产生的事件列表。基础路径点无默认行为，
            始终返回空列表，子类可重写该方法实现具体逻辑
        """
        # 基础路径点单元无状态更新需求
        # 保留空实现供子类扩展
        return []
