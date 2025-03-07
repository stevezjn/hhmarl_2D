"""
    Geodesics computations
"""

from typing import Tuple

from geographiclib.geodesic import Geodesic

from utils.angles import normalize_angle


def geodetic_distance_km(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    """
    计算两个地理坐标点之间的测地线距离（单位：公里）

    使用WGS84椭球体模型进行高精度测地线计算，基于逆测地线问题解法。该方法适用于地球表面
    任意两点间距离计算，包含椭球体曲率修正。

    :param lat_1: 起始点纬度坐标，单位：度，有效范围[-90, 90]
    :param lon_1: 起始点经度坐标，单位：度，有效范围[-180, 180]
    :param lat_2: 终点纬度坐标，单位：度，有效范围[-90, 90]
    :param lon_2: 终点经度坐标，单位：度，有效范围[-180, 180]
    :return: 两点间测地线距离（单位：公里），返回浮点数保留原始计算精度

    实现说明：
    - 使用Geodesic.WGS84基准椭球体参数
    - Inverse方法设置outmask=DISTANCE仅计算距离参数
    - s12字段返回的是米为单位的距离值，需转换为公里
    """
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.DISTANCE)
    return r["s12"] / 1000.0


def geodetic_bearing_deg(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    """
    计算两个地理坐标点之间的初始大地方位角（以度为单位）

    参数:
    lat_1 -- 起始点的纬度（十进制度数，WGS84坐标系）
    lon_1 -- 起始点的经度（十进制度数，WGS84坐标系）
    lat_2 -- 终止点的纬度（十进制度数，WGS84坐标系）
    lon_2 -- 终止点的经度（十进制度数，WGS84坐标系）

    返回值:
    float -- 归一化后的方位角（0-360度），以真北方向为基准顺时针测量

    说明:
    基于WGS84椭球模型进行逆地理编码计算
    该方位角表示从起始点指向终止点的初始方向
    """
    # 执行逆地理编码计算，仅获取方位角参数
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.AZIMUTH)
    # 将初始方位角归一化到0-360度范围
    return normalize_angle(r["azi1"])


def geodetic_direct(lat: float, lon: float, heading: float, distance: float) -> Tuple[float, float]:
    """
    根据起点坐标、方位角和距离计算目标点的大地坐标（WGS84椭球体）

    Args:
        lat: 起点纬度(单位：度)，取值范围[-90, 90]
        lon: 起点经度(单位：度)，取值范围[-180, 180]
        heading: 前进方位角(单位：度)，正北方向为0度，顺时针增加
        distance: 沿测地线移动距离(单位：米)，必须为非负数

    Returns:
        Tuple[float, float]: 包含目标点纬度(lat2)和经度(lon2)的元组(单位：度)

    实现说明：
        使用GeographicLib库的WGS84椭球体参数进行测地正解计算
    """
    # 调用Geodesic正解算法，指定输出纬度和经度参数
    d = Geodesic.WGS84.Direct(lat, lon, heading, distance, outmask=Geodesic.LATITUDE | Geodesic.LONGITUDE)
    return d["lat2"], d["lon2"]
