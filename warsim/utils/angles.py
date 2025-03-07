"""
    Angles computations
"""

import math

DEG_TO_RAD = math.pi / 180


def normalize_angle(a: float) -> float:
    """
    将输入角度归一化到[0.0, 360.0)区间

    Args:
        a: 输入的角度值，单位为度。允许负值和超过360度的数值

    Returns:
        经过归一化处理后的角度值，始终满足 0.0 <= 返回值 < 360.0
    """
    # 处理超过360度的角度值（连续减法避免浮点数精度问题）
    while a >= 360.0:
        a -= 360
    # 处理负角度值（连续加法避免浮点数精度问题）
    while a < 0.0:
        a += 360
    return a


def sum_angles(a: float, b: float) -> float:
    """
    计算两个角度的和并进行角度归一化处理

    参数:
        a (float): 第一个角度值，单位为度
        b (float): 第二个角度值，单位为度

    返回:
        float: 归一化后的角度和，具体归一化范围由normalize_angle函数定义
        （典型范围可能是[0, 360)或[-180, 180)）
    """
    return normalize_angle(a + b)


def signed_heading_diff(actual: float, desired: float) -> float:
    """计算两个航向角度之间的有符号角度差值

    参数：
        actual: 当前航向角度（单位：度），取值范围 [0, 360)
        desired: 目标航向角度（单位：度），取值范围 [0, 360)

    返回值：
        float: 介于 [-180, 180) 的有符号角度差值，
        正值表示目标航向在实际航向的顺时针方向，负值表示逆时针方向
    """
    # actual and desired in [0, 360)
    # 处理环状角度空间的边界问题
    delta = desired - actual
    if delta < -180:
        # 处理逆时针方向超过180度的情况，添加完整周角
        delta = 360 + delta
    if delta > 180:
        # 处理顺时针方向超过180度的情况，减去完整周角
        delta = -360 + delta
    return delta
