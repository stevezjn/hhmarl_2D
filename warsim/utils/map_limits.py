"""
    Implements a latitude-longitude rectangle that defines the allowable region
    for a simulation.
"""

from geographiclib.geodesic import Geodesic
import numpy as np


class MapLimits:
    """表示地理矩形区域的边界框，并提供坐标转换和距离计算方法

    属性:
        left_lon (float): 区域左侧经度（西经，单位：度）
        bottom_lat (float): 区域底部纬度（南纬，单位：度）
        right_lon (float): 区域右侧经度（东经，单位：度）
        top_lat (float): 区域顶部纬度（北纬，单位：度）
    """
    def __init__(self, left_lon, bottom_lat, right_lon, top_lat):
        """
        初始化地理边界框对象。

        参数:
            left_lon (float): 左边界经度，单位度
            bottom_lat (float): 下边界纬度，单位度
            right_lon (float): 右边界经度，单位度
            top_lat (float): 上边界纬度，单位度

        说明:
            通过左下角(left_lon, bottom_lat)和右上角(right_lon, top_lat)
            坐标定义矩形地理区域，适用于地图坐标系统的边界描述
        """
        self.left_lon = left_lon
        self.bottom_lat = bottom_lat
        self.right_lon = right_lon
        self.top_lat = top_lat

    def latitude_extent(self):
        """计算纬度范围的跨度值

        该方法通过顶部纬度(top_lat)减去底部纬度(bottom_lat)来计算纬度范围的实际跨度

        Returns:
            float: 纬度范围的差值，单位与输入的纬度单位保持一致（通常为度）
        """
        return self.top_lat - self.bottom_lat

    def longitude_extent(self):
        """
        计算并返回经度范围的跨度

        Returns:
            float: 当前对象左右经度的差值（right_lon - left_lon）
            表示地理空间数据的东西向覆盖范围
        """
        return self.right_lon - self.left_lon

    def max_latitude_extent_km(self):
        """
        计算地理区域的最大纬度跨度（千米）

        通过计算左右经度边界上的南北距离，返回最大的纬度跨度值。
        使用WGS84椭球体模型进行地理距离计算。

        Returns:
            float: 最大纬度跨度（千米），取左右边界计算结果的最大值
        """
        # 计算左经度边界的南北距离（底部纬度到顶部纬度）
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.top_lat, self.left_lon,
                                    outmask=Geodesic.DISTANCE)
        # 计算右经度边界的南北距离（底部纬度到顶部纬度）
        d2 = Geodesic.WGS84.Inverse(self.bottom_lat, self.right_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        # 返回两个边界的最大跨度值（转换为千米）
        return max(d1["s12"] / 1000.0, d2["s12"] / 1000.0)

    def max_longitude_extent_km(self):
        """
        计算地理区域在经度方向上的最大延伸距离（公里）

        通过计算底部纬度和顶部纬度对应的东西边界距离，取两者中的最大值作为结果。
        使用WGS84椭球体模型进行地理距离计算，结果转换为公里单位。

        Returns:
            float: 经度方向最大延伸距离（单位：公里）
        """
        # 计算底部纬度对应的东西最大距离（self.left_lon到self.right_lon）
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.bottom_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        # 计算顶部纬度对应的东西最大距离（self.left_lon到self.right_lon）
        d2 = Geodesic.WGS84.Inverse(self.top_lat, self.left_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        # 返回两个计算结果中的最大值（转换为公里）
        return max(d1["s12"] / 1000.0, d2["s12"] / 1000.0)

    def relative_position(self, lat, lon):
        """
        计算给定地理坐标相对于当前范围的归一化位置
        
        Args:
            lat (float): 输入纬度坐标，单位与类实例中保存的坐标一致
            lon (float): 输入经度坐标，单位与类实例中保存的坐标一致
            
        Returns:
            tuple[float, float]: 包含两个元素的元组，分别表示：
                - 纬度相对位置（经过0-1范围的裁剪）
                - 经度相对位置（经过0-1范围的裁剪）
        """
        # 计算纬度相对位置：当前坐标与下边界的差值与纬度范围的比例
        lat_rel = (lat - self.bottom_lat) / self.latitude_extent()
        # 计算经度相对位置：当前坐标与左边界的差值与经度范围的比例
        lon_rel = (lon - self.left_lon) / self.longitude_extent()
        # 对相对位置进行数值裁剪，确保结果在[0,1]区间内
        return np.clip(lat_rel, 0, 1), np.clip(lon_rel, 0, 1)

    def absolute_position(self, lat_rel, lon_rel):
        """
        将相对经纬度比例转换为绝对地理坐标
        
        通过相对比例参数计算在指定地理范围内的实际坐标位置。
        纬度计算：相对比例 × 纬度范围 + 区域最南端纬度
        经度计算：相对比例 × 经度范围 + 区域最西端经度

        Args:
            lat_rel (float): 纬度相对比例，取值范围[0.0, 1.0]，0表示最南端，1表示最北端
            lon_rel (float): 经度相对比例，取值范围[0.0, 1.0]，0表示最西端，1表示最东端

        Returns:
            tuple: 包含绝对地理坐标的元组 (latitude, longitude)
        """
        lat = lat_rel * self.latitude_extent() + self.bottom_lat
        lon = lon_rel * self.longitude_extent() + self.left_lon
        return lat, lon

    def in_boundary(self, lat, lon):
        """
        检查给定经纬度是否位于边界范围内
        
        参数:
        self (object): 包含边界坐标的对象实例
        lat (float): 要检查的纬度值
        lon (float): 要检查的经度值

        返回:
        bool: 当经度在左右边界之间且纬度在上下边界之间时返回True，否则返回False
        """
        return self.left_lon <= lon <= self.right_lon and self.bottom_lat <= lat <= self.top_lat
