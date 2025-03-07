"""
    ScenarioPlotter produces a graphical representation of a war scenario
"""

import io
import math
from collections import namedtuple
from typing import List, Tuple, Optional

import cairo
import cartopy
import matplotlib.pyplot as plt

import sys
import os
# 将当前脚本的父级目录添加到Python模块搜索路径中
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits

ColorRGBA = namedtuple('ColorRGBA', ['red', 'green', 'blue', 'alpha'])


class PlotConfig:
    """
    绘图配置类，用于管理可视化组件的样式参数

    属性配置说明：
        - 显示相关：控制网格显示及坐标系比例
        - 颜色配置：定义背景和边框的基础颜色方案
        - 文本渲染：配置精灵信息和状态消息的字体样式参数

    Attributes:
        show_grid (bool): 是否显示背景网格，默认True
        units_scale (int): 坐标系单位与像素的换算比例，默认35像素/单位
        background_color (str): 画布背景十六进制颜色值，默认#191b24
        borders_color (str): 边框十六进制颜色值，默认#ffffff
        sprites_info_font (str): 精灵信息字体名称，默认sans-serif
        sprites_info_font_style: 精灵信息字体倾斜样式，默认正常字体
        sprites_info_font_size (int): 精灵信息字号，默认12pt
        sprites_info_spacing (int): 精灵信息行间距，默认26像素
        status_message_font (str): 状态信息字体名称，默认sans-serif
        status_message_font_style: 状态信息字体倾斜样式
        status_message_font_size (int): 状态信息字号，默认14pt
    """
    def __init__(self):
        # 显示配置初始化
        self.show_grid = True
        self.units_scale = 35
        # 颜色配置初始化
        self.background_color = '#191b24'
        self.borders_color = '#ffffff'
        # 精灵信息文本配置
        self.sprites_info_font = "sans-serif"
        self.sprites_info_font_style = cairo.FONT_SLANT_NORMAL
        self.sprites_info_font_size = 12
        self.sprites_info_spacing = 26
        # 状态消息文本配置
        self.status_message_font = "sans-serif"
        self.status_message_font_style = cairo.FONT_SLANT_NORMAL
        self.status_message_font_size = 14


class Drawable:
    """地图可绘制对象的基类
    
    属性:
        zorder (int): 绘制层级控制参数，数值大的元素会覆盖在数值小的元素上层
    """
    def __init__(self, zorder):
        """ Base class for anything that can be drawn on a map

        :param zorder: z-ordering position for drawing
        """
        self.zorder = zorder


class StatusMessage(Drawable):
    def __init__(self, text, text_color=ColorRGBA(1, 1, 1, 1), zorder: int = 0):
        """在界面左下角显示状态消息
        
        参数说明:
        text : str
            需要显示的状态文本内容
        text_color : ColorRGBA
            文本颜色配置，使用RGBA颜色空间，默认值为白色不透明
        zorder : int
            绘制层级控制参数，数值越大显示在越上层

        功能特性:
        - 继承Drawable基类的绘制能力
        - 通过zorder参数实现图层叠加控制
        - 支持自定义颜色和文本内容
        """
        """ Shows a message in the bottom left position """
        super().__init__(zorder)
        self.text = text
        self.text_color = text_color


class TopLeftMessage(Drawable):
    def __init__(self, text, text_color=ColorRGBA(1, 1, 1, 1), zorder: int = 0):
        """
        在屏幕左上角显示文本消息的可绘制对象
        
        参数说明：
        text: str
            需要显示的文本内容，支持多行文本
        text_color: ColorRGBA, 可选
            文本颜色，使用RGBA颜色格式（默认白色不透明）
            各分量范围为[0.0, 1.0]，默认值(1,1,1,1)表示纯白色
        zorder: int, 可选
            绘制层级，数值越大显示在越上层（默认0）
        """
        """ Shows a message in the top left position """
        super().__init__(zorder)
        # 初始化消息文本属性
        self.text = text
        # 设置颜色属性，保留默认值或使用自定义颜色
        self.text_color = text_color


class PolyLine(Drawable):
    def __init__(self, points: List[Tuple[float, float]], line_width: float = 1.0,
                 dash: Optional[Tuple[float, float]] = None, edge_color=ColorRGBA(1, 1, 1, 1), zorder: int = 0):
        """
        表示由多个线段组成的折线图形

        Args:
            points: 折线顶点坐标列表，格式为[(x0,y0), (x1,y1), ...]
            line_width: 线条宽度（单位：像素），默认1.0像素宽
            dash: 虚线模式配置，格式为(实线长度, 空白长度)，例如(5.0, 2.0)表示
                5单位实线接2单位空白的重复模式。None表示使用实线
            edge_color: 线条颜色，使用ColorRGBA类型表示RGBA颜色值
                (红,绿,蓝,透明度)，默认纯白色不透明
            zorder: 图形元素的绘制层级，数值大的元素会覆盖在数值小的元素之上

        Note:
            继承自Drawable基类，通过zorder参数控制绘制顺序
        """
        """ Draw a series of lines """
        super().__init__(zorder)
        self.points = points
        self.line_width = line_width
        self.dash = dash
        self.edge_color = edge_color


class Rect(Drawable):
    def __init__(self, left_lon: float, bottom_lat: float, right_lon: float, top_lat: float, line_width: float = 1.0,
                 edge_color=ColorRGBA(1, 1, 1, 1), fill_color=ColorRGBA(1, 1, 1, 0), zorder: int = 0):
        """
        代表一个可绘制的矩形对象

        Args:
            left_lon: 矩形左侧经度坐标（最小经度值）
            bottom_lat: 矩形底部纬度坐标（最小纬度值）
            right_lon: 矩形右侧经度坐标（最大经度值）
            top_lat: 矩形顶部纬度坐标（最大纬度值）
            line_width: 边框线宽，默认1.0像素
            edge_color: 边框颜色，默认白色不透明（RGBA格式）
            fill_color: 填充颜色，默认透明（RGBA格式）
            zorder: 绘制层级，数值越大显示在越上层，默认0
        """
        """ Draw a square """
        super().__init__(zorder)
        # 地理坐标初始化
        self.left_lon = left_lon
        self.bottom_lat = bottom_lat
        self.right_lon = right_lon
        self.top_lat = top_lat
        # 样式属性初始化
        self.line_width = line_width
        self.edge_color = edge_color
        self.fill_color = fill_color


class Arc(Drawable):
    def __init__(self, center_lat: float, center_lon: float, radius: float, angle1: float, angle2: float,
                 line_width: float = 1.0, dash: Optional[Tuple[float, float]] = None, edge_color=None,
                 fill_color=None, zorder: int = 0):
        """ 
        初始化圆弧图形对象
        
        Args:
            center_lat: 圆弧中心点的纬度坐标 (单位：度)
            center_lon: 圆弧中心点的经度坐标 (单位：度)
            radius: 圆弧的半径 (单位：米)
            angle1: 圆弧起始角度 (单位：度，正东方向为0度，逆时针方向计算)
            angle2: 圆弧结束角度 (单位：度)
            line_width: 边界线宽度 (默认1.0)
            dash: 虚线模式元组，格式为(实线长度, 间隔长度) 
            edge_color: 边界颜色 (RGB/RGBA元组或颜色名称字符串)
            fill_color: 填充颜色 (RGB/RGBA元组或颜色名称字符串)
            zorder: 图层绘制顺序，数值越大显示在越上层
        """
        """ Draw a square """
        super().__init__(zorder)
        # 初始化地理位置参数
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        # 初始化角度参数
        self.angle1 = angle1
        self.angle2 = angle2
        # 初始化图形样式参数
        self.line_width = line_width
        self.dash = dash
        self.edge_color = edge_color
        self.fill_color = fill_color
        # 初始化图层顺序参数（通过父类构造函数已设置）
        self.zorder = zorder


class Sprite(Drawable):
    def __init__(self, lat: float, lon: float, heading: float, edge_color=ColorRGBA(1, 1, 1, 1),
                 fill_color=ColorRGBA(.5, .5, .5, 1), info_text: Optional[str] = None, zorder: int = 0):
        """ Base class for all the drawable shapes

        :param lat: latitude [-90, +90]
        :param lon: longitude [-180, +180]
        :param heading: [0, 360)
        :param edge_color: the outline color
        :param fill_color: the fill color
        :param info_text: if present, the text to be displayed under the shape
        """
        """可绘制精灵的基类，定义地图元素的通用属性和行为

        参数:
            lat (float): 纬度坐标，取值范围[-90, +90]
            lon (float): 经度坐标，取值范围[-180, +180]
            heading (float): 朝向角度，以正北为0度顺时针旋转[0, 360)
            edge_color (ColorRGBA): 轮廓颜色，RGBA格式，默认白色(1,1,1,1)
            fill_color (ColorRGBA): 填充颜色，RGBA格式，默认灰色(0.5,0.5,0.5,1)
            info_text (Optional[str]): 关联信息文本，若存在则显示在图形下方
            zorder (int): 绘制层级，数值越大显示越靠前
        """
        super().__init__(zorder)
        # 地理坐标属性
        self.lat = lat
        self.lon = lon
        # 方向属性
        self.heading = heading
        # 图形样式属性
        self.edge_color = edge_color
        self.fill_color = fill_color
        # 信息展示属性
        self.info_text = info_text


class Airplane(Sprite):
    def __init__(self, lat: float, lon: float, heading: float, edge_color='#ffffff', fill_color='#888888',
                 info_text: Optional[str] = None, zorder: int = 0):
        """
        飞机图形精灵类构造函数

        Args:
            lat: 纬度坐标（单位：度），类型为浮点数
            lon: 经度坐标（单位：度），类型为浮点数
            heading: 飞机航向角（单位：度），0表示正北，顺时针方向增加，类型为浮点数
            edge_color: 飞机轮廓颜色（十六进制字符串），默认为白色#ffffff
            fill_color: 飞机填充颜色（十六进制字符串），默认为灰色#888888
            info_text: 可选的信息文本内容，用于显示附加信息，默认为None
            zorder: 图层渲染顺序，数值大的元素会覆盖在数值小的元素之上，默认为0

        Returns:
            None: 构造函数不返回任何值
        """
        """ An airplane shaped sprite """
        super().__init__(lat, lon, heading, edge_color, fill_color, info_text, zorder)


class SamBattery(Sprite):
    def __init__(self, lat: float, lon: float, heading: float, missile_range_km: float, radar_range_km: float,
                 radar_amplitude_deg: float, edge_color='#ffffff', fill_color='#888888',
                 info_text: Optional[str] = None, zorder: int = 0):
        """ 
        SAM防空导弹阵地可视化精灵

        Args:
            lat (float): 纬度坐标，单位：度
            lon (float): 经度坐标，单位：度
            heading (float): 朝向角度，单位：度
            missile_range_km (float): 导弹射程，单位：公里
            radar_range_km (float): 雷达探测范围，单位：公里
            radar_amplitude_deg (float): 雷达扇区角度，单位：度
            edge_color (str): 轮廓颜色，HEX格式，默认白色
            fill_color (str): 填充颜色，HEX格式，默认灰色
            info_text (Optional[str]): 附加信息文本，默认无
            zorder (int): 绘制层级，数值越大显示越靠前，默认0
        """
        """ An SAM battery shaped sprite """
        super().__init__(lat, lon, heading, edge_color, fill_color, info_text, zorder)
        # 初始化武器系统参数
        self.missile_range_km = missile_range_km
        self.radar_range_km = radar_range_km
        self.radar_amplitude_deg = radar_amplitude_deg


class Missile(Sprite):
    def __init__(self, lat: float, lon: float, heading: float, edge_color='#ffffff', fill_color='#888888',
                 info_text: Optional[str] = None, zorder: int = 0):
        """
        初始化导弹精灵对象

        Args:
            lat: 导弹的初始纬度坐标（十进制度数）
            lon: 导弹的初始经度坐标（十进制度数）
            heading: 导弹的初始航向角（度，0-360）
            edge_color: 导弹轮廓颜色（十六进制颜色码）
            fill_color: 导弹填充颜色（十六进制颜色码）
            info_text: 可选的附加信息文本，用于显示导弹状态或标识
            zorder: 渲染层级顺序，数值大的对象会覆盖在数值小的对象之上

        Note:
            继承自 Sprite 基类，通过父类构造函数初始化基础属性
        """
        """ An missile shaped sprite """
        super().__init__(lat, lon, heading, edge_color, fill_color, info_text, zorder)


class Waypoint(Sprite):
    def __init__(self, lat: float, lon: float, edge_color='#ffffff', fill_color='#888888',
                 info_text: Optional[str] = None, zorder: int = 0):
        """
        表示地图路径点的精灵类，继承自Sprite基类
        
        Args:
            lat: 纬度坐标(浮点型)
            lon: 经度坐标(浮点型)
            edge_color: 路径点边框颜色(十六进制字符串，默认'#ffffff')
            fill_color: 路径点填充颜色(十六进制字符串，默认'#888888')
            info_text: 可选的附加信息文本(字符串，默认None)
            zorder: 渲染层级控制参数(整型，默认0表示基础层级)

        Note:
            继承父类Sprite的初始化逻辑，设置路径点的位置、颜色和层级属性
        """
        """ An waypoint shaped sprite """
        super().__init__(lat, lon, 0, edge_color, fill_color, info_text, zorder)


class BackgroundMesh:
    def __init__(self, lons, lats, vals, cmap: str, vmin: float = None, vmax: float = None):
        """
        初始化背景网格对象

        Args:
            lons (array): 经度坐标数组，定义网格的经度位置
            lats (array): 纬度坐标数组，定义网格的纬度位置
            vals (array): 网格点对应的数值数组，用于颜色映射
            cmap (str): 颜色映射表名称（例如'viridis', 'jet'等）
            vmin (float, optional): 颜色映射的最小值边界，默认取数据最小值
            vmax (float, optional): 颜色映射的最大值边界，默认取数据最大值
        """
        """ Class to define a background mesh """
        self.lons = lons
        self.lats = lats
        self.vals = vals
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax


class ScenarioPlotter:
    """场景绘图器，用于生成包含多种军事元素的战略态势图

    Attributes:
        map_extents (MapLimits): 地图边界限制对象
        dpi (int): 输出图像分辨率（默认200）
        bg_mesh (Optional[BackgroundMesh]): 背景网格数据
        cfg (PlotConfig): 绘图配置对象
        projection: 墨卡托投影坐标系
        _background: 生成的背景图像表面
        img_width (int): 背景图像宽度
        img_height (int): 背景图像高度
        pixels_km (float): 每公里对应的像素数
    """
    def __init__(self, map_extents: MapLimits, dpi=200, background_mesh: Optional[BackgroundMesh] = None,
                 config=PlotConfig()):
        """初始化场景绘图器

        Args:
            map_extents: 地图边界限制对象，包含经纬度范围信息
            dpi: 图像分辨率（点/英寸）
            background_mesh: 可选背景网格数据
            config: 绘图配置对象
        """
        # 初始化基础属性
        self.map_extents = map_extents
        self.dpi = dpi
        self.bg_mesh = background_mesh
        self.cfg = config
        # 创建墨卡托投影坐标系
        self.projection = cartopy.crs.Mercator(central_longitude=(map_extents.left_lon + map_extents.right_lon) / 2)
        # 构建背景图像
        self._background = self._build_background_image()
        # 计算图像尺寸和比例尺
        self.img_width = self._background.get_width()
        self.img_height = self._background.get_height()
        self.pixels_km = self.img_width / map_extents.max_longitude_extent_km()

    def _build_background_image(self):
        """生成背景地图图像

        步骤：
        1. 创建matplotlib图形和坐标系
        2. 添加背景网格数据（如果存在）
        3. 设置地图边界、背景色、海岸线等基础元素
        4. 导出为Cairo兼容的图像表面

        Returns:
            cairo.ImageSurface: 生成的背景图像表面
        """
        # Produce a background map
        # 创建matplotlib图形和坐标系
        plt.figure()
        ax = plt.axes(projection=self.projection)
        ax.set_extent((self.map_extents.left_lon, self.map_extents.right_lon,
                       self.map_extents.bottom_lat, self.map_extents.top_lat))
        # 添加背景网格数据
        if self.bg_mesh is not None:
            ax.pcolormesh(self.bg_mesh.lons, self.bg_mesh.lats, self.bg_mesh.vals,
                          vmin=self.bg_mesh.vmin, vmax=self.bg_mesh.vmax,
                          cmap=self.bg_mesh.cmap, shading='nearest', transform=cartopy.crs.PlateCarree())
        # 设置基础地图元素
        ax.patch.set_facecolor(self.cfg.background_color)
        ax.add_feature(cartopy.feature.BORDERS, edgecolor=self.cfg.borders_color, linewidth=0.2, linestyle='-',
                       alpha=1)
        ax.coastlines(resolution='110m')
        # ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.5, 0.5, 0.5))
        # 添加网格线（根据配置）
        if self.cfg.show_grid:
            ax.gridlines(linewidth=0.2)

        # Make it a cairo image
        # 导出到内存缓冲区并转换为Cairo表面
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return cairo.ImageSurface.create_from_png(buf)

    def to_png(self, filename: str, objects: List[Drawable]):
        """生成最终PNG图像

        Args:
            filename: 输出文件名
            objects: 需要绘制的军事对象列表

        处理流程：
        1. 创建Cairo绘图上下文
        2. 绘制背景图像
        3. 坐标系统转换（笛卡尔坐标系）
        4. 遍历绘制所有军事对象
        5. 保存为PNG文件
        """
        # 初始化Cairo绘图表面
        # Setup Cairo surface
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.img_width, self.img_height)
        ctx = cairo.Context(surface)

        # Draw the background
        # 绘制背景
        ctx.set_source_surface(self._background)
        ctx.paint()

        # Put the origin to left-bottom as in a Cartesian axis
        # 坐标系转换：左上角原点转为左下角原点
        ctx.translate(0, surface.get_height())
        ctx.scale(1, -1)

        # Draw objects
        # 绘制所有军事对象
        for o in objects:
            if isinstance(o, Airplane):
                self._draw_airplane(ctx, o)
            elif isinstance(o, SamBattery):
                self._draw_sam_battery(ctx, o)
            elif isinstance(o, Waypoint):
                self._draw_waypoint(ctx, o)
            elif isinstance(o, Missile):
                self._draw_missile(ctx, o)
            elif isinstance(o, StatusMessage):
                self._draw_status_message(ctx, o)
            elif isinstance(o, TopLeftMessage):
                self._draw_top_left_message(ctx, o)
            elif isinstance(o, PolyLine):
                self._draw_poly_line(ctx, o)
            elif isinstance(o, Arc):
                self._draw_arc(ctx, o)
            elif isinstance(o, Rect):
                self._draw_rect(ctx, o)
            else:
                raise RuntimeError(f"Can't draw object of type {type(o)}")

        # Write the png
        # 输出到文件
        surface.write_to_png(filename)

    @staticmethod
    def _get_image_angle(heading: float):
        """将航向角转换为Cairo旋转角度（弧度）

        Args:
            heading: 地理航向角度（0-360度）

        Returns:
            旋转弧度值（顺时针方向）
        """
        return -heading / 180 * math.pi

    def _get_image_xya(self, lat: float, lon: float, heading: float):
        """
        将地理坐标转换为图像坐标系中的坐标及旋转角度
        
        参数:
        lat (float): 输入纬度坐标，单位度
        lon (float): 输入经度坐标，单位度
        heading (float): 方向角，单位度，正北为0度顺时针增加
        
        返回:
        tuple: 包含三个元素的元组
            - x (float): 图像坐标系中的横向坐标（像素单位）
            - y (float): 图像坐标系中的纵向坐标（像素单位）
            - a (float): 旋转角度（弧度单位，顺时针为正方向）
        """
        # 计算纬度/经度在图像范围内的归一化相对位置
        lat_rel = (lat - self.map_extents.bottom_lat) / (self.map_extents.top_lat - self.map_extents.bottom_lat)
        lon_rel = (lon - self.map_extents.left_lon) / (self.map_extents.right_lon - self.map_extents.left_lon)
        # 将航向角转换为弧度并取反（顺时针方向转为数学正方向）
        a = -heading / 180 * math.pi
        # 将相对位置转换为实际图像像素坐标
        return lon_rel * self.img_width, lat_rel * self.img_height, a

    def _get_image_distance(self, dst_meters: float):
        """根据地理距离计算对应的图像像素距离
        
        通过地理坐标系的正向解算方法，将实际地理距离转换为地图图像中的垂直像素距离。
        该计算基于地图底部纬度的比例关系实现。

        Args:
            dst_meters: 需要转换的实际地理距离（米），正数表示向北方向的距离

        Returns:
            float: 对应的图像垂直方向像素距离值。正值表示图像坐标系中向上方向的距离

        """
        # 计算从地图底部纬度向北移动指定米数后的新坐标点
        d1 = geodetic_direct(self.map_extents.bottom_lat, self.map_extents.left_lon, 0, dst_meters)
        # 将纬度变化量转换为相对于地图垂直范围的百分比
        lat_rel = (d1[0] - self.map_extents.bottom_lat) / (self.map_extents.top_lat - self.map_extents.bottom_lat)
        # 将纬度比例转换为实际像素高度
        return lat_rel * self.img_height

    def _draw_status_message(self, ctx, o: StatusMessage):
        """
        绘制状态消息到指定绘图上下文
        
        参数：
            ctx (cairo.Context): Cairo绘图上下文对象，用于执行绘制操作
            o (StatusMessage): 状态消息对象，包含以下属性：
                text (str): 需要显示的文本内容
                text_color (tuple): 文本颜色RGBA元组，取值范围0.0-1.0
        
        说明：
            1. 使用配置的字体、字号和样式进行文本渲染
            2. 文本固定定位在画布左上角(10,8)坐标位置
            3. 自动保存和恢复绘图上下文状态以保持环境清洁
        """
        msg = f"> {o.text}"
        # 保存绘图上下文原始状态
        ctx.save()
        # 配置文本颜色（使用状态消息对象的RGBA值）
        ctx.set_source_rgba(*o.text_color)
        # 设置字体类型和样式（从配置参数获取）
        ctx.select_font_face(self.cfg.status_message_font, self.cfg.status_message_font_style)
        # 创建字体变换矩阵（xx控制水平缩放，yy控制垂直缩放及方向）
        ctx.set_font_matrix(
            cairo.Matrix(xx=self.cfg.status_message_font_size, yy=-self.cfg.status_message_font_size))
        # 定位绘制起点并渲染文本
        # x_bearing, y_bearing, width, height = ctx.text_extents(msg)[:4]
        ctx.move_to(10, 8)
        ctx.show_text(msg)
        # 清理绘图路径并恢复上下文原始状态
        ctx.new_path()
        ctx.restore()

    def _draw_top_left_message(self, ctx, o: TopLeftMessage):
        """
        在画布左上角绘制状态消息
        
        Args:
            ctx: Cairo绘图上下文对象
            o: TopLeftMessage类型对象，包含需要绘制的消息内容及样式参数
                - text: 要显示的文本内容
                - text_color: 文本颜色(RGBA元组)
        Returns:
            None
        """
        # 构造显示文本内容
        msg = f"{o.text}"
        # 保存绘图上下文状态
        ctx.save()
        # 设置文本颜色和字体参数
        ctx.set_source_rgba(*o.text_color)
        ctx.select_font_face(self.cfg.status_message_font, self.cfg.status_message_font_style)
        ctx.set_font_matrix(
            cairo.Matrix(xx=self.cfg.status_message_font_size, yy=-self.cfg.status_message_font_size))
        # 计算文本布局尺寸
        x_bearing, y_bearing, width, height = ctx.text_extents(msg)[:4]
        # 定位到画布左上角区域（右侧留10像素边距，底部留5像素边距）
        ctx.move_to(self.img_width - width - 10, self.img_height - height - 5)
        # 绘制文本并恢复上下文状态
        ctx.show_text(msg)
        ctx.new_path()
        ctx.restore()

    # Draw status text (not rotated)
    def _draw_text(self, ctx, x, y, text):
        """
        在指定坐标绘制居中文本
        
        Args:
            ctx: cairo上下文对象，用于执行绘制操作
            x: int/float 文本中心点的X轴坐标
            y: int/float 文本中心点的Y轴坐标
            text: str 需要绘制的文本内容
            
        Returns:
            None
        """
        # 配置字体样式和大小
        ctx.select_font_face(self.cfg.sprites_info_font, self.cfg.sprites_info_font_style)
        # 执行文本绘制操作
        ctx.set_font_matrix(cairo.Matrix(xx=self.cfg.sprites_info_font_size, yy=-self.cfg.sprites_info_font_size))
        x_bearing, y_bearing, width, height = ctx.text_extents(text)[:4]
        ctx.move_to(x - width / 2 - x_bearing, y - height / 2 - y_bearing)
        ctx.show_text(text)

    def _draw_airplane(self, ctx, o: Airplane):
        """绘制飞机图形到指定绘图上下文
        
        Args:
            ctx: 绘图上下文对象（如Cairo上下文）
            o: 飞机对象，包含位置、航向、样式等信息
        
        Returns:
            None: 直接修改绘图上下文状态
        """
        # 计算飞机在画布上的坐标和旋转角度
        # Translate to airplane position
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.heading)
        ctx.save()
        ctx.translate(x, y)

        # 绘制非旋转状态文本（相对飞机位置）
        # Draw status text (not rotated)
        if o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)

        # 应用飞机航向旋转
        # Rotate to airplane heading
        ctx.rotate(angle)

        # 绘制飞机主体形状
        ctx.set_source_rgba(*o.fill_color)
        ctx.set_line_width(1)
        # 构建飞机轮廓路径（左右对称结构）
        # 右侧轮廓绘制
        # First half
        ctx.move_to(0.00 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.line_to(0.06 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.line_to(0.08 * self.cfg.units_scale, -0.31 * self.cfg.units_scale)
        ctx.line_to(0.28 * self.cfg.units_scale, -0.29 * self.cfg.units_scale)
        ctx.line_to(0.28 * self.cfg.units_scale, -0.19 * self.cfg.units_scale)
        ctx.line_to(0.09 * self.cfg.units_scale, 0.03 * self.cfg.units_scale)
        ctx.line_to(0.09 * self.cfg.units_scale, 0.05 * self.cfg.units_scale)
        ctx.line_to(0.13 * self.cfg.units_scale, 0.04 * self.cfg.units_scale)
        ctx.line_to(0.13 * self.cfg.units_scale, 0.08 * self.cfg.units_scale)
        ctx.line_to(0.05 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, 0.44 * self.cfg.units_scale)
        # Symmetric half
        ctx.line_to(-0.05 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(-0.13 * self.cfg.units_scale, 0.08 * self.cfg.units_scale)
        ctx.line_to(-0.13 * self.cfg.units_scale, 0.04 * self.cfg.units_scale)
        ctx.line_to(-0.09 * self.cfg.units_scale, 0.05 * self.cfg.units_scale)
        ctx.line_to(-0.09 * self.cfg.units_scale, 0.03 * self.cfg.units_scale)
        ctx.line_to(-0.28 * self.cfg.units_scale, -0.19 * self.cfg.units_scale)
        ctx.line_to(-0.28 * self.cfg.units_scale, -0.29 * self.cfg.units_scale)
        ctx.line_to(-0.08 * self.cfg.units_scale, -0.31 * self.cfg.units_scale)
        ctx.line_to(-0.06 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        # 绘制中心参考十字标记
        # Ref. point
        ctx.move_to(0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.line_to(-0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.move_to(0.0 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, -0.1 * self.cfg.units_scale)
        ctx.stroke()

        ctx.restore()

    def _draw_sam_battery(self, ctx, o: SamBattery):
        """
        绘制SAM电池单元的图形元素
        
        参数:
        ctx: 绘图上下文对象（如Cairo上下文）
        o: SamBattery实例，包含电池单元的位置、状态及配置参数
        """
        # ========== 坐标系变换 ==========
        # Translate to SAM battery position
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.heading)
        ctx.save()
        # 将坐标系平移至电池位置
        ctx.translate(x, y)

        # ========== 绘制状态文本 ==========
        # Draw status text (not rotated)
        if o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)

        # ========== 绘制电池主体 ==========
        # 应用方位角旋转
        ctx.rotate(angle)
        ctx.set_source_rgba(*o.fill_color)
        ctx.set_line_width(1)
        # 绘制方形电池图标
        # Square
        ctx.move_to(0.15 * self.cfg.units_scale, -0.15 * self.cfg.units_scale)
        ctx.line_to(0.15 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(-0.15 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(-0.15 * self.cfg.units_scale, -0.15 * self.cfg.units_scale)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        # ========== 绘制雷达指示 ==========
        # 绘制双圆弧雷达符号
        # Radar
        ctx.arc(0.0, 0.0, 0.4 * self.cfg.units_scale, -1 + math.pi / 2.0, 1 + math.pi / 2.0)
        ctx.new_sub_path()
        ctx.arc(0.0, 0.0, 0.6 * self.cfg.units_scale, -1 + math.pi / 2.0, 1 + math.pi / 2.0)
        ctx.stroke()
        # ========== 绘制参考点十字 ==========
        # Ref. point
        ctx.move_to(0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.line_to(-0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.move_to(0.0 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, -0.1 * self.cfg.units_scale)
        ctx.stroke()
        # ========== 绘制雷达范围 ==========
        # 设置虚线样式
        # Radar limit
        ctx.set_dash([5, 5])
        ctx.move_to(0, 0)
        ctx.arc(0.0, 0.0, self.pixels_km * o.radar_range_km,
                math.pi / 2.0 - o.radar_amplitude_deg / 180 * math.pi, math.pi / 2.0)
        ctx.close_path()
        ctx.stroke()
        # ========== 绘制导弹射程 ==========
        # Missile range
        ctx.arc(0.0, 0.0, self.pixels_km * o.missile_range_km, 0, 2.0 * math.pi)
        ctx.stroke()

        # 恢复原始坐标系
        ctx.restore()

    def _draw_waypoint(self, ctx, o: Waypoint):
        """
        绘制单个路径点的图形元素
        
        参数：
            ctx: 绘图上下文对象（如Cairo Context）
            o: Waypoint实例，包含路径点的坐标、角度、颜色等信息
        """
        # 坐标变换到路径点位置（包含平移和旋转）
        # Translate to waypoint position
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.heading)
        ctx.save()
        ctx.translate(x, y)

        # 绘制状态文字（不随图形旋转）
        # Draw status text (not rotated)
        if o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)

        # 绘制路径点圆形标记
        ctx.rotate(angle)
        ctx.set_line_width(1)
        # Circle
        ctx.new_path()
        ctx.arc(0.0, 0.0, 0.1 * self.cfg.units_scale, 0, 2 * math.pi)
        ctx.close_path()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        # 绘制路径点旗帜标记
        # Flag
        ctx.move_to(0.0, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0, 0.25 * self.cfg.units_scale)
        ctx.stroke()
        ctx.move_to(0.0, 0.25 * self.cfg.units_scale)
        ctx.line_to(0.0, 0.45 * self.cfg.units_scale)
        ctx.line_to(0.25 * self.cfg.units_scale, 0.35 * self.cfg.units_scale)
        ctx.close_path()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()

        ctx.restore()

    def _draw_missile(self, ctx, o: Missile):
        """
        绘制导弹图形到指定绘图上下文
        
        Args:
            ctx: 绘图上下文对象（如Cairo Context）
            o: Missile对象，包含导弹的位置、颜色、状态信息等绘制参数
        Returns:
            None 无返回值
        """
        # 转换坐标系到导弹目标点位置
        # Translate to waypoint position
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.heading)
        ctx.save()
        ctx.translate(x, y)

        # 绘制未旋转的状态文本（位于导弹图形上方）
        # Draw status text (not rotated)
        if o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)

        # 旋转坐标系后绘制导弹图形
        ctx.rotate(angle)
        ctx.set_line_width(1)
        # 构建导弹多边形路径（箭头形状）
        # Body
        ctx.move_to(0.05 * self.cfg.units_scale, -0.3 * self.cfg.units_scale)
        ctx.line_to(0.07 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, 0.4 * self.cfg.units_scale)
        # 对称绘制另一侧
        # Sym
        ctx.line_to(-0.07 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(-0.05 * self.cfg.units_scale, -0.3 * self.cfg.units_scale)
        ctx.close_path()
        # 填充主体颜色并绘制边框
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()

        ctx.restore()

    def _draw_poly_line(self, ctx, o: PolyLine):
        """
        绘制多段线图形
        
        参数:
        ctx: 绘图上下文对象，用于执行底层绘图操作
        o: PolyLine实例，包含多段线的配置参数和坐标点数据
        
        返回值:
        None
        """
        # 顶点数量不足时直接返回
        if len(o.points) <= 1:
            return

        # 保存当前绘图上下文状态
        ctx.save()
        # 配置线条属性
        ctx.set_line_width(o.line_width)
        if o.dash:
            ctx.set_dash(o.dash)
        ctx.set_source_rgba(*o.edge_color)
        # 构建多段线路径
        ctx.new_path()
        x, y, a = self._get_image_xya(o.points[0][0], o.points[0][1], 0)
        ctx.move_to(x, y)
        # 迭代添加线段顶点
        for i in range(1, len(o.points)):
            x, y, a = self._get_image_xya(o.points[i][0], o.points[i][1], 0)
            ctx.line_to(x, y)
        # 执行描边操作并恢复上下文状态
        ctx.stroke()
        ctx.restore()

    def _draw_rect(self, ctx, o: Rect):
        """在绘图上下文上绘制矩形形状对象
        
        Args:
            ctx: 绘图上下文对象（如cairo.Context）
            o: Rect类型，包含矩形的位置、样式配置参数
                - top_lat/left_lon: 左上角地理坐标
                - bottom_lat/right_lon: 右下角地理坐标
                - line_width: 边框线宽
                - fill_color: 填充颜色RGBA值
                - edge_color: 边框颜色RGBA值
        
        Returns:
            None
        """
        # 保存当前绘图上下文状态
        ctx.save()

        # 将地理坐标转换为图像坐标，并计算矩形宽高
        left, top, _ = self._get_image_xya(o.top_lat, o.left_lon, 0)
        right, bottom, _ = self._get_image_xya(o.bottom_lat, o.right_lon, 0)
        ctx.rectangle(left, top, right - left, bottom - top)

        # 应用样式设置：先填充后描边（保留填充区域进行描边）
        ctx.set_line_width(o.line_width)
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        # 恢复原始绘图上下文状态
        ctx.restore()

    def _draw_arc(self, ctx, o: Arc):
        """
        绘制圆弧图形

        参数:
        ctx: 图形上下文对象（如Cairo上下文）
        o (Arc): 圆弧对象，包含以下属性：
            - center_lat: 圆心纬度坐标
            - center_lon: 圆心经度坐标
            - angle1: 起始角度（单位：度）
            - angle2: 终止角度（单位：度）
            - radius: 圆弧半径
            - line_width: 线宽
            - dash: 虚线样式
            - fill_color: 填充颜色
            - edge_color: 边框颜色

        返回值:
        None: 无返回值
        """
        # 保存上下文状态并创建新路径
        ctx.save()

        # 转换地理坐标到画布坐标并计算角度
        x, y, a1 = self._get_image_xya(o.center_lat, o.center_lon, o.angle1 - 90)
        a2 = self._get_image_angle(o.angle2 - 90)
        ctx.new_path()
        # 绘制圆弧路径（自动处理角度顺序）
        ctx.arc(x, y, self._get_image_distance(o.radius), min(a1, a2), max(a1, a2))

        # 设置图形样式参数
        ctx.set_line_width(o.line_width)
        if o.dash:
            ctx.set_dash(o.dash)
        # 填充和描边处理
        if o.fill_color:
            ctx.set_source_rgba(*o.fill_color)
            # 保留路径用于后续描边
            ctx.fill_preserve()
        if o.edge_color:
            ctx.set_source_rgba(*o.edge_color)
            ctx.stroke()
        # 恢复原始上下文状态
        ctx.restore()
