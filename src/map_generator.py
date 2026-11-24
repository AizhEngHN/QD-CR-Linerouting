"""
地图生成模块
============

功能说明：
    生成300x300网格地图，用于电力输电线路规划仿真实验。
    地图包含硬约束区域（禁止区域）和软约束区域（惩罚区域）。

编码规则：
    - -1: 硬约束（禁止区域，不能布线也不能布塔）
    - 0: 无约束（正常区域，可以自由布线布塔）
    - 0 < value <= 1: 软约束代价（值越大，在该位置布塔的惩罚越高）

作者：[Your Name]
日期：2026-01
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

# 导入绑图库并设置中文字体
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


@dataclass
class MapConfig:
    """
    地图配置参数类
    
    使用dataclass装饰器自动生成__init__等方法，
    用于存储地图生成的所有可配置参数。
    
    属性说明：
        width_km: 地图宽度，单位为公里
        height_km: 地图高度，单位为公里
        resolution_m: 网格分辨率，单位为米（每个网格的边长）
        
        num_hard_zones: 硬约束区域的数量
        hard_zone_min_size: 硬约束区域的最小尺寸（网格数）
        hard_zone_max_size: 硬约束区域的最大尺寸（网格数）
        
        num_soft_zones: 软约束区域的数量
        soft_zone_min_size: 软约束区域的最小尺寸（网格数）
        soft_zone_max_size: 软约束区域的最大尺寸（网格数）
        soft_cost_min: 软约束的最小代价值
        soft_cost_max: 软约束的最大代价值
        
        substation_a: 变电站A的位置坐标（网格坐标）
        substation_b: 变电站B的位置坐标（网格坐标）
        
        seed: 随机数种子，用于复现相同的地图
    """
    # 地图基本参数
    width_km: float = 30.0          # 地图宽度(km)
    height_km: float = 30.0         # 地图高度(km)
    resolution_m: float = 200.0     # 网格分辨率(m)，即每个格子代表200m x 200m
    
    # 硬约束区域参数（住民区、机场等，规整的矩形或圆形）
    num_hard_zones: int = 2         # 硬约束区域数量
    hard_coverage: float = 0.3      # 硬约束目标覆盖率
    
    # 软约束区域参数（地形约束如河流、山坡，不规则形状）
    num_soft_zones: int = 8         # 软约束区域数量
    soft_coverage: float = 0.4      # 软约束目标覆盖率
    soft_cost_min: float = 0.2      # 软约束最小代价（0-1之间）
    soft_cost_max: float = 0.9      # 软约束最大代价（0-1之间）
    
    # 变电站位置（网格坐标，原点在左下角）
    substation_a: Tuple[int, int] = (10, 10)      # 变电站A位置（起点）
    substation_b: Tuple[int, int] = (140, 140)    # 变电站B位置（终点）
    
    # 随机种子（设置后可以复现相同的地图）
    seed: Optional[int] = None


class GridMap:
    """
    网格地图类
    
    该类负责创建、管理和操作输电线路规划所需的网格地图。
    地图使用二维numpy数组存储，每个元素代表一个100m x 100m的网格。
    
    主要功能：
        1. 初始化空白地图
        2. 随机生成硬约束和软约束区域
        3. 查询任意位置的约束类型和代价
        4. 保存和加载地图数据
        5. 可视化地图
    
    使用示例：
        >>> config = MapConfig(seed=42)
        >>> grid_map = GridMap(config)
        >>> grid_map.generate_hard_constraints()
        >>> grid_map.generate_soft_constraints()
        >>> grid_map.save('data/my_map')
    """
    
    def __init__(self, config: MapConfig = None):
        """
        初始化网格地图
        
        参数：
            config: MapConfig对象，包含地图的所有配置参数。
                   如果为None，则使用默认配置。
        """
        # 如果没有提供配置，使用默认配置
        self.config = config or MapConfig()
        
        # 根据地图尺寸和分辨率计算网格数量
        # 例如：30km / 100m = 300个网格
        self.grid_width = int(self.config.width_km * 1000 / self.config.resolution_m)
        self.grid_height = int(self.config.height_km * 1000 / self.config.resolution_m)
        
        # 初始化地图矩阵，全部填充0（表示无约束的正常区域）
        # 使用float32类型以支持0-1之间的软约束代价值
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # 保存变电站位置
        self.substation_a = self.config.substation_a
        self.substation_b = self.config.substation_b
        
        # 设置随机数种子（如果指定了的话）
        # 这样可以保证每次生成相同的地图
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
    
    def generate_hard_constraints(self) -> None:
        """
        生成硬约束区域（住民区、机场等）
        
        硬约束区域在地图矩阵中用-1表示，代表完全禁止的区域。
        生成2-3个大块，使用规整的矩形或圆形，总覆盖率约15%。
        """
        total_cells = self.grid_width * self.grid_height
        target_cells = int(total_cells * self.config.hard_coverage)
        cells_per_zone = target_cells // self.config.num_hard_zones
        
        # 计算每个区域的大致尺寸（假设正方形）
        zone_size = int(np.sqrt(cells_per_zone))
        
        for i in range(self.config.num_hard_zones):
            # 随机大小（在目标尺寸的80%-120%之间变化）
            w = int(zone_size * np.random.uniform(0.8, 1.2))
            h = int(zone_size * np.random.uniform(0.8, 1.2))
            
            # 确保尺寸有效
            w = min(w, self.grid_width - 10)
            h = min(h, self.grid_height - 10)
            
            # 尝试找到不重叠的位置
            placed = False
            for _ in range(200):
                x = np.random.randint(5, max(6, self.grid_width - w - 5))
                y = np.random.randint(5, max(6, self.grid_height - h - 5))
                
                # 检查该区域是否与已有约束重叠
                region = self.grid[y:y+h, x:x+w]
                if np.all(region == 0):
                    # 使用规整形状：矩形或椭圆
                    self._fill_hard_zone(x, y, w, h)
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place hard zone {i+1}")
        
        # 确保变电站周围没有硬约束
        self._clear_around_point(self.substation_a, radius=8)
        self._clear_around_point(self.substation_b, radius=8)
    
    def _fill_hard_zone(self, x: int, y: int, w: int, h: int) -> None:
        """
        填充硬约束区域（规整形状：矩形或椭圆）
        """
        shape_type = np.random.choice(['rect', 'ellipse'], p=[0.5, 0.5])
        
        if shape_type == 'rect':
            # 矩形（住民区、工业区）
            self.grid[y:y+h, x:x+w] = -1
        else:
            # 椭圆形（机场、体育场）
            cy, cx = y + h // 2, x + w // 2
            ry, rx = h // 2, w // 2
            for iy in range(y, min(y+h, self.grid_height)):
                for ix in range(x, min(x+w, self.grid_width)):
                    if ((ix - cx)**2 / (rx**2 + 1e-6) + (iy - cy)**2 / (ry**2 + 1e-6)) <= 1:
                        self.grid[iy, ix] = -1
    
    def generate_soft_constraints(self) -> None:
        """
        生成软约束区域（地形约束：河流、山坡等）
        
        软约束区域用0-1之间的值表示，代表增加施工难度的区域。
        生成6-10个块，使用不规则形状，总覆盖率约35%。
        """
        total_cells = self.grid_width * self.grid_height
        target_cells = int(total_cells * self.config.soft_coverage)
        cells_per_zone = target_cells // self.config.num_soft_zones
        
        # 计算每个区域的大致尺寸
        zone_size = int(np.sqrt(cells_per_zone))
        
        for i in range(self.config.num_soft_zones):
            # 随机代价值
            cost = np.random.uniform(self.config.soft_cost_min, self.config.soft_cost_max)
            
            # 随机大小（软约束可以小一些）
            w = int(zone_size * np.random.uniform(0.3, 1.4))
            h = int(zone_size * np.random.uniform(0.3, 1.4))
            
            # 确保尺寸有效
            w = min(max(w, 20), self.grid_width - 10)
            h = min(max(h, 20), self.grid_height - 10)
            
            # 尝试找到不重叠的位置（与硬约束和其他软约束都不重叠）
            placed = False
            for _ in range(200):
                x = np.random.randint(5, max(6, self.grid_width - w - 5))
                y = np.random.randint(5, max(6, self.grid_height - h - 5))
                
                # 检查该区域是否与已有约束重叠（只有全为0才可以放置）
                region = self.grid[y:y+h, x:x+w]
                if np.all(region == 0):
                    # 填充不规则形状
                    self._fill_soft_zone(x, y, w, h, cost)
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place soft zone {i+1}")
    
    def _fill_soft_zone(self, x: int, y: int, w: int, h: int, value: float) -> None:
        """
        填充软约束区域（不规则形状：河流、山坡）
        """
        shape_type = np.random.choice(['river', 'mountain', 'blob'], p=[0.35, 0.35, 0.3])
        
        if shape_type == 'river':
            self._fill_river(x, y, w, h, value)
        elif shape_type == 'mountain':
            self._fill_mountain(x, y, w, h, value)
        else:
            self._fill_blob(x, y, w, h, value)
    
    def _fill_river(self, x: int, y: int, w: int, h: int, value: float) -> None:
        """
        填充河流形状（蜿蜒的带状）
        权重从边缘到中心渐变：边缘代价小，中心代价大（模拟河流中心水深）
        只填充值为0的区域（不覆盖硬约束和已有软约束）
        """
        # 决定河流方向
        if w >= h:
            # 水平河流
            river_width = max(25, h // 2)
            center_y = y + h // 2
            amplitude = h // 3
            frequency = np.random.uniform(0.015, 0.035)
            phase = np.random.uniform(0, 2 * np.pi)
            half_width = river_width // 2
            
            for ix in range(x, min(x + w, self.grid_width)):
                wave = int(amplitude * np.sin(frequency * (ix - x) + phase))
                cy = center_y + wave
                
                for iy in range(cy - half_width, cy + half_width + 1):
                    if 0 <= iy < self.grid_height and self.grid[iy, ix] == 0:
                        # 计算到河流中心线的距离比例（0=中心，1=边缘）
                        dist_ratio = abs(iy - cy) / (half_width + 1e-6)
                        dist_ratio = min(dist_ratio, 1.0)
                        # 边缘代价小，中心代价大
                        # 中心: value, 边缘: value * 0.2
                        gradient_value = value * (1.0 - dist_ratio * 0.8)
                        gradient_value = max(0.1, gradient_value)
                        self.grid[iy, ix] = gradient_value
        else:
            # 垂直河流
            river_width = max(25, w // 2)
            center_x = x + w // 2
            amplitude = w // 3
            frequency = np.random.uniform(0.015, 0.035)
            phase = np.random.uniform(0, 2 * np.pi)
            half_width = river_width // 2
            
            for iy in range(y, min(y + h, self.grid_height)):
                wave = int(amplitude * np.sin(frequency * (iy - y) + phase))
                cx = center_x + wave
                
                for ix in range(cx - half_width, cx + half_width + 1):
                    if 0 <= ix < self.grid_width and self.grid[iy, ix] == 0:
                        # 计算到河流中心线的距离比例（0=中心，1=边缘）
                        dist_ratio = abs(ix - cx) / (half_width + 1e-6)
                        dist_ratio = min(dist_ratio, 1.0)
                        # 边缘代价小，中心代价大
                        gradient_value = value * (1.0 - dist_ratio * 0.8)
                        gradient_value = max(0.1, gradient_value)
                        self.grid[iy, ix] = gradient_value
    
    def _fill_mountain(self, x: int, y: int, w: int, h: int, value: float) -> None:
        """
        填充山坡形状（不规则多边形）
        权重从边缘到中心渐变：边缘代价小，中心代价大（模拟山峰中心海拔高）
        只填充值为0的区域（不覆盖硬约束和已有软约束）
        """
        cx, cy = x + w // 2, y + h // 2
        num_vertices = np.random.randint(8, 14)
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        
        # 生成不规则多边形的顶点
        vertices = []
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            r = np.random.uniform(0.5, 1.0)
            vx = cx + int(r * w // 2 * np.cos(angle))
            vy = cy + int(r * h // 2 * np.sin(angle))
            vertices.append((vx, vy))
        
        # 填充多边形（只填充值为0的区域）
        for iy in range(y, min(y + h, self.grid_height)):
            for ix in range(x, min(x + w, self.grid_width)):
                if self._point_in_polygon(ix, iy, vertices):
                    if self.grid[iy, ix] == 0:
                        # 计算到中心的距离比例（0=中心，1=边缘）
                        dist = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
                        dist_ratio = min(dist / (max_dist + 1e-6), 1.0)
                        # 边缘代价小，中心代价大
                        # 中心: value, 边缘: value * 0.2
                        gradient_value = value * (1.0 - dist_ratio * 0.8)
                        gradient_value = max(0.1, gradient_value)
                        self.grid[iy, ix] = gradient_value
    
    def _fill_blob(self, x: int, y: int, w: int, h: int, value: float) -> None:
        """
        填充不规则斑块（湿地、森林等）
        权重从边缘到中心渐变：边缘代价小，中心代价大
        只填充值为0的区域（不覆盖硬约束和已有软约束）
        """
        cx, cy = x + w // 2, y + h // 2
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        
        # 生成随机种子点
        num_seeds = np.random.randint(5, 10)
        seeds = [(np.random.randint(x, x + w), np.random.randint(y, y + h)) 
                 for _ in range(num_seeds)]
        
        for iy in range(y, min(y + h, self.grid_height)):
            for ix in range(x, min(x + w, self.grid_width)):
                dist_center = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
                min_seed_dist = min(np.sqrt((ix - sx) ** 2 + (iy - sy) ** 2) 
                                   for sx, sy in seeds)
                
                # 基于距离决定是否填充（只填充值为0的区域）
                if dist_center / max_dist < 0.9 and min_seed_dist < max_dist * 0.7:
                    if self.grid[iy, ix] == 0:
                        # 计算到中心的距离比例（0=中心，1=边缘）
                        dist_ratio = min(dist_center / (max_dist + 1e-6), 1.0)
                        # 边缘代价小，中心代价大
                        # 中心: value, 边缘: value * 0.2
                        gradient_value = value * (1.0 - dist_ratio * 0.8)
                        gradient_value = max(0.1, gradient_value)
                        self.grid[iy, ix] = gradient_value
    
    def _point_in_polygon(self, x: int, y: int, vertices: list) -> bool:
        """判断点是否在多边形内部（射线法）"""
        n = len(vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6) + xi):
                inside = not inside
            j = i
        return inside
    
    def _clear_around_point(self, point: Tuple[int, int], radius: int) -> None:
        """
        清除指定点周围的约束（内部方法）
        
        随机选择不同的自然形状：
        - rect: 矩形（建筑物、工厂）
        - ellipse: 椭圆形（湖泊）
        - river: 河流（蜿蜒带状）
        - mountain: 山坡（不规则多边形）
        """
        # 根据约束类型选择不同的形状分布
        if value == -1.0:  # 硬约束更多是建筑物、湖泊
            shape_type = np.random.choice(['rect', 'ellipse', 'mountain'], p=[0.4, 0.3, 0.3])
        else:  # 软约束更多是河流、山坡
            shape_type = np.random.choice(['river', 'mountain', 'ellipse'], p=[0.4, 0.35, 0.25])
        
        if shape_type == 'rect':
            self._fill_rect(x, y, w, h, value)
        elif shape_type == 'ellipse':
            self._fill_ellipse(x, y, w, h, value)
        elif shape_type == 'river':
            self._fill_river(x, y, w, h, value)
        else:  # mountain
            self._fill_mountain(x, y, w, h, value)
    
    def _fill_rect(self, x: int, y: int, w: int, h: int, value: float) -> None:
        """填充矩形区域（建筑物、工厂）"""
        y_end = min(y + h, self.grid_height)
        x_end = min(x + w, self.grid_width)
        self.grid[y:y_end, x:x_end] = value
    
    def _fill_ellipse(self, x: int, y: int, w: int, h: int, value: float) -> None:
        """填充椭圆形区域（湖泊）"""
        cy, cx = y + h // 2, x + w // 2
        ry, rx = h // 2, w // 2
        
        # 使用向量化操作加速
        yy, xx = np.ogrid[y:min(y+h, self.grid_height), x:min(x+w, self.grid_width)]
        mask = ((xx - cx)**2 / (rx**2 + 1e-6) + (yy - cy)**2 / (ry**2 + 1e-6)) <= 1
        self.grid[y:min(y+h, self.grid_height), x:min(x+w, self.grid_width)][mask] = value
    
    def _fill_river(self, x: int, y: int, w: int, h: int, value: float) -> None:
        """
        填充河流形状（蜿蜒的带状）
        使用正弦波生成蜿蜒的河流，河流更宽以增加覆盖率
        """
        # 决定河流方向：选择较长的方向
        if w >= h:
            # 水平河流
            river_width = max(12, h // 2)  # 河流宽度（加宽）
            center_y = y + h // 2
            amplitude = h // 4  # 蜿蜒幅度
            frequency = np.random.uniform(0.015, 0.04)
            phase = np.random.uniform(0, 2 * np.pi)
            
            for ix in range(x, min(x + w, self.grid_width)):
                wave = int(amplitude * np.sin(frequency * (ix - x) + phase))
                cy = center_y + wave
                
                for iy in range(cy - river_width // 2, cy + river_width // 2 + 1):
                    if 0 <= iy < self.grid_height:
                        self.grid[iy, ix] = value
        else:
            # 垂直河流
            river_width = max(12, w // 2)
            center_x = x + w // 2
            amplitude = w // 4
            frequency = np.random.uniform(0.015, 0.04)
            phase = np.random.uniform(0, 2 * np.pi)
            
            for iy in range(y, min(y + h, self.grid_height)):
                wave = int(amplitude * np.sin(frequency * (iy - y) + phase))
                cx = center_x + wave
                
                for ix in range(cx - river_width // 2, cx + river_width // 2 + 1):
                    if 0 <= ix < self.grid_width:
                        self.grid[iy, ix] = value
    
    def _fill_mountain(self, x: int, y: int, w: int, h: int, value: float) -> None:
        """
        填充山坡形状（不规则多边形）
        使用多个随机顶点生成不规则多边形
        """
        cx, cy = x + w // 2, y + h // 2
        num_vertices = np.random.randint(8, 15)
        
        # 生成不规则多边形的顶点
        vertices = []
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            # 随机半径，产生不规则边缘（0.6-1.0范围，保证较大面积）
            r = np.random.uniform(0.6, 1.0)
            vx = cx + int(r * w // 2 * np.cos(angle))
            vy = cy + int(r * h // 2 * np.sin(angle))
            vertices.append((vx, vy))
        
        # 使用扫描线算法填充多边形
        for iy in range(y, min(y + h, self.grid_height)):
            for ix in range(x, min(x + w, self.grid_width)):
                if self._point_in_polygon(ix, iy, vertices):
                    self.grid[iy, ix] = value
    
    def _point_in_polygon(self, x: int, y: int, vertices: list) -> bool:
        """判断点是否在多边形内部（射线法）"""
        n = len(vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6) + xi):
                inside = not inside
            j = i
        return inside
    
    def _clear_around_point(self, point: Tuple[int, int], radius: int) -> None:
        """
        清除指定点周围的约束（内部方法）
        
        用于确保变电站周围有足够的空间，不被约束区域覆盖。
        
        参数：
            point: 中心点坐标 (x, y)
            radius: 清除半径（网格数）
        """
        x, y = point
        
        # 计算清除范围，确保不超出地图边界
        x_min = max(0, x - radius)
        x_max = min(self.grid_width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.grid_height, y + radius + 1)
        
        # 将该范围内的所有值设为0（正常区域）
        self.grid[y_min:y_max, x_min:x_max] = 0
    
    def is_hard_constraint(self, x: int, y: int) -> bool:
        """
        检查指定位置是否为硬约束区域
        
        参数：
            x: 网格x坐标
            y: 网格y坐标
            
        返回：
            True - 该位置是硬约束（禁止区域）
            False - 该位置不是硬约束
            
        注意：
            超出地图边界的位置也被视为硬约束
        """
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.grid[y, x] == -1
        return True  # 边界外视为硬约束，不允许布线
    
    def is_soft_constraint(self, x: int, y: int) -> bool:
        """
        检查指定位置是否为软约束区域
        
        参数：
            x: 网格x坐标
            y: 网格y坐标
            
        返回：
            True - 该位置是软约束（惩罚区域）
            False - 该位置不是软约束
        """
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return 0 < self.grid[y, x] <= 1
        return False
    
    def get_cost(self, x: int, y: int) -> float:
        """
        获取指定位置的代价值
        
        参数：
            x: 网格x坐标
            y: 网格y坐标
            
        返回：
            代价值：
            - 0: 正常区域或硬约束区域（硬约束返回0是因为不应该在那里布塔）
            - 0-1: 软约束区域的代价值
        """
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return max(0, self.grid[y, x])  # 硬约束(-1)返回0，软约束返回实际值
        return 0
    
    def validate_no_overlap(self) -> dict:
        """
        验证约束区域之间是否有重叠
        
        检测规则：
            1. 每个网格只能属于一种类型：正常(0)、硬约束(-1)、软约束(0-1)
            2. 由于我们的填充逻辑只填充值为0的区域，理论上不会有重叠
            3. 此函数用于验证生成结果的正确性
        
        返回：
            dict: 包含验证结果的字典
                - is_valid: bool, 是否无重叠
                - hard_count: int, 硬约束网格数
                - soft_count: int, 软约束网格数
                - normal_count: int, 正常网格数
                - hard_ratio: float, 硬约束占比
                - soft_ratio: float, 软约束占比
                - normal_ratio: float, 正常区域占比
        """
        total_cells = self.grid.size
        
        # 统计各类型网格数量
        hard_count = np.sum(self.grid == -1)
        soft_count = np.sum((self.grid > 0) & (self.grid <= 1))
        normal_count = np.sum(self.grid == 0)
        
        # 检查是否有异常值（不在预期范围内的值）
        abnormal_count = total_cells - hard_count - soft_count - normal_count
        
        # 验证总数是否正确（如果有重叠，总数会不对）
        is_valid = (abnormal_count == 0) and (hard_count + soft_count + normal_count == total_cells)
        
        result = {
            'is_valid': is_valid,
            'hard_count': int(hard_count),
            'soft_count': int(soft_count),
            'normal_count': int(normal_count),
            'abnormal_count': int(abnormal_count),
            'hard_ratio': hard_count / total_cells,
            'soft_ratio': soft_count / total_cells,
            'normal_ratio': normal_count / total_cells,
            'total_cells': total_cells
        }
        
        return result
    
    def print_validation_report(self) -> None:
        """
        打印验证报告
        
        输出约束区域的统计信息和重叠检测结果
        """
        result = self.validate_no_overlap()
        
        print(f"\n{'='*50}")
        print(f"Constraint Overlap Validation Report")
        print(f"{'='*50}")
        print(f"  Grid size: {self.grid_width} x {self.grid_height}")
        print(f"  Total cells: {result['total_cells']:,}")
        print(f"{'='*50}")
        print(f"  Hard constraint: {result['hard_count']:,} ({result['hard_ratio']*100:.1f}%)")
        print(f"  Soft constraint: {result['soft_count']:,} ({result['soft_ratio']*100:.1f}%)")
        print(f"  Normal area:     {result['normal_count']:,} ({result['normal_ratio']*100:.1f}%)")
        print(f"{'='*50}")
        
        if result['abnormal_count'] > 0:
            print(f"  WARNING: Found {result['abnormal_count']} abnormal cells!")
        
        if result['is_valid']:
            print(f"  Status: PASSED - No overlap detected")
        else:
            print(f"  Status: FAILED - Overlap or abnormal values detected!")
        print(f"{'='*50}")
    
    def save(self, filepath: str) -> None:
        """
        保存地图到文件
        
        会生成两个文件：
            1. .csv文件：存储地图矩阵数据（CSV格式，便于直接查看和编辑）
            2. .json文件：存储元数据（配置参数、变电站位置等）
        
        参数：
            filepath: 保存路径（不需要包含扩展名）
            
        示例：
            >>> grid_map.save('data/map_seed42')
            # 会生成 data/map_seed42.csv 和 data/map_seed42.json
        
        CSV文件格式说明：
            - 每行代表地图的一行（从y=0开始）
            - 每列代表地图的一列（从x=0开始）
            - 值的含义：-1=硬约束，0=正常区域，0<value<=1=软约束代价
        """
        filepath = Path(filepath)
        
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存网格矩阵数据（CSV格式，便于直接查看）
        # 使用4位小数精度，足够表示软约束的渐变值
        np.savetxt(filepath.with_suffix('.csv'), self.grid, 
                   delimiter=',', fmt='%.4f')
        
        # 准备元数据
        metadata = {
            'config': asdict(self.config),  # 配置参数
            'grid_width': self.grid_width,   # 网格宽度
            'grid_height': self.grid_height, # 网格高度
            'substation_a': self.substation_a,  # 变电站A位置
            'substation_b': self.substation_b   # 变电站B位置
        }
        
        # 保存元数据（JSON格式，便于查看和编辑）
        with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Map saved: {filepath.with_suffix('.csv')}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GridMap':
        """
        从文件加载地图（类方法）
        
        参数：
            filepath: 地图文件路径（不需要包含扩展名）
            
        返回：
            加载的GridMap对象
            
        示例：
            >>> grid_map = GridMap.load('data/map_seed42')
        
        注意：
            支持加载.csv格式（新格式）和.npy格式（旧格式）
        """
        filepath = Path(filepath)
        
        # 加载元数据
        with open(filepath.with_suffix('.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 从元数据重建配置对象
        config_dict = metadata['config']
        # 将列表转换回元组（JSON不支持元组）
        config_dict['substation_a'] = tuple(config_dict['substation_a'])
        config_dict['substation_b'] = tuple(config_dict['substation_b'])
        config = MapConfig(**config_dict)
        
        # 创建地图对象
        grid_map = cls(config)
        
        # 加载网格矩阵数据（优先尝试CSV格式，兼容旧的NPY格式）
        csv_path = filepath.with_suffix('.csv')
        npy_path = filepath.with_suffix('.npy')
        
        if csv_path.exists():
            # 加载CSV格式（新格式）
            grid_map.grid = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
            print(f"Map loaded: {csv_path}")
        elif npy_path.exists():
            # 加载NPY格式（旧格式，向后兼容）
            grid_map.grid = np.load(npy_path)
            print(f"Map loaded: {npy_path}")
        else:
            raise FileNotFoundError(f"Map data file not found: {csv_path} or {npy_path}")
        
        return grid_map
    
    def visualize(self, save_path: str = None, show: bool = True, 
                  grid_interval: int = 10) -> None:
        """
        可视化地图
        
        生成地图的彩色可视化图像，包括：
            - 不同颜色表示不同类型的区域
            - 网格线
            - 变电站位置标记
            - 图例说明
        
        参数：
            save_path: 图片保存路径（如果为None则不保存）
            show: 是否显示图片（在交互环境中使用）
            grid_interval: 网格线间隔（每隔多少个网格画一条线）
        """
        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 创建RGB显示矩阵（用于存储每个像素的颜色）
        display = np.zeros((*self.grid.shape, 3))
        
        # 设置不同区域的颜色
        # 正常区域 - 白色
        normal_mask = self.grid == 0
        display[normal_mask] = [1, 1, 1]  # RGB: 白色
        
        # 硬约束区域 - 红色
        hard_mask = self.grid == -1
        display[hard_mask] = [0.8, 0.2, 0.2]  # RGB: 深红色
        
        # 软约束区域 - 浅黄到深橙渐变（代价越高颜色越深，更明显的渐变）
        soft_mask = (self.grid > 0) & (self.grid <= 1)
        soft_values = self.grid[soft_mask]
        # 归一化到0-1范围，增强对比度
        if len(soft_values) > 0:
            min_val = soft_values.min()
            max_val = soft_values.max()
            if max_val > min_val:
                normalized = (soft_values - min_val) / (max_val - min_val)
            else:
                normalized = np.ones_like(soft_values) * 0.5
            # 浅黄(1,1,0.6) -> 深橙(1,0.3,0)
            display[soft_mask, 0] = 1.0                           # R: 保持最大
            display[soft_mask, 1] = 1.0 - normalized * 0.7        # G: 从1.0到0.3
            display[soft_mask, 2] = 0.6 - normalized * 0.6        # B: 从0.6到0
        
        # 显示地图图像
        # origin='lower' 使得y轴从下往上增加（符合数学坐标系习惯）
        ax.imshow(display, origin='lower', extent=[0, self.grid_width, 0, self.grid_height])
        
        # 绘制网格线
        # 设置网格线的样式：灰色、细线、半透明
        for x in range(0, self.grid_width + 1, grid_interval):
            ax.axvline(x=x, color='gray', linewidth=0.3, alpha=0.5)
        for y in range(0, self.grid_height + 1, grid_interval):
            ax.axhline(y=y, color='gray', linewidth=0.3, alpha=0.5)
        
        # 标记变电站位置
        # 变电站A - 蓝色三角形
        ax.plot(self.substation_a[0], self.substation_a[1], 
                'b^', markersize=15, markeredgecolor='black', 
                markeredgewidth=2, label='变电站A (起点)')
        
        # 变电站B - 绿色方形
        ax.plot(self.substation_b[0], self.substation_b[1], 
                'gs', markersize=15, markeredgecolor='black',
                markeredgewidth=2, label='变电站B (终点)')
        
        # 创建图例元素
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='black', label='正常区域 (可通行)'),
            mpatches.Patch(facecolor=(0.8, 0.2, 0.2), label='硬约束 (禁止区域)'),
            mpatches.Patch(facecolor=(1.0, 1.0, 0.6), label='软约束-边缘 (低代价)'),
            mpatches.Patch(facecolor=(1.0, 0.3, 0.0), label='软约束-中心 (高代价)'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                      markersize=12, markeredgecolor='black', label='变电站A (起点)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                      markersize=12, markeredgecolor='black', label='变电站B (终点)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 设置坐标轴标签和标题
        ax.set_xlabel('X坐标 (网格)', fontsize=12)
        ax.set_ylabel('Y坐标 (网格)', fontsize=12)
        ax.set_title(f'输电线路规划地图 ({self.grid_width}×{self.grid_height}网格，'
                    f'每格{int(self.config.resolution_m)}m)', fontsize=14)
        
        # 设置坐标轴范围
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        
        # 添加刻度（每50个网格一个主刻度）
        ax.set_xticks(range(0, self.grid_width + 1, 50))
        ax.set_yticks(range(0, self.grid_height + 1, 50))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Image saved: {save_path}")
        
        # 显示图片
        if show:
            plt.show()
        
        # 关闭图形，释放内存
        plt.close()


def generate_and_save_map(output_dir: str = 'data', seed: int = 42) -> GridMap:
    """
    生成并保存地图的便捷函数
    
    这是一个一站式函数，完成以下操作：
        1. 创建地图配置
        2. 初始化地图
        3. 生成硬约束和软约束区域
        4. 保存地图数据和可视化图片
    
    参数：
        output_dir: 输出目录路径
        seed: 随机数种子（相同的种子会生成相同的地图）
        
    返回：
        生成的GridMap对象
        
    示例：
        >>> grid_map = generate_and_save_map(seed=42)
        >>> # 会在data目录下生成map_seed42_hard2.csv, map_seed42_hard2.json, map_seed42_hard2.png
    """
    # 创建配置（使用指定的随机种子）
    config = MapConfig(seed=seed)
    
    # 初始化地图
    grid_map = GridMap(config)
    
    # 生成约束区域
    grid_map.generate_hard_constraints()  # 先生成硬约束
    grid_map.generate_soft_constraints()  # 再生成软约束（不会覆盖硬约束）
    
    # 构建输出文件路径（包含硬约束数量）
    num_hard_zones = config.num_hard_zones
    output_path = Path(output_dir) / f'map_seed{seed}_hard{num_hard_zones}'
    
    # 保存地图数据
    grid_map.save(str(output_path))
    
    # 保存可视化图片（不显示，只保存）
    grid_map.visualize(save_path=str(output_path) + '.png', show=False)
    
    return grid_map


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == '__main__':
    """
    当直接运行此脚本时执行的代码
    
    功能：
        1. 生成一个示例地图
        2. 打印地图统计信息
    """
    # 生成示例地图（使用种子42保证可复现）
    grid_map = generate_and_save_map(seed=42)
    
    # 计算并打印统计信息
    # 统计各类型网格的数量
    hard_count = np.sum(grid_map.grid == -1)  # 硬约束网格数
    soft_count = np.sum((grid_map.grid > 0) & (grid_map.grid <= 1))  # 软约束网格数
    normal_count = np.sum(grid_map.grid == 0)  # 正常网格数
    total_count = grid_map.grid.size  # 总网格数
    
    # 打印统计结果 (使用英文避免终端编码问题，图片中仍显示中文)
    print(f"\n{'='*50}")
    print(f"Map Statistics")
    print(f"{'='*50}")
    print(f"  Grid size: {grid_map.grid_width} x {grid_map.grid_height}")
    print(f"  Cell size: {int(grid_map.config.resolution_m)}m x {int(grid_map.config.resolution_m)}m")
    print(f"  Real area: {grid_map.config.width_km}km x {grid_map.config.height_km}km")
    print(f"{'='*50}")
    print(f"  Hard constraint cells: {hard_count:,} ({hard_count/total_count*100:.1f}%)")
    print(f"  Soft constraint cells: {soft_count:,} ({soft_count/total_count*100:.1f}%)")
    print(f"  Normal cells:          {normal_count:,} ({normal_count/total_count*100:.1f}%)")
    print(f"{'='*50}")
    print(f"  Substation A: {grid_map.substation_a}")
    print(f"  Substation B: {grid_map.substation_b}")
    print(f"{'='*50}")
    
    # 执行重叠检测验证
    grid_map.print_validation_report()
