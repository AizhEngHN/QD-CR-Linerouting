"""
路径评估模块
============

功能说明：
    评估输电线路路径的总代价，采用统一的"等效杆塔数量"单位。
    
    代价函数：F = sum(a_i) + lambda * L
    其中：
        - a_i = 1 + f(theta_i) + c_i （第i个杆塔的权重）
        - f(theta_i) = (1 - cos(theta_i)) / 2 （转角惩罚）
        - c_i = g(p_i) （地形惩罚）
        - lambda = 1 / L_equiv （线路权重系数）
        - L 为路径总长度（km）

作者：Power Line Routing Project Team
日期：2026-01
版本：2.0 (基于工程数据的改进版本)
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import math
import time
from routing_geometry import segment_is_clear


class EvaluationBudgetExceeded(RuntimeError):
    """No objective call was made because its count or time budget expired."""


@dataclass
class CostWeights:
    """
    代价权重配置类
    
    基于工程实际的统一单位设计：所有代价以"等效杆塔数量"为单位
    
    属性说明：
        tower_to_line_km: 等效比例，1个杆塔相当于多少公里导线
                         基于工程数据：220kV杆塔约20-30万元，导线约8-12万元/km
                         因此 1个杆塔 ≈ 2.5km 导线
        angle_weight: 转角权重系数（默认1.0，表示90度转角增加0.5个杆塔代价）
        terrain_weight: 地形权重系数（默认1.0，表示最困难地形增加1个杆塔代价）
        max_span_km: 最大跨距约束（单位：km），相邻杆塔之间的最大距离
                     仿真参数，需要针对实际电压等级和工程条件校准。
        hard_constraint_penalty: 硬约束惩罚（极大值，使路径不可行）
    """
    tower_to_line_km: float = 2.5             # 1个杆塔 = 2.5km导线（基于工程数据）
    angle_weight: float = 1.0                 # 转角权重系数
    terrain_weight: float = 1.0               # 地形权重系数
    max_span_km: float = 2.0                  # 仿真最大档距假设（km），非标准声明
    hard_constraint_penalty: float = 1e6      # 硬约束惩罚


class PathEvaluator:
    """
    路径评估器类
    
    实现基于工程数据的代价函数：F = sum(a_i) + lambda * L
    
    核心思想：
        - 所有代价统一到"等效杆塔数量"单位
        - 每个杆塔的权重根据转角和地形动态调整
        - 线路长度转换为等效杆塔数量
    
    主要功能：
        1. 验证路径的可行性（是否经过硬约束区域）
        2. 计算路径的总代价（等效杆塔数量）
        3. 提供详细的代价分解信息
    
    使用示例：
        >>> from map_generator import GridMap
        >>> grid_map = GridMap.load('data/map_seed42_hard2')
        >>> evaluator = PathEvaluator(grid_map)
        >>> path = [(10, 10), (75, 75), (140, 140)]
        >>> cost, details = evaluator.evaluate(path)
        >>> print(f"Total cost: {cost:.2f} equivalent towers")
    """
    
    def __init__(self, grid_map, weights: CostWeights = None,
                 max_evaluations=None, deadline=None):
        """
        初始化路径评估器
        
        参数：
            grid_map: GridMap对象，包含地图数据
            weights: CostWeights对象，代价权重配置（如果为None则使用默认配置）
        """
        self.grid_map = grid_map
        self.weights = weights or CostWeights()
        
        # 缓存地图数据以加速访问
        self.grid = grid_map.grid
        self.grid_width = grid_map.grid_width
        self.grid_height = grid_map.grid_height
        
        # 计算网格分辨率（km）
        self.resolution_km = grid_map.config.resolution_m / 1000.0
        if self.resolution_km <= 0 or self.weights.max_span_km <= 0:
            raise ValueError("Resolution and maximum span must be positive")
        if self.weights.tower_to_line_km <= 0:
            raise ValueError("Tower-to-line conversion must be positive")
        self.evaluation_count = 0
        self.max_evaluations = max_evaluations
        self.deadline = deadline
        self._clearance_cache = {}

    def segment_is_clear(self, start, end):
        """Cached geometry for a fixed map. Use a new evaluator after map edits."""
        a, b = tuple(start), tuple(end)
        key = (a, b) if a <= b else (b, a)
        if key not in self._clearance_cache:
            self._clearance_cache[key] = segment_is_clear(self.grid, a, b)
        return self._clearance_cache[key]
    
    def calculate_angle(self, p_prev: Tuple[int, int], 
                       p_curr: Tuple[int, int], 
                       p_next: Tuple[int, int]) -> float:
        """
        计算转角角度（弧度）
        
        计算前后两段线路之间的夹角，用于转角惩罚。
        
        参数：
            p_prev: 前一个点坐标 (x, y)
            p_curr: 当前点坐标 (x, y)
            p_next: 下一个点坐标 (x, y)
            
        返回：
            角度（弧度），范围 [0, π]
            
        注意：
            - 0 表示直线（无转角）
            - π/2 表示90度转角
            - π 表示180度转角（掉头）
        """
        # 向量1：从前一个点到当前点
        vec1 = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]], dtype=float)
        # 向量2：从当前点到下一个点
        vec2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]], dtype=float)
        
        # 计算向量长度
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)
        
        # 处理退化情况（重合点）
        if len1 < 1e-6 or len2 < 1e-6:
            return 0.0
        
        # 计算夹角（使用点积）
        cos_angle = np.dot(vec1, vec2) / (len1 * len2)
        # 限制在[-1, 1]范围内（避免数值误差）
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # 返回角度（弧度）
        angle = np.arccos(cos_angle)
        return angle
    
    def angle_penalty(self, theta: float) -> float:
        """
        转角惩罚函数
        
        使用公式：f(theta) = (1 - cos(theta)) / 2
        
        特点：
            - theta = 0°   -> f = 0.0 (无惩罚，直线)
            - theta = 90°  -> f = 0.5 (中等惩罚)
            - theta = 180° -> f = 1.0 (最大惩罚，掉头)
        
        参数：
            theta: 转角角度（弧度）
            
        返回：
            惩罚值，范围 [0, 1]，乘以 angle_weight 后加到杆塔权重上
        """
        return (1.0 - np.cos(theta)) / 2.0 * self.weights.angle_weight
    
    def is_valid_path(self, path: List[Tuple[int, int]]) -> Tuple[bool, str]:
        """
        检查路径是否可行
        
        检查项：
            1. 路径至少包含2个点（起点和终点）
            2. 起点必须是变电站A
            3. 终点必须是变电站B
            4. 所有点必须在地图范围内
            5. 不能经过硬约束区域（grid值=-1）
            6. 相邻点之间的距离不能超过最大跨距（span约束）
        
        参数：
            path: 路径点列表，每个点为(x, y)坐标
            
        返回：
            (is_valid, message): 
                - is_valid: 布尔值，True表示路径可行
                - message: 字符串，描述不可行的原因（如果可行则为空字符串）
        """
        if len(path) < 2:
            return False, "Path must contain at least 2 points"
        if any(len(point) != 2 or any(not math.isfinite(v) or int(v) != v for v in point)
               for point in path):
            return False, "Path points must contain finite integer grid coordinates"
        path = [(int(x), int(y)) for x, y in path]
        
        # 检查起点和终点
        start = tuple(path[0])
        end = tuple(path[-1])
        expected_start = self.grid_map.substation_a
        expected_end = self.grid_map.substation_b
        
        if start != expected_start:
            return False, f"Path must start at substation A {expected_start}, but starts at {start}"
        
        if end != expected_end:
            return False, f"Path must end at substation B {expected_end}, but ends at {end}"
        
        # 检查所有点是否在地图范围内
        for i, (x, y) in enumerate(path):
            if not all(math.isfinite(v) and int(v) == v for v in (x, y)):
                return False, f"Point {i} must contain finite integer grid coordinates"
            if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                return False, f"Point {i} ({x}, {y}) is out of map bounds"
        
        # 检查是否经过硬约束区域
        for i, (x, y) in enumerate(path):
            if self.grid[y, x] == -1:
                return False, f"Point {i} ({x}, {y}) is in hard constraint area (forbidden)"
        
        # 检查span约束（相邻点之间的距离）
        max_span_grid = self.weights.max_span_km / self.resolution_km  # 转换为网格单位
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # 计算欧氏距离（网格单位）
            distance_grid = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance_grid == 0:
                return False, f"Zero-length segment at point {i}"
            
            if distance_grid > max_span_grid:
                distance_km = distance_grid * self.resolution_km
                return False, (f"Span constraint violated between point {i} ({x1}, {y1}) "
                             f"and point {i+1} ({x2}, {y2}): "
                             f"distance = {distance_km:.2f} km > max_span = {self.weights.max_span_km:.2f} km")
            if not self.segment_is_clear((x1, y1), (x2, y2)):
                return False, f"Segment {i} intersects a forbidden cell"
        
        return True, ""
    
    def evaluate(self, path: List[Tuple[int, int]]) -> Tuple[float, dict]:
        """
        评估路径的总代价（等效杆塔数量）
        
        公式：F = sum(a_i) + lambda * L
        其中：
            - a_i = 1 + f(theta_i) + c_i （第i个杆塔的权重）
            - f(theta_i) = (1 - cos(theta_i)) / 2 （转角惩罚）
            - c_i = g(p_i) （地形惩罚）
            - lambda = 1 / L_equiv （线路权重系数）
            - L 为路径总长度（km）
        
        参数：
            path: 路径点列表，每个点为(x, y)坐标
            
        返回：
            (total_cost, details):
                - total_cost: 总代价（等效杆塔数量）
                - details: 代价详细信息字典
        """
        if self.max_evaluations is not None and self.evaluation_count >= self.max_evaluations:
            raise EvaluationBudgetExceeded("Objective evaluation budget exhausted")
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise EvaluationBudgetExceeded("Wall-clock budget exhausted")
        self.evaluation_count += 1
        # 首先检查路径可行性
        is_valid, message = self.is_valid_path(path)
        if not is_valid:
            return float('inf'), {
                'valid': False,
                'error': message,
                'hard_constraint_penalty': float('inf')
            }
        path = [(int(x), int(y)) for x, y in path]
        
        # 1. 计算路径总长度（km）
        total_distance_km = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            grid_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance_km += grid_distance * self.resolution_km
        
        # 2. 计算每个杆塔的权重 a_i
        num_towers = len(path) - 2  # 不包括起点和终点变电站
        tower_weights = []
        angle_penalties = []
        terrain_penalties = []
        angles_deg = []  # 记录转角（度数，用于输出）
        
        for i in range(1, len(path) - 1):  # 遍历所有杆塔位置（不包括起点和终点）
            x, y = path[i]
            
            # 2.1 基础权重
            a_i = 1.0
            
            # 2.2 转角惩罚
            theta = self.calculate_angle(path[i-1], path[i], path[i+1])
            angle_penalty_val = self.angle_penalty(theta)
            a_i += angle_penalty_val
            angle_penalties.append(angle_penalty_val)
            angles_deg.append(np.degrees(theta))  # 转换为度数便于理解
            
            # 2.3 地形惩罚
            cell_value = self.grid[y, x]
            if 0 < cell_value <= 1:
                terrain_penalty_val = cell_value * self.weights.terrain_weight
                a_i += terrain_penalty_val
                terrain_penalties.append(terrain_penalty_val)
            else:
                terrain_penalties.append(0.0)
            
            tower_weights.append(a_i)
        
        # 3. 计算总杆塔代价
        total_tower_cost = sum(tower_weights) if tower_weights else 0.0
        
        # 4. 计算线路代价（等效杆塔数量）
        lambda_param = 1.0 / self.weights.tower_to_line_km
        line_cost = lambda_param * total_distance_km
        
        # 5. 总代价（等效杆塔数量）
        total_cost = total_tower_cost + line_cost
        
        # 6. 硬约束检查（理论上不应该出现，因为已经验证过）
        hard_penalty = 0.0
        for x, y in path:
            if self.grid[y, x] == -1:
                hard_penalty = self.weights.hard_constraint_penalty
                total_cost = float('inf')
                break
        
        # 7. 返回详细信息
        details = {
            'valid': True,
            'total_cost': total_cost,
            'total_tower_cost': total_tower_cost,
            'line_cost': line_cost,
            'num_towers': num_towers,
            'avg_tower_weight': np.mean(tower_weights) if tower_weights else 1.0,
            'max_tower_weight': max(tower_weights) if tower_weights else 1.0,
            'min_tower_weight': min(tower_weights) if tower_weights else 1.0,
            'total_angle_penalty': sum(angle_penalties),
            'total_terrain_penalty': sum(terrain_penalties),
            'total_distance_km': total_distance_km,
            'path_length': len(path),
            'tower_weights': tower_weights,
            'angle_penalties': angle_penalties,
            'terrain_penalties': terrain_penalties,
            'angles_deg': angles_deg,
            'hard_constraint_penalty': hard_penalty
        }
        
        return total_cost, details
    
    def print_evaluation(self, path: List[Tuple[int, int]]) -> None:
        """
        打印路径评估结果（格式化输出）
        
        参数：
            path: 路径点列表
        """
        cost, details = self.evaluate(path)
        
        print(f"\n{'='*70}")
        print(f"Path Evaluation Report (Equivalent Tower Units)")
        print(f"{'='*70}")
        
        if not details['valid']:
            print(f"  Status: INVALID")
            print(f"  Error: {details['error']}")
            print(f"{'='*70}")
            return
        
        print(f"  Status: VALID")
        print(f"  Total Cost: {cost:.2f} equivalent towers")
        print(f"{'='*70}")
        print(f"  Cost Breakdown:")
        print(f"    - Tower cost:        {details['total_tower_cost']:.2f} towers")
        print(f"    - Line cost:         {details['line_cost']:.2f} equiv towers")
        print(f"      (Total distance:   {details['total_distance_km']:.2f} km)")
        print(f"{'='*70}")
        print(f"  Tower Analysis:")
        print(f"    - Number of towers:  {details['num_towers']}")
        print(f"    - Avg tower weight:  {details['avg_tower_weight']:.3f}")
        print(f"    - Max tower weight:  {details['max_tower_weight']:.3f}")
        print(f"    - Min tower weight:  {details['min_tower_weight']:.3f}")
        print(f"    - Angle penalty:     {details['total_angle_penalty']:.3f}")
        print(f"    - Terrain penalty:   {details['total_terrain_penalty']:.3f}")
        print(f"{'='*70}")
        print(f"  Configuration:")
        print(f"    - Tower/Line ratio:  1 tower = {self.weights.tower_to_line_km:.1f} km")
        print(f"    - Angle weight:      {self.weights.angle_weight:.1f}")
        print(f"    - Terrain weight:    {self.weights.terrain_weight:.1f}")
        print(f"{'='*70}")


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    """
    测试路径评估器（新版本：基于等效杆塔数量）
    """
    from map_generator import GridMap
    
    # 加载地图
    print("Loading map...")
    grid_map = GridMap.load('data/map_seed42_hard2')
    
    # 创建评估器（使用默认权重）
    print("\nUsing cost configuration:")
    print(f"  - Tower to line ratio: 1 tower = 2.5 km")
    print(f"  - Angle weight: 1.0")
    print(f"  - Terrain weight: 1.0")
    print(f"\nCost function: F = sum(a_i) + L/2.5")
    print(f"  where a_i = 1 + (1-cos(theta_i))/2 + g(p_i)")
    
    evaluator = PathEvaluator(grid_map)
    
    # 测试路径1：简单直线路径（3个点，1个杆塔，无转角）
    print("\n" + "="*70)
    print("Test 1: Simple straight path")
    print("Path: (10,10) -> (75,75) -> (140,140)")
    print("="*70)
    path1 = [(10, 10), (75, 75), (140, 140)]
    evaluator.print_evaluation(path1)
    
    # 测试路径2：带更多杆塔的直线路径（5个点，3个杆塔，无转角）
    print("\n" + "="*70)
    print("Test 2: Straight path with more towers")
    print("Path: (10,10) -> (45,45) -> (80,80) -> (115,115) -> (140,140)")
    print("="*70)
    path2 = [(10, 10), (45, 45), (80, 80), (115, 115), (140, 140)]
    evaluator.print_evaluation(path2)
    
    # 测试路径3：带转角的路径（测试转角代价）
    print("\n" + "="*70)
    print("Test 3: Path with turns")
    print("Path: (10,10) -> (10,75) -> (75,75) -> (140,75) -> (140,140)")
    print("="*70)
    path3 = [(10, 10), (10, 75), (75, 75), (140, 75), (140, 140)]
    evaluator.print_evaluation(path3)
    
    # 测试路径4：对比不同权重设置
    print("\n" + "="*70)
    print("Test 4: Different weight configurations")
    print("="*70)
    
    # 配置1：标准配置
    weights1 = CostWeights(tower_to_line_km=2.5)
    eval1 = PathEvaluator(grid_map, weights1)
    cost1, _ = eval1.evaluate(path3)
    print(f"  Config 1 (1 tower = 2.5 km): {cost1:.2f} equiv towers")
    
    # 配置2：杆塔成本相对更高
    weights2 = CostWeights(tower_to_line_km=2.0)
    eval2 = PathEvaluator(grid_map, weights2)
    cost2, _ = eval2.evaluate(path3)
    print(f"  Config 2 (1 tower = 2.0 km): {cost2:.2f} equiv towers")
    
    # 配置3：导线成本相对更高
    weights3 = CostWeights(tower_to_line_km=3.0)
    eval3 = PathEvaluator(grid_map, weights3)
    cost3, _ = eval3.evaluate(path3)
    print(f"  Config 3 (1 tower = 3.0 km): {cost3:.2f} equiv towers")
    
    print(f"\n  Cost difference: {abs(cost1-cost2):.2f} (Config 1 vs 2)")
    print(f"  Cost difference: {abs(cost1-cost3):.2f} (Config 1 vs 3)")
    print("="*70)
