"""
变异算子模块
============

功能说明：
    实现MAP-Elites的变异算子，包括：
    1. add_point: 添加新的路径点
    2. remove_point: 删除路径点
    3. move_point: 移动路径点
    
作者：Power Line Routing Project Team
日期：2026-01-22
"""

import random
import numpy as np
from typing import List, Tuple
from qd_individual import Individual
from path_evaluator import PathEvaluator, CostWeights


class MutationOperators:
    """
    变异算子类
    
    包含三种基本变异操作，现在会尝试维持span约束。
    如果变异后违反span约束，会尝试修复。
    
    属性说明：
        grid_map: GridMap对象，包含地形数据
        max_span_km: 最大跨距约束（km）
        max_span_grid: 最大跨距约束（网格单位）
    """
    
    def __init__(self, grid_map, max_span_km: float = 2.0):
        """
        初始化变异算子
        
        参数：
            grid_map: GridMap对象
            max_span_km: 最大跨距约束（km），默认2.0km
        """
        self.grid_map = grid_map
        self.max_span_km = max_span_km
        
        # 计算网格单位的最大跨距
        resolution_km = grid_map.config.resolution_m / 1000.0
        self.max_span_grid = max_span_km / resolution_km
        self.geometry = PathEvaluator(grid_map, CostWeights(max_span_km=max_span_km))

    def _edge_valid(self, a, b):
        return (tuple(a) != tuple(b)
                and np.hypot(b[0] - a[0], b[1] - a[1]) <= self.max_span_grid + 1e-12
                and self.geometry.segment_is_clear(a, b))
    
    # ========================================================================
    # 变异算子
    # ========================================================================
    
    def add_point(self, individual: Individual) -> Individual:
        """
        添加点变异算子
        
        策略：
            1. 随机选择一条边（两个相邻点之间）
            2. 在边上插值生成新点（t ∈ [0.3, 0.7]）
            3. 添加随机扰动（±3格）
            4. 确保新点与前后点都满足span约束
        
        参数：
            individual: 父代个体
            
        返回：
            变异后的个体
        """
        new_ind = individual.copy()
        
        # 至少需要2个点（起点和终点）
        if len(new_ind.path) < 2:
            return new_ind
        
        # 随机选择一条边
        edge_idx = random.randint(0, len(new_ind.path) - 2)
        p1 = new_ind.path[edge_idx]
        p2 = new_ind.path[edge_idx + 1]
        
        # 在边上插值
        t = random.uniform(0.3, 0.7)
        x_base = int(p1[0] + t * (p2[0] - p1[0]))
        y_base = int(p1[1] + t * (p2[1] - p1[1]))
        
        # 尝试添加随机扰动
        max_attempts = 20
        for attempt in range(max_attempts):
            # 随机扰动（尝试次数越多，扰动越小）
            max_offset = max(1, 3 - attempt // 7)
            dx = random.randint(-max_offset, max_offset)
            dy = random.randint(-max_offset, max_offset)
            x_new = np.clip(x_base + dx, 0, self.grid_map.grid_width - 1)
            y_new = np.clip(y_base + dy, 0, self.grid_map.grid_height - 1)
            
            # 检查与前后点的距离
            dist_to_p1 = np.sqrt((x_new - p1[0])**2 + (y_new - p1[1])**2)
            dist_to_p2 = np.sqrt((x_new - p2[0])**2 + (y_new - p2[1])**2)
            
            # 如果满足span约束，接受这个点
            if self._edge_valid(p1, (x_new, y_new)) and self._edge_valid((x_new, y_new), p2):
                new_ind.path.insert(edge_idx + 1, (x_new, y_new))
                break
        else:
            # 如果所有尝试都失败，使用基准点（不加扰动）
            x_new = np.clip(x_base, 0, self.grid_map.grid_width - 1)
            y_new = np.clip(y_base, 0, self.grid_map.grid_height - 1)
            if self._edge_valid(p1, (x_new, y_new)) and self._edge_valid((x_new, y_new), p2):
                new_ind.path.insert(edge_idx + 1, (x_new, y_new))
        
        # 清除适应度和行为
        new_ind.fitness = None
        new_ind.behavior = None
        
        return new_ind
    
    def remove_point(self, individual: Individual) -> Individual:
        """
        删除点变异算子
        
        策略：
            1. 随机选择一个中间点
            2. 检查删除后是否违反span约束
            3. 如果违反，不删除该点
        
        参数：
            individual: 父代个体
            
        返回：
            变异后的个体
        """
        new_ind = individual.copy()
        
        # 至少需要3个点（起点、中间点、终点）
        if len(new_ind.path) <= 2:
            return new_ind
        
        # 随机选择一个中间点
        point_idx = random.randint(1, len(new_ind.path) - 2)
        
        # 检查删除后的span约束
        # 删除点i后，点i-1和点i+1会直接相连
        p_prev = new_ind.path[point_idx - 1]
        p_next = new_ind.path[point_idx + 1]
        
        dist = np.sqrt((p_next[0] - p_prev[0])**2 + (p_next[1] - p_prev[1])**2)
        
        # 只有在不违反span约束时才删除
        if self._edge_valid(p_prev, p_next):
            new_ind.path.pop(point_idx)
        # 否则保持原样（不删除）
        
        # 清除适应度和行为
        new_ind.fitness = None
        new_ind.behavior = None
        
        return new_ind
    
    def move_point(self, individual: Individual) -> Individual:
        """
        移动点变异算子
        
        策略：
            1. 随机选择一个中间点
            2. 随机移动（±5格）
            3. 检查移动后是否违反span约束
            4. 如果违反，尝试更小的移动距离
        
        参数：
            individual: 父代个体
            
        返回：
            变异后的个体
        """
        new_ind = individual.copy()
        
        # 至少需要3个点
        if len(new_ind.path) <= 2:
            return new_ind
        
        # 随机选择一个中间点
        point_idx = random.randint(1, len(new_ind.path) - 2)
        x, y = new_ind.path[point_idx]
        
        # 获取前后点
        p_prev = new_ind.path[point_idx - 1]
        p_next = new_ind.path[point_idx + 1]
        
        # 尝试多次移动，找到满足span约束的位置
        max_attempts = 20
        for attempt in range(max_attempts):
            # 随机偏移（尝试次数越多，偏移越小）
            max_offset = max(1, 5 - attempt // 4)
            dx = random.randint(-max_offset, max_offset)
            dy = random.randint(-max_offset, max_offset)
            x_new = np.clip(x + dx, 0, self.grid_map.grid_width - 1)
            y_new = np.clip(y + dy, 0, self.grid_map.grid_height - 1)
            
            # 检查与前后点的距离
            dist_to_prev = np.sqrt((x_new - p_prev[0])**2 + (y_new - p_prev[1])**2)
            dist_to_next = np.sqrt((x_new - p_next[0])**2 + (y_new - p_next[1])**2)
            
            # 如果满足span约束，接受这个移动
            if (self._edge_valid(p_prev, (x_new, y_new))
                    and self._edge_valid((x_new, y_new), p_next)):
                new_ind.path[point_idx] = (x_new, y_new)
                break
        # 如果所有尝试都失败，保持原位置不变
        
        # 清除适应度和行为
        new_ind.fitness = None
        new_ind.behavior = None
        
        return new_ind
    
    def mutate(self, individual: Individual, 
               mutation_probs: dict = None) -> Individual:
        """
        应用随机变异算子
        
        根据概率分布随机选择一个变异算子。
        
        参数：
            individual: 父代个体
            mutation_probs: 变异算子概率分布
                默认：{'add_point': 0.30, 'remove_point': 0.20, 'move_point': 0.50}
        
        返回：
            变异后的个体
        """
        if mutation_probs is None:
            mutation_probs = {
                'add_point': 0.30,
                'remove_point': 0.20,
                'move_point': 0.50
            }
        
        # 随机选择变异算子
        operators = list(mutation_probs.keys())
        probs = list(mutation_probs.values())
        if (any(op not in {'add_point', 'remove_point', 'move_point'} for op in operators)
                or not probs or any(not np.isfinite(p) or p < 0 for p in probs)
                or sum(probs) <= 0):
            raise ValueError("Mutation probabilities must name implemented operators and have positive mass")
        selected_op = random.choices(operators, weights=probs, k=1)[0]
        
        # 应用变异算子
        if selected_op == 'add_point':
            return self.add_point(individual)
        elif selected_op == 'remove_point':
            return self.remove_point(individual)
        elif selected_op == 'move_point':
            return self.move_point(individual)
        else:
            raise ValueError(f"Unknown mutation operator: {selected_op}")


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    """
    测试变异算子
    """
    from map_generator import GridMap
    from qd_individual import Individual
    
    # 加载地图
    print("Loading map...")
    grid_map = GridMap.load('data/map_seed42_hard2')
    
    # 创建变异算子
    mutator = MutationOperators(grid_map)
    
    # 创建测试个体
    test_path = [(10, 10), (50, 50), (100, 100), (140, 140)]
    test_ind = Individual(path=test_path)
    
    print(f"\nOriginal individual: {test_ind}")
    print(f"Path: {test_ind.path}")
    
    # 测试add_point
    print("\n" + "="*60)
    print("Testing add_point mutation...")
    mutated = mutator.add_point(test_ind)
    print(f"Original: {len(test_ind.path)} points")
    print(f"After add_point: {len(mutated.path)} points")
    print(f"New path: {mutated.path}")
    
    # 测试remove_point
    print("\n" + "="*60)
    print("Testing remove_point mutation...")
    mutated = mutator.remove_point(test_ind)
    print(f"Original: {len(test_ind.path)} points")
    print(f"After remove_point: {len(mutated.path)} points")
    print(f"New path: {mutated.path}")
    
    # 测试move_point
    print("\n" + "="*60)
    print("Testing move_point mutation...")
    mutated = mutator.move_point(test_ind)
    print(f"Original: {len(test_ind.path)} points")
    print(f"After move_point: {len(mutated.path)} points")
    print(f"Original path: {test_ind.path}")
    print(f"New path: {mutated.path}")
    
    # 测试多次变异
    print("\n" + "="*60)
    print("Testing multiple mutations...")
    current = test_ind
    for i in range(10):
        mutated = mutator.mutate(current)
        print(f"Mutation {i+1}: {len(mutated.path)} points")
        current = mutated
    
    print(f"\nFinal path after 10 mutations: {len(current.path)} points")
