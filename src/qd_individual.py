"""
QD个体表示模块
==============

功能说明：
    定义Individual类，表示一条输电线路路径。
    使用点序列表示，支持可变长度路径。
    
作者：Power Line Routing Project Team
日期：2026-01-22
"""

import copy
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Individual:
    """
    QD个体：表示一条输电线路路径
    
    属性说明：
        path: 路径点列表 [(x0,y0), (x1,y1), ..., (xn,yn)]
              端点由具体地图和 PathEvaluator 检查，不硬编码坐标
        fitness: 适应度（总代价，越小越好），单位：等效杆塔数
        behavior: 行为描述符 (b1, b2)
                  b1 = 路径长度 (km)
                  b2 = 平均地形代价 (0-1)
        metadata: 其他信息（可选）
    
    使用示例：
        >>> path = [(10, 10), (50, 50), (100, 100), (140, 140)]
        >>> individual = Individual(path=path)
        >>> print(individual)
        Individual(points=4, towers=2, fitness=None, behavior=None)
    """
    path: List[Tuple[int, int]]
    fitness: Optional[float] = None
    behavior: Optional[Tuple[float, float]] = None
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """
        初始化后的验证
        
        检查：
            1. 路径至少有2个点
            地图相关的端点和工程约束由 PathEvaluator 检查。
        """
        if len(self.path) < 2:
            raise ValueError("Path must have at least 2 points")
    
    def __len__(self):
        """返回路径点数"""
        return len(self.path)
    
    def num_towers(self):
        """
        返回杆塔数量（不包括起点和终点变电站）
        
        返回：
            杆塔数量
        """
        return len(self.path) - 2
    
    def copy(self):
        """
        创建深拷贝
        
        返回：
            新的Individual对象，与原对象完全独立
        """
        return Individual(
            path=copy.deepcopy(self.path),
            fitness=self.fitness,
            behavior=self.behavior,
            metadata=copy.deepcopy(self.metadata) if self.metadata is not None else None
        )
    
    def to_dict(self):
        """
        转换为字典（用于保存到JSON）
        
        返回：
            包含所有属性的字典
        """
        return {
            'path': self.path,
            'fitness': self.fitness,
            'behavior': self.behavior,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """
        从字典创建Individual对象（用于从JSON加载）
        
        参数：
            data: 包含Individual属性的字典
            
        返回：
            Individual对象
        """
        return cls(
            path=data['path'],
            fitness=data.get('fitness'),
            behavior=data.get('behavior'),
            metadata=data.get('metadata')
        )
    
    def __repr__(self):
        """字符串表示"""
        fitness_str = f"{self.fitness:.2f}" if self.fitness is not None else "None"
        return (f"Individual(points={len(self.path)}, "
                f"towers={self.num_towers()}, "
                f"fitness={fitness_str}, "
                f"behavior={self.behavior})")
