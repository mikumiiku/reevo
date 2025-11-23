"""
Lineage Tracker for Genetic Algorithm
Inspired by Clade-Metaproductivity (CMP) from HGM paper, but simplified for practical use.
"""

import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class LineageTracker:
    """
    追踪遗传算法中个体的谱系关系，计算谱系质量分数。
    
    核心概念：
    - Lineage (谱系): 从同一祖先衍生的所有后代
    - Clade (分支): 某个个体及其所有后代
    - CMP (Clade-Metaproductivity): 谱系的自我改进潜力
    """
    
    def __init__(self):
        self.individuals: Dict[str, dict] = {}  # id -> individual dict
        self.lineages: Dict[str, List[str]] = {}  # lineage_id -> [individual_ids]
        self.parent_child_map: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        
        logging.info("LineageTracker initialized")
    
    def register_individual(self, individual: dict, parents: Optional[List[dict]] = None) -> None:
        """
        注册新个体并建立谱系关系
        
        Args:
            individual: 个体字典，必须包含基本信息
            parents: 父代个体列表，None表示初始种子
        """
        # 确保个体有唯一ID
        if 'id' not in individual:
            individual['id'] = str(uuid.uuid4())
        
        ind_id = individual['id']
        self.individuals[ind_id] = individual
        
        if parents is None or len(parents) == 0:
            # 初始种子个体，自己就是谱系根
            individual['lineage_id'] = ind_id
            individual['parent_ids'] = []
            individual['generation'] = 0
            individual['birth_iteration'] = individual.get('birth_iteration', 0)
            
            self.lineages[ind_id] = [ind_id]
            logging.debug(f"Registered seed individual {ind_id[:8]}")
        else:
            # 继承谱系ID（从第一个父代）
            individual['lineage_id'] = parents[0]['lineage_id']
            individual['parent_ids'] = [p['id'] for p in parents]
            individual['generation'] = max(p.get('generation', 0) for p in parents) + 1
            individual['birth_iteration'] = individual.get('birth_iteration', 0)
            
            # 更新谱系
            lineage_id = individual['lineage_id']
            if lineage_id not in self.lineages:
                self.lineages[lineage_id] = []
            self.lineages[lineage_id].append(ind_id)
            
            # 更新父子关系
            for parent in parents:
                parent_id = parent['id']
                if parent_id not in self.parent_child_map:
                    self.parent_child_map[parent_id] = []
                self.parent_child_map[parent_id].append(ind_id)
            
            logging.debug(f"Registered individual {ind_id[:8]} from lineage {lineage_id[:8]}, generation {individual['generation']}")
        
        # 初始化谱系统计
        individual['clade_best_obj'] = individual.get('obj', float('inf'))
        individual['clade_avg_obj'] = individual.get('obj', float('inf'))
        individual['clade_size'] = 1
        individual['clade_diversity'] = 0.0
    
    def get_clade_members(self, individual_id: str) -> List[dict]:
        """
        获取某个体的所有后代（包括自己）
        
        Args:
            individual_id: 个体ID
            
        Returns:
            该个体的分支中所有成员
        """
        clade = [self.individuals[individual_id]]
        
        # BFS遍历所有后代
        queue = [individual_id]
        visited = {individual_id}
        
        while queue:
            current_id = queue.pop(0)
            if current_id in self.parent_child_map:
                for child_id in self.parent_child_map[current_id]:
                    if child_id not in visited:
                        visited.add(child_id)
                        clade.append(self.individuals[child_id])
                        queue.append(child_id)
        
        return clade
    
    def update_clade_stats(self, individual_id: str) -> None:
        """
        更新个体所在分支的统计信息
        
        Args:
            individual_id: 个体ID
        """
        if individual_id not in self.individuals:
            logging.warning(f"Individual {individual_id} not found in tracker")
            return
        
        ind = self.individuals[individual_id]
        
        # 获取整个谱系的成员（不仅是分支）
        lineage_id = ind['lineage_id']
        lineage_members = [self.individuals[i] for i in self.lineages.get(lineage_id, [])]
        
        # 只统计成功执行的个体
        valid_members = [m for m in lineage_members if m.get('exec_success', False)]
        
        if not valid_members:
            ind['clade_best_obj'] = float('inf')
            ind['clade_avg_obj'] = float('inf')
            ind['clade_size'] = 0
            ind['clade_diversity'] = 0.0
            return
        
        objs = [m['obj'] for m in valid_members]
        ind['clade_best_obj'] = min(objs)
        ind['clade_avg_obj'] = np.mean(objs)
        ind['clade_size'] = len(valid_members)
        
        # 多样性度量：目标值的标准差（归一化）
        if len(objs) > 1:
            ind['clade_diversity'] = np.std(objs) / (np.mean(objs) + 1e-10)
        else:
            ind['clade_diversity'] = 0.0
    
    def compute_clade_score(self, individual_id: str, 
                           alpha: float = 0.5, 
                           beta: float = 0.3, 
                           gamma: float = 0.2) -> float:
        """
        计算谱系分数（越小越好，用于最小化问题）
        
        Score = α * best_obj + β * avg_obj - γ * diversity
        
        Args:
            individual_id: 个体ID
            alpha: 最优后代权重
            beta: 平均表现权重
            gamma: 多样性权重（多样性越高越好，所以用负号）
            
        Returns:
            谱系分数
        """
        if individual_id not in self.individuals:
            return float('inf')
        
        ind = self.individuals[individual_id]
        
        best_score = ind.get('clade_best_obj', float('inf'))
        avg_score = ind.get('clade_avg_obj', float('inf'))
        diversity = ind.get('clade_diversity', 0.0)
        
        # 多样性是好的，所以用负号
        clade_score = alpha * best_score + beta * avg_score - gamma * diversity
        
        return clade_score
    
    def get_lineage_context(self, individual_id: str) -> str:
        """
        获取个体的谱系上下文信息，用于LLM提示
        
        Args:
            individual_id: 个体ID
            
        Returns:
            谱系上下文描述字符串
        """
        if individual_id not in self.individuals:
            return "Individual not found in lineage tracker."
        
        ind = self.individuals[individual_id]
        lineage_id = ind['lineage_id']
        lineage_members = [self.individuals[i] for i in self.lineages.get(lineage_id, [])]
        valid_members = [m for m in lineage_members if m.get('exec_success', False)]
        
        if not valid_members:
            return f"Lineage ID: {lineage_id[:8]}...\nGeneration: {ind['generation']}\nNo successful ancestors in this lineage yet."
        
        best_ancestor = min(valid_members, key=lambda x: x['obj'])
        objs = [m['obj'] for m in valid_members]
        
        context = f"""Lineage Information:
- Lineage ID: {lineage_id[:8]}...
- Generation: {ind['generation']}
- Lineage Size: {len(valid_members)} successful individuals
- Best Ancestor Score: {best_ancestor['obj']:.6f}
- Average Lineage Score: {np.mean(objs):.6f}
- Lineage Diversity: {np.std(objs):.6f}
- Current Individual Score: {ind.get('obj', 'N/A')}
"""
        return context
    
    def get_statistics(self) -> Dict:
        """
        获取整体统计信息
        
        Returns:
            统计信息字典
        """
        total_individuals = len(self.individuals)
        total_lineages = len(self.lineages)
        
        valid_individuals = [ind for ind in self.individuals.values() if ind.get('exec_success', False)]
        
        if valid_individuals:
            generations = [ind['generation'] for ind in valid_individuals]
            max_generation = max(generations)
            avg_generation = np.mean(generations)
            
            lineage_sizes = [len(self.lineages[lid]) for lid in self.lineages]
            avg_lineage_size = np.mean(lineage_sizes)
        else:
            max_generation = 0
            avg_generation = 0
            avg_lineage_size = 0
        
        stats = {
            'total_individuals': total_individuals,
            'valid_individuals': len(valid_individuals),
            'total_lineages': total_lineages,
            'max_generation': max_generation,
            'avg_generation': avg_generation,
            'avg_lineage_size': avg_lineage_size,
        }
        
        return stats
    
    def log_statistics(self) -> None:
        """记录统计信息到日志"""
        stats = self.get_statistics()
        logging.info(f"Lineage Statistics: {stats}")
