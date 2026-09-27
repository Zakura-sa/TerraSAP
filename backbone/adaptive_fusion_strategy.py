"""
自适应融合策略
Adaptive Fusion Strategy

独立的双路径EMA自适应融合工具；未接入主训练流程。
根据任务阶段和性能表现动态调整原始和调制TSP的权重
"""

import torch
import torch.nn as nn
import logging
import numpy as np


class AdaptiveFusionStrategy(nn.Module):
    """
    自适应融合策略类
    
    根据任务ID、性能历史和训练阶段动态调整双路径EMA的融合权重
    """
    
    def __init__(self, args=None):
        super().__init__()
        
        # 融合策略配置
        self.strategy = args.get('adaptive_fusion_strategy', 'linear') if args else 'linear'
        self.adaptation_rate = args.get('fusion_adaptation_rate', 0.1) if args else 0.1
        self.initial_weight = args.get('dual_ema_fusion_weight', 0.5) if args else 0.5
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(self.initial_weight))
        
        # 性能历史记录
        self.performance_history = []
        self.task_weights = {}  # 每个任务的最优权重
        self.current_task_id = 0
        
        # 统计信息
        self.adaptation_count = 0
        self.weight_history = []
        
        logging.info(f"🔧 自适应融合策略初始化:")
        logging.info(f"  策略类型: {self.strategy}")
        logging.info(f"  适应率: {self.adaptation_rate}")
        logging.info(f"  初始权重: {self.initial_weight}")
        
    def set_task_id(self, task_id):
        """设置当前任务ID"""
        self.current_task_id = task_id
        
        # 如果有该任务的历史最优权重，使用它
        if task_id in self.task_weights:
            optimal_weight = self.task_weights[task_id]
            self.fusion_weight.data.fill_(optimal_weight)
            logging.info(f"🎯 任务{task_id}: 使用历史最优权重 {optimal_weight:.4f}")
        else:
            # 新任务，使用初始权重
            self.fusion_weight.data.fill_(self.initial_weight)
            logging.info(f"🆕 任务{task_id}: 使用初始权重 {self.initial_weight:.4f}")
    
    def compute_fusion_weights(self, performance_metrics=None):
        """
        计算融合权重
        
        Args:
            performance_metrics: 性能指标字典，包含accuracy, loss等
            
        Returns:
            fusion_weight: 融合权重 [0, 1]，越接近1越偏向原始TSP
        """
        current_weight = torch.sigmoid(self.fusion_weight).item()  # 确保在[0,1]范围
        
        if self.strategy == 'linear':
            # 线性策略：根据任务ID线性调整
            # 早期任务更依赖原始TSP，后期任务更依赖调制TSP
            task_factor = max(0.1, 1.0 - self.current_task_id * 0.15)
            adapted_weight = current_weight * task_factor
            
        elif self.strategy == 'exponential':
            # 指数衰减策略
            decay_factor = np.exp(-self.current_task_id * 0.2)
            adapted_weight = current_weight * decay_factor
            
        elif self.strategy == 'performance_based':
            # 基于性能的自适应策略
            if performance_metrics and len(self.performance_history) > 0:
                current_acc = performance_metrics.get('accuracy', 0.0)
                recent_avg = np.mean(self.performance_history[-3:]) if len(self.performance_history) >= 3 else current_acc
                
                if current_acc > recent_avg:
                    # 性能提升，保持当前权重
                    adapted_weight = current_weight
                else:
                    # 性能下降，调整权重
                    adapted_weight = current_weight + self.adaptation_rate * (0.5 - current_weight)
            else:
                adapted_weight = current_weight
                
        elif self.strategy == 'dynamic':
            # 动态策略：结合任务ID和性能
            task_factor = max(0.2, 1.0 - self.current_task_id * 0.1)
            
            if performance_metrics and len(self.performance_history) > 1:
                current_acc = performance_metrics.get('accuracy', 0.0)
                prev_acc = self.performance_history[-1]
                performance_factor = 1.0 + 0.1 * (current_acc - prev_acc)
                performance_factor = max(0.5, min(1.5, performance_factor))
            else:
                performance_factor = 1.0
                
            adapted_weight = current_weight * task_factor * performance_factor
            
        else:
            # 默认：固定权重
            adapted_weight = current_weight
        
        # 确保权重在合理范围内
        adapted_weight = max(0.1, min(0.9, adapted_weight))
        
        return adapted_weight
    
    def update_fusion_history(self, performance_metrics):
        """
        更新性能历史和融合权重
        
        Args:
            performance_metrics: 性能指标字典
        """
        if performance_metrics:
            accuracy = performance_metrics.get('accuracy', 0.0)
            self.performance_history.append(accuracy)
            
            # 保持历史记录长度
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
            
            # 记录当前权重
            current_weight = torch.sigmoid(self.fusion_weight).item()
            self.weight_history.append(current_weight)
            
            # 更新任务最优权重
            if self.current_task_id not in self.task_weights or accuracy > max(self.performance_history[:-1], default=0):
                self.task_weights[self.current_task_id] = current_weight
                logging.info(f"📈 任务{self.current_task_id}: 更新最优权重 {current_weight:.4f} (准确率: {accuracy:.4f})")
            
            self.adaptation_count += 1
    
    def get_fusion_stats(self):
        """获取融合策略统计信息"""
        current_weight = torch.sigmoid(self.fusion_weight).item()
        
        stats = {
            'strategy': self.strategy,
            'current_weight': current_weight,
            'current_task_id': self.current_task_id,
            'adaptation_count': self.adaptation_count,
            'performance_history_length': len(self.performance_history),
            'task_weights': dict(self.task_weights),
            'recent_performance': self.performance_history[-3:] if len(self.performance_history) >= 3 else self.performance_history
        }
        
        return stats
    
    def reset_for_new_task(self, task_id):
        """为新任务重置状态"""
        self.set_task_id(task_id)
        # 不清空性能历史，保持跨任务的学习经验
        logging.info(f"🔄 融合策略重置为任务{task_id}")
    
    def forward(self, original_tsp, modulated_tsp, performance_metrics=None):
        """
        前向传播：计算融合后的TSP
        
        Args:
            original_tsp: 原始TSP
            modulated_tsp: 调制TSP
            performance_metrics: 性能指标（可选）
            
        Returns:
            fused_tsp: 融合后的TSP
        """
        # 计算自适应权重
        fusion_weight = self.compute_fusion_weights(performance_metrics)
        
        # 融合TSP
        fused_tsp = fusion_weight * original_tsp + (1 - fusion_weight) * modulated_tsp
        
        return fused_tsp


def create_adaptive_fusion_strategy(args):
    """创建自适应融合策略实例"""
    if args.get('enable_dual_path_ema', False):
        return AdaptiveFusionStrategy(args)
    else:
        return None
