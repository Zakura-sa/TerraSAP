"""
轻量级特征调制模块
Lightweight Feature Modulation

完全兼容ASP设计的特征调制方案：
1. 极少参数（仅3个标量参数）
2. 在TSP生成后进行微调
3. 不干扰EMA机制和原型分类器
4. 推理时完全关闭
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class LightweightFeatureModulation(nn.Module):
    """
    轻量级特征调制模块
    在TSP生成过程中进行微调，完全兼容ASP的EMA机制
    """
    
    def __init__(self, feature_dim=768, modulation_strength=0.1, args=None):
        super().__init__()
        self.modulation_strength = modulation_strength
        self.feature_dim = feature_dim

        # 极轻量的调制参数（仅3个标量）
        self.spatial_scale = nn.Parameter(torch.ones(1))
        self.semantic_scale = nn.Parameter(torch.ones(1))
        self.consistency_weight = nn.Parameter(torch.tensor(0.5))

        # 协同优化：空间上下文感知调制（阶段1）
        self.enable_spatial_context_modulation = args.get('enable_spatial_context_modulation', False) if args else False
        if self.enable_spatial_context_modulation:
            # 空间上下文融合层（极少参数）
            spatial_context_dim = args.get('spatial_context_dim', 128) if args else 128
            self.context_fusion_layer = nn.Linear(spatial_context_dim, feature_dim)
            self.context_gate = nn.Parameter(torch.tensor(0.2))  # 优化：增强上下文影响强度
            # 添加空间上下文的自适应权重
            self.spatial_context_weight = nn.Parameter(torch.tensor(1.0))
            logging.info(f"✅ 空间上下文感知调制已启用: context_dim={spatial_context_dim}, 参数数量={spatial_context_dim * feature_dim + feature_dim + 2}")
        else:
            self.context_fusion_layer = None
            self.context_gate = None
            self.spatial_context_weight = None

        # 统计信息
        self._modulation_count = 0
        self._total_calls = 0
        self._context_modulation_count = 0

        logging.info(f"🔧 轻量级特征调制初始化:")
        logging.info(f"  特征维度: {feature_dim}")
        logging.info(f"  调制强度: {modulation_strength}")
        base_params = 3
        context_params = (spatial_context_dim * feature_dim + feature_dim + 1) if self.enable_spatial_context_modulation else 0
        logging.info(f"  可训练参数: {base_params + context_params}个 (基础{base_params} + 上下文{context_params})")
        
    def forward(self, tsp, training=True):
        """
        对TSP进行轻量级调制
        
        Args:
            tsp: 原始TSP [B, depth, prompt_length, feature_dim]
            training: 是否为训练模式
            
        Returns:
            调制后的TSP，形状与输入相同
        """
        self._total_calls += 1
        
        if not training:
            return tsp  # 推理时不调制，保持确定性
            
        self._modulation_count += 1
        
        # 输入验证
        if len(tsp.shape) != 4:
            logging.warning(f"TSP形状异常: {tsp.shape}，跳过调制")
            return tsp
            
        batch_size, depth, prompt_length, feature_dim = tsp.shape
        
        # 1. 空间一致性调制（无参数）
        spatial_modulated = self._spatial_consistency_modulation(tsp)
        
        # 2. 语义平滑调制（无参数）
        semantic_modulated = self._semantic_smoothing_modulation(tsp)
        
        # 3. 自适应融合（仅3个参数）
        # 确保权重为正数
        spatial_w = torch.abs(self.spatial_scale)
        semantic_w = torch.abs(self.semantic_scale)
        consistency_w = torch.abs(self.consistency_weight)
        
        total_weight = spatial_w + semantic_w + consistency_w + 1e-8  # 避免除零
        
        modulated_tsp = (
            spatial_w * spatial_modulated + 
            semantic_w * semantic_modulated + 
            consistency_w * tsp
        ) / total_weight
        
        # 4. 强度控制
        final_tsp = (1 - self.modulation_strength) * tsp + self.modulation_strength * modulated_tsp
        
        return final_tsp
    
    def _spatial_consistency_modulation(self, tsp):
        """空间一致性调制（无参数操作）"""
        batch_size, depth, prompt_length, feature_dim = tsp.shape
        
        # 在prompt_length维度进行平滑
        # 重塑为 [B*depth, feature_dim, prompt_length] 进行1D卷积（确保内存连续）
        reshaped = tsp.contiguous().view(batch_size * depth, prompt_length, feature_dim).transpose(1, 2)
        
        # 使用平均池化进行平滑
        if prompt_length >= 3:
            smoothed = F.avg_pool1d(reshaped, kernel_size=3, stride=1, padding=1)
        else:
            smoothed = reshaped  # 序列太短时不进行平滑
        
        # 恢复原始形状（确保内存连续）
        smoothed = smoothed.transpose(1, 2).contiguous().view(batch_size, depth, prompt_length, feature_dim)
        
        return smoothed
    
    def _semantic_smoothing_modulation(self, tsp):
        """语义平滑调制（无参数操作）"""
        # 在feature_dim维度进行层归一化
        normalized = F.layer_norm(tsp, [tsp.size(-1)])
        
        # 添加微小的结构化噪声（仅在训练时）
        if self.training:
            # 生成空间相关的噪声
            noise = torch.randn_like(tsp) * 0.01
            
            # 对噪声进行空间平滑，使其具有空间一致性
            batch_size, depth, prompt_length, feature_dim = noise.shape
            
            if prompt_length >= 3:
                # 重塑并平滑噪声（确保内存连续）
                noise_reshaped = noise.contiguous().view(batch_size * depth, prompt_length, feature_dim).transpose(1, 2)
                smoothed_noise = F.avg_pool1d(noise_reshaped, kernel_size=3, stride=1, padding=1)
                noise = smoothed_noise.transpose(1, 2).contiguous().view(batch_size, depth, prompt_length, feature_dim)
            
            normalized = normalized + noise
            
        return normalized

    def _spatial_context_guided_modulation(self, tsp, spatial_context):
        """空间上下文引导的调制（协同优化阶段1 - 优化版）"""
        if spatial_context is None or self.context_fusion_layer is None:
            return tsp

        batch_size, depth, prompt_length, feature_dim = tsp.shape

        # 将空间上下文投影到特征维度
        context_features = self.context_fusion_layer(spatial_context)  # [B, feature_dim]

        # 优化：添加层归一化提升稳定性
        context_features = F.layer_norm(context_features, [feature_dim])

        # 扩展上下文特征到TSP的形状
        context_expanded = context_features.unsqueeze(1).unsqueeze(2).expand(
            batch_size, depth, prompt_length, feature_dim
        )

        # 优化：使用自适应权重和门控机制
        gate_weight = torch.sigmoid(self.context_gate)
        spatial_weight = torch.abs(self.spatial_context_weight) if self.spatial_context_weight is not None else 1.0

        # 多种融合策略
        # 1. 加性融合（原有）
        additive_modulation = tsp + gate_weight * spatial_weight * context_expanded

        # 2. 乘性融合（新增）
        multiplicative_modulation = tsp * (1.0 + 0.1 * gate_weight * spatial_weight * torch.tanh(context_expanded))

        # 3. 混合融合
        alpha = 0.7  # 加性权重
        context_modulated = alpha * additive_modulation + (1 - alpha) * multiplicative_modulation

        return context_modulated

    def forward_with_spatial_context(self, tsp, spatial_context=None, training=True):
        """
        带空间上下文的特征调制（协同优化阶段1）

        Args:
            tsp: 原始TSP [B, depth, prompt_length, feature_dim]
            spatial_context: 空间上下文向量 [B, spatial_context_dim] 或 None
            training: 是否为训练模式

        Returns:
            调制后的TSP，形状与输入相同
        """
        self._total_calls += 1

        if not training:
            return tsp  # 推理时不调制，保持确定性

        self._modulation_count += 1

        # 输入验证
        if len(tsp.shape) != 4:
            logging.warning(f"TSP形状异常: {tsp.shape}，跳过调制")
            return tsp

        # 1. 空间上下文引导调制（如果启用且有上下文）
        if self.enable_spatial_context_modulation and spatial_context is not None:
            self._context_modulation_count += 1
            context_modulated = self._spatial_context_guided_modulation(tsp, spatial_context)
        else:
            context_modulated = tsp

        # 2. 原有的调制流程
        batch_size, depth, prompt_length, feature_dim = context_modulated.shape

        # 空间一致性调制
        spatial_modulated = self._spatial_consistency_modulation(context_modulated)

        # 语义平滑调制
        semantic_modulated = self._semantic_smoothing_modulation(context_modulated)

        # 自适应融合
        spatial_w = torch.abs(self.spatial_scale)
        semantic_w = torch.abs(self.semantic_scale)
        consistency_w = torch.abs(self.consistency_weight)

        total_weight = spatial_w + semantic_w + consistency_w + 1e-8

        modulated_tsp = (
            spatial_w * spatial_modulated +
            semantic_w * semantic_modulated +
            consistency_w * context_modulated
        ) / total_weight

        # 强度控制
        final_tsp = (1 - self.modulation_strength) * tsp + self.modulation_strength * modulated_tsp

        return final_tsp

    def get_modulation_stats(self):
        """获取调制统计信息"""
        stats = {
            'total_calls': self._total_calls,
            'modulation_count': self._modulation_count,
            'modulation_ratio': self._modulation_count / max(self._total_calls, 1),
            'parameters': {
                'spatial_scale': self.spatial_scale.item(),
                'semantic_scale': self.semantic_scale.item(),
                'consistency_weight': self.consistency_weight.item()
            },
            'modulation_strength': self.modulation_strength
        }

        # 添加空间上下文调制统计
        if self.enable_spatial_context_modulation:
            stats['context_modulation_count'] = self._context_modulation_count
            stats['context_modulation_ratio'] = self._context_modulation_count / max(self._modulation_count, 1)
            stats['context_gate_weight'] = self.context_gate.item() if self.context_gate is not None else 0.0

        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self._modulation_count = 0
        self._total_calls = 0
        self._context_modulation_count = 0
    
    def set_modulation_strength(self, strength):
        """动态调整调制强度"""
        old_strength = self.modulation_strength
        self.modulation_strength = max(0.0, min(1.0, strength))  # 限制在[0,1]范围
        logging.info(f"调制强度调整: {old_strength:.3f} -> {self.modulation_strength:.3f}")
    
    def get_parameter_count(self):
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FeatureModulationConfig:
    """特征调制配置类"""
    
    def __init__(self, args):
        self.enable_feature_modulation = args.get('enable_feature_modulation', False)
        self.modulation_strength = args.get('modulation_strength', 0.05)
        self.adaptive_strength = args.get('adaptive_modulation_strength', False)
        self.min_strength = args.get('min_modulation_strength', 0.01)
        self.max_strength = args.get('max_modulation_strength', 0.2)
        
    def create_modulation_module(self, feature_dim=768):
        """创建特征调制模块"""
        if not self.enable_feature_modulation:
            return None
            
        return LightweightFeatureModulation(
            feature_dim=feature_dim,
            modulation_strength=self.modulation_strength
        )
    
    def log_config(self):
        """记录配置信息"""
        if self.enable_feature_modulation:
            logging.info("✅ 轻量级特征调制配置:")
            logging.info(f"  启用状态: {self.enable_feature_modulation}")
            logging.info(f"  调制强度: {self.modulation_strength}")
            logging.info(f"  自适应强度: {self.adaptive_strength}")
            if self.adaptive_strength:
                logging.info(f"  强度范围: [{self.min_strength}, {self.max_strength}]")
        else:
            logging.info("轻量级特征调制未启用")


def test_lightweight_modulation():
    """测试轻量级特征调制模块"""
    print("🔧 测试轻量级特征调制模块...")
    
    # 创建测试数据
    batch_size, depth, prompt_length, feature_dim = 4, 12, 3, 768
    test_tsp = torch.randn(batch_size, depth, prompt_length, feature_dim)
    
    # 创建调制模块
    modulation = LightweightFeatureModulation(feature_dim=feature_dim, modulation_strength=0.1)
    
    print(f"输入TSP形状: {test_tsp.shape}")
    print(f"参数数量: {modulation.get_parameter_count()}")
    
    # 测试训练模式
    modulation.train()
    modulated_tsp = modulation(test_tsp, training=True)
    
    print(f"调制后TSP形状: {modulated_tsp.shape}")
    print(f"形状一致性: {'✅' if modulated_tsp.shape == test_tsp.shape else '❌'}")
    
    # 测试推理模式
    modulation.eval()
    eval_tsp = modulation(test_tsp, training=False)
    
    print(f"推理模式一致性: {'✅' if torch.equal(eval_tsp, test_tsp) else '❌'}")
    
    # 测试统计信息
    stats = modulation.get_modulation_stats()
    print(f"调制统计: {stats}")
    
    print("✅ 轻量级特征调制模块测试完成")


if __name__ == "__main__":
    test_lightweight_modulation()
