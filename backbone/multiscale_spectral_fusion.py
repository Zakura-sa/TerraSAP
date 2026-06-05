"""
多尺度光谱特征融合模块
MultiScale Spectral Fusion Module

方案一实现：针对遥感图像的多尺度和光谱特性优化
- 仅增加2个关键参数组：scale_weights, spectral_weights
- 在空间感知提示编码器之后、轻量级调制之前插入
- 完全兼容ASP核心设计、空间提示模块、轻量级调制模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class MultiScaleSpectralFusion(nn.Module):
    """
    多尺度光谱特征融合模块
    
    设计理念：
    1. 多尺度特征提取：使用不同尺度的池化捕获多层次空间信息
    2. 光谱特征增强：通过可学习权重增强光谱特征表达
    3. 轻量级融合：仅使用少量参数实现高效融合
    4. 遥感优化：专门针对遥感图像的特点设计
    """
    
    def __init__(self, feature_dim=768, args=None):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 从配置中获取参数
        if args:
            self.num_scales = args.get('multiscale_num_scales', 3)
            self.scales = args.get('multiscale_scales', [1, 2, 4])
            self.spectral_feature_ratio = args.get('spectral_feature_ratio', 0.25)
            self.fusion_weight = args.get('multiscale_fusion_weight', 0.3)
            self.spectral_enhancement_weight = args.get('spectral_enhancement_weight', 0.2)
            self.enable_spectral_weighting = args.get('enable_spectral_weighting', True)
            self.pooling_type = args.get('multiscale_pooling_type', 'adaptive_avg')
            self.fusion_mode = args.get('spectral_fusion_mode', 'weighted_sum')
        else:
            # 默认配置
            self.num_scales = 3
            self.scales = [1, 2, 4]
            self.spectral_feature_ratio = 0.25
            self.fusion_weight = 0.3
            self.spectral_enhancement_weight = 0.2
            self.enable_spectral_weighting = True
            self.pooling_type = 'adaptive_avg'
            self.fusion_mode = 'weighted_sum'
        
        # 核心参数1：多尺度权重 (仅num_scales个参数)
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        # 核心参数2：光谱权重 (仅feature_dim//4个参数，保持轻量级)
        spectral_dim = int(feature_dim * self.spectral_feature_ratio)
        if self.enable_spectral_weighting:
            self.spectral_weights = nn.Parameter(torch.ones(spectral_dim))
        else:
            self.register_parameter('spectral_weights', None)
        
        # 统计信息
        self._fusion_count = 0
        self._total_calls = 0
        
        # 计算参数数量
        param_count = self.num_scales
        if self.enable_spectral_weighting:
            param_count += spectral_dim
        
        logging.info(f"🔧 多尺度光谱特征融合模块初始化:")
        logging.info(f"  特征维度: {feature_dim}")
        logging.info(f"  多尺度数量: {self.num_scales}, 尺度: {self.scales}")
        logging.info(f"  光谱特征比例: {self.spectral_feature_ratio}")
        logging.info(f"  融合权重: {self.fusion_weight}")
        logging.info(f"  光谱增强权重: {self.spectral_enhancement_weight}")
        logging.info(f"  总参数数量: {param_count}")
        
    def forward(self, features, training=True):
        """
        多尺度光谱特征融合前向传播
        
        Args:
            features: 输入特征 [B, seq_len, feature_dim] 或 [B, depth, prompt_length, feature_dim]
            training: 是否为训练模式
            
        Returns:
            融合后的特征，形状与输入相同
        """
        self._total_calls += 1
        
        if not training:
            # 推理时可以选择性地应用融合（保持一致性）
            pass
            
        self._fusion_count += 1
        
        # 输入形状处理
        original_shape = features.shape
        if len(original_shape) == 4:
            # [B, depth, prompt_length, feature_dim] -> [B*depth, prompt_length, feature_dim]
            batch_size, depth, prompt_length, feature_dim = original_shape
            features = features.reshape(batch_size * depth, prompt_length, feature_dim)
            reshape_needed = True
        else:
            # [B, seq_len, feature_dim]
            reshape_needed = False
        
        batch_size, seq_len, feature_dim = features.shape
        
        # 1. 多尺度特征提取
        multiscale_features = self._extract_multiscale_features(features)
        
        # 2. 光谱特征增强
        spectral_enhanced_features = self._enhance_spectral_features(features)
        
        # 3. 多尺度光谱融合
        fused_features = self._fuse_multiscale_spectral(
            multiscale_features, spectral_enhanced_features, features
        )
        
        # 4. 恢复原始形状
        if reshape_needed:
            fused_features = fused_features.reshape(original_shape)
        
        return fused_features
    
    def _extract_multiscale_features(self, features):
        """
        提取多尺度特征
        
        Args:
            features: [B, seq_len, feature_dim]
            
        Returns:
            多尺度特征列表
        """
        batch_size, seq_len, feature_dim = features.shape
        multiscale_features = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # 原始尺度
                scaled_features = features
            else:
                # 对于小序列长度，直接使用1D池化避免2D重塑问题
                if seq_len <= 16:  # 对于小序列（如prompt_length=6），使用1D处理
                    # 使用1D平均池化
                    features_1d = features.transpose(1, 2)  # [B, feature_dim, seq_len]
                    if scale <= seq_len:
                        pooled_1d = F.avg_pool1d(features_1d, kernel_size=scale, stride=1, padding=scale//2)
                    else:
                        # 如果scale大于序列长度，直接使用全局平均池化
                        pooled_1d = F.adaptive_avg_pool1d(features_1d, seq_len)
                    scaled_features = pooled_1d.transpose(1, 2)  # [B, seq_len, feature_dim]
                else:
                    # 对于较长序列，使用2D池化
                    # 需要重塑为2D进行池化
                    # 假设seq_len可以重塑为接近正方形
                    h = w = int(seq_len ** 0.5)
                    if h * w != seq_len:
                        # 如果不是完全平方数，使用最接近的矩形
                        h = int(seq_len ** 0.5)
                        w = seq_len // h
                        if h * w < seq_len:
                            h += 1
                        # 填充到h*w
                        pad_len = h * w - seq_len
                        if pad_len > 0:
                            features_padded = F.pad(features, (0, 0, 0, pad_len))
                        else:
                            features_padded = features
                    else:
                        features_padded = features

                    # 重塑为2D: [B, feature_dim, h, w]
                    features_2d = features_padded.reshape(batch_size, h, w, feature_dim).permute(0, 3, 1, 2)

                    # 多尺度池化（添加边界检查）
                    target_h, target_w = max(1, h//scale), max(1, w//scale)
                    if self.pooling_type == 'adaptive_avg':
                        pooled = F.adaptive_avg_pool2d(features_2d, (target_h, target_w))
                        # 上采样回原始尺寸
                        if target_h > 0 and target_w > 0 and h > 0 and w > 0:
                            scaled_2d = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
                        else:
                            scaled_2d = features_2d  # 回退到原始特征
                    else:
                        # 使用平均池化
                        kernel_size = min(scale, min(h, w))  # 确保kernel_size不超过特征图尺寸
                        if kernel_size > 0:
                            scaled_2d = F.avg_pool2d(features_2d, kernel_size, stride=1, padding=kernel_size//2)
                        else:
                            scaled_2d = features_2d

                    # 重塑回序列格式: [B, seq_len, feature_dim]
                    scaled_features = scaled_2d.permute(0, 2, 3, 1).reshape(batch_size, -1, feature_dim)
                    # 裁剪到原始序列长度
                    scaled_features = scaled_features[:, :seq_len, :]
            
            multiscale_features.append(scaled_features)
        
        return multiscale_features
    
    def _enhance_spectral_features(self, features):
        """
        增强光谱特征
        
        Args:
            features: [B, seq_len, feature_dim]
            
        Returns:
            光谱增强的特征
        """
        if not self.enable_spectral_weighting or self.spectral_weights is None:
            return features
        
        batch_size, seq_len, feature_dim = features.shape
        spectral_dim = self.spectral_weights.size(0)
        
        # 选择部分特征维度进行光谱增强
        spectral_features = features[:, :, :spectral_dim]  # [B, seq_len, spectral_dim]
        
        # 应用可学习的光谱权重
        enhanced_spectral = spectral_features * self.spectral_weights.unsqueeze(0).unsqueeze(0)
        
        # 将增强的光谱特征与原始特征结合
        enhanced_features = features.clone()
        enhanced_features[:, :, :spectral_dim] = enhanced_spectral
        
        return enhanced_features
    
    def _fuse_multiscale_spectral(self, multiscale_features, spectral_features, original_features):
        """
        融合多尺度和光谱特征
        
        Args:
            multiscale_features: 多尺度特征列表
            spectral_features: 光谱增强特征
            original_features: 原始特征
            
        Returns:
            融合后的特征
        """
        # 1. 多尺度特征加权融合
        # 确保权重为正数并归一化
        normalized_weights = F.softmax(self.scale_weights, dim=0)

        multiscale_fused = torch.zeros_like(original_features)
        for i, (features, weight) in enumerate(zip(multiscale_features, normalized_weights)):
            # 确保特征形状匹配
            if features.shape != original_features.shape:
                # 如果形状不匹配，裁剪或填充到原始形状
                if features.size(1) > original_features.size(1):
                    features = features[:, :original_features.size(1), :]
                elif features.size(1) < original_features.size(1):
                    pad_len = original_features.size(1) - features.size(1)
                    features = F.pad(features, (0, 0, 0, pad_len))
            multiscale_fused += weight * features
        
        # 2. 多尺度与光谱特征融合
        if self.fusion_mode == 'weighted_sum':
            # 加权求和融合
            fused_features = (
                (1 - self.fusion_weight - self.spectral_enhancement_weight) * original_features +
                self.fusion_weight * multiscale_fused +
                self.spectral_enhancement_weight * spectral_features
            )
        elif self.fusion_mode == 'adaptive':
            # 自适应融合（可以扩展）
            fused_features = original_features + self.fusion_weight * (multiscale_fused - original_features)
            fused_features = fused_features + self.spectral_enhancement_weight * (spectral_features - fused_features)
        else:
            # 默认简单融合
            fused_features = original_features + self.fusion_weight * multiscale_fused
        
        return fused_features
    
    def get_parameter_count(self):
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_fusion_stats(self):
        """获取融合统计信息"""
        return {
            'total_calls': self._total_calls,
            'fusion_count': self._fusion_count,
            'parameter_count': self.get_parameter_count(),
            'num_scales': self.num_scales,
            'spectral_dim': self.spectral_weights.size(0) if self.spectral_weights is not None else 0
        }


class MultiScaleSpectralConfig:
    """多尺度光谱融合配置类"""
    
    def __init__(self, args):
        self.enable_multiscale_spectral_fusion = args.get('enable_multiscale_spectral_fusion', False)
        self.num_scales = args.get('multiscale_num_scales', 3)
        self.scales = args.get('multiscale_scales', [1, 2, 4])
        self.spectral_feature_ratio = args.get('spectral_feature_ratio', 0.25)
        self.fusion_weight = args.get('multiscale_fusion_weight', 0.3)
        self.spectral_enhancement_weight = args.get('spectral_enhancement_weight', 0.2)
        
    def create_fusion_module(self, feature_dim=768, args=None):
        """创建多尺度光谱融合模块"""
        if not self.enable_multiscale_spectral_fusion:
            return None
            
        return MultiScaleSpectralFusion(feature_dim=feature_dim, args=args)
