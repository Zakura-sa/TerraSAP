"""
新类感知分类器
New Class Aware Classifier

解决新类分类瓶颈的智能权重初始化方案
- 使用支持集特征计算类原型
- 用类原型初始化新类分类器权重
- 完全兼容fs_epoch=0约束
- 零参数开销，仅改进初始化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from torch.utils.data import DataLoader
from .linears import CosineLinear


class NewClassAwareClassifier(CosineLinear):
    """
    新类感知分类器
    
    核心改进：使用支持集特征的类原型来初始化新类分类器权重，
    而不是随机初始化，从而显著提升新类分类性能。
    """
    
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True, args=None):
        super().__init__(in_features, out_features, nb_proxy, to_reduce, sigma)

        # 初始化策略配置
        self.init_strategy = 'class_prototype'  # 'class_prototype', 'best_sample', 'enhanced_prototype'
        self.noise_scale = 0.1  # 增强原型的噪声尺度
        self.use_old_weight_fallback = True  # 是否使用旧权重平均作为回退

        # 协同优化：空间感知分类器初始化（阶段1）
        self.enable_spatial_aware_classifier_init = args.get('enable_spatial_aware_classifier_init', False) if args else False
        if self.enable_spatial_aware_classifier_init:
            logging.info(f"✅ 空间感知分类器初始化已启用")

        logging.info(f"🔧 新类感知分类器初始化:")
        logging.info(f"  输入特征维度: {in_features}")
        logging.info(f"  输出类别数: {out_features}")
        logging.info(f"  初始化策略: {self.init_strategy}")
        logging.info(f"  空间感知初始化: {self.enable_spatial_aware_classifier_init}")
        
    def smart_expand_with_support(self, new_classes, support_loader, backbone, device):
        """
        使用支持集智能扩展分类器（支持空间感知初始化）

        Args:
            new_classes: 新增类别数量
            support_loader: 支持集数据加载器
            backbone: 特征提取网络
            device: 设备
        """
        old_classes = self.out_features
        total_classes = old_classes + new_classes

        logging.info(f"🚀 智能扩展分类器: {old_classes} -> {total_classes} 类别")

        # 协同优化：提取空间感知的支持集特征
        if self.enable_spatial_aware_classifier_init:
            support_features, support_labels, spatial_contexts = self._extract_spatial_aware_features(
                support_loader, backbone, device
            )
        else:
            support_features, support_labels = self._extract_support_features(
                support_loader, backbone, device
            )
            spatial_contexts = None
        
        logging.info(f"  支持集特征形状: {support_features.shape}")
        logging.info(f"  支持集标签范围: {support_labels.min().item()}-{support_labels.max().item()}")
        
        # 创建新权重矩阵
        new_weight = torch.zeros(total_classes, self.in_features).to(self.weight.device)
        new_weight[:old_classes] = self.weight.data  # 保留旧权重
        
        # 统计初始化情况
        init_stats = {'prototype': 0, 'fallback': 0, 'random': 0}
        
        # 智能初始化新类权重（支持空间感知）
        for new_class_idx in range(old_classes, total_classes):
            # 注意：support_labels中的标签是全局标签，直接使用
            class_mask = (support_labels == new_class_idx)

            if class_mask.sum() > 0:
                class_features = support_features[class_mask]

                # 协同优化：使用空间感知的权重计算
                if self.enable_spatial_aware_classifier_init and spatial_contexts is not None:
                    class_spatial_contexts = spatial_contexts[class_mask]
                    init_weight = self._compute_spatial_aware_class_weight(class_features, class_spatial_contexts)
                    logging.info(f"  类别 {new_class_idx}: 使用 {class_mask.sum().item()} 个样本的空间感知原型初始化")
                else:
                    init_weight = self._compute_class_init_weight(class_features)
                    logging.info(f"  类别 {new_class_idx}: 使用 {class_mask.sum().item()} 个样本的标准原型初始化")

                new_weight[new_class_idx] = F.normalize(init_weight, p=2, dim=0)
                init_stats['prototype'] += 1
            else:
                # 回退策略
                if self.use_old_weight_fallback and old_classes > 0:
                    # 使用旧类权重的平均作为初始化
                    old_weight_mean = self.weight.data.mean(0)
                    new_weight[new_class_idx] = F.normalize(old_weight_mean, p=2, dim=0)
                    init_stats['fallback'] += 1
                    logging.info(f"  类别 {new_class_idx}: 使用旧权重平均初始化（回退）")
                else:
                    # 最后回退：随机初始化
                    stdv = 1. / math.sqrt(self.in_features)
                    new_weight[new_class_idx].uniform_(-stdv, stdv)
                    new_weight[new_class_idx] = F.normalize(new_weight[new_class_idx], p=2, dim=0)
                    init_stats['random'] += 1
                    logging.info(f"  类别 {new_class_idx}: 随机初始化（最后回退）")
        
        # 更新分类器权重
        self.weight = nn.Parameter(new_weight)
        self.out_features = total_classes
        
        logging.info(f"✅ 分类器扩展完成:")
        logging.info(f"  原型初始化: {init_stats['prototype']} 个类别")
        logging.info(f"  回退初始化: {init_stats['fallback']} 个类别")
        logging.info(f"  随机初始化: {init_stats['random']} 个类别")
        
    def _extract_support_features(self, support_loader, backbone, device):
        """
        提取支持集的特征表示
        
        Args:
            support_loader: 支持集数据加载器
            backbone: 特征提取网络
            device: 设备
            
        Returns:
            features: 支持集特征 [N, feature_dim]
            labels: 支持集标签 [N]
        """
        backbone.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for _, inputs, targets in support_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 提取特征（使用backbone的forward_features方法）
                if hasattr(backbone, 'backbone'):
                    # ASP架构：backbone.backbone是实际的特征提取器
                    features, _ = backbone.backbone(inputs, perturb_var=0)  # 获取特征，忽略KL
                else:
                    # 其他架构的回退
                    features = backbone(inputs)['features']
                
                features_list.append(features)
                labels_list.append(targets)
        
        return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

    def _extract_spatial_aware_features(self, support_loader, backbone, device):
        """
        提取空间感知的支持集特征（协同优化阶段1）

        Args:
            support_loader: 支持集数据加载器
            backbone: 特征提取网络
            device: 设备

        Returns:
            features: 提取的特征 [N, feature_dim]
            labels: 对应的标签 [N]
            spatial_contexts: 空间上下文向量 [N, spatial_context_dim] 或 None
        """
        features_list = []
        labels_list = []
        spatial_contexts_list = []

        backbone.eval()
        with torch.no_grad():
            for _, inputs, targets in support_loader:  # 修复：正确的数据格式解包
                inputs = inputs.to(device)
                targets = targets.to(device)

                # 尝试提取空间感知特征
                spatial_context = None

                if hasattr(backbone, 'backbone'):
                    # ASP架构：检查是否支持空间上下文
                    asp_backbone = backbone.backbone

                    # 检查是否启用了空间上下文传递
                    if (hasattr(asp_backbone, 'enable_spatial_context_pipeline') and
                        asp_backbone.enable_spatial_context_pipeline and
                        hasattr(asp_backbone, 'Prompt_Encoder') and
                        hasattr(asp_backbone.Prompt_Encoder, 'forward_with_context')):

                        # 使用空间感知的特征提取
                        try:
                            # 获取patch嵌入
                            x_patches = asp_backbone.patch_embed(inputs)
                            x_features = x_patches.mean(dim=1)  # 全局特征
                            tip_features = asp_backbone.TIP.mean(dim=0).mean(dim=0).unsqueeze(0).expand(inputs.size(0), -1)

                            # 生成空间上下文
                            spatial_context = asp_backbone.Prompt_Encoder.generate_spatial_context(tip_features, x_features)

                            # 正常的特征提取
                            features, _ = asp_backbone(inputs, perturb_var=0)

                        except Exception as e:
                            logging.warning(f"空间感知特征提取失败，回退到标准方法: {str(e)}")
                            features, _ = asp_backbone(inputs, perturb_var=0)
                            spatial_context = None
                    else:
                        # 标准特征提取
                        features, _ = asp_backbone(inputs, perturb_var=0)
                else:
                    # 其他架构的回退
                    features = backbone(inputs)['features']

                features_list.append(features)
                labels_list.append(targets)

                if spatial_context is not None:
                    spatial_contexts_list.append(spatial_context)
                else:
                    # 如果没有空间上下文，添加None占位符
                    spatial_contexts_list.append(torch.zeros(inputs.size(0), 1).to(device))

        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        # 处理空间上下文
        if len(spatial_contexts_list) > 0 and spatial_contexts_list[0] is not None:
            all_spatial_contexts = torch.cat(spatial_contexts_list, dim=0)
            # 检查是否为有效的空间上下文（不是占位符）
            if all_spatial_contexts.size(1) > 1:
                return all_features, all_labels, all_spatial_contexts

        return all_features, all_labels, None

    def _compute_class_init_weight(self, class_features):
        """
        计算类别的初始化权重
        
        Args:
            class_features: 类别的特征 [N, feature_dim]
            
        Returns:
            init_weight: 初始化权重 [feature_dim]
        """
        if self.init_strategy == 'class_prototype':
            # 方法1：类原型（特征均值）
            return class_features.mean(0)
            
        elif self.init_strategy == 'best_sample':
            # 方法2：最优样本（最接近类中心的样本）
            center = class_features.mean(0)
            distances = torch.norm(class_features - center.unsqueeze(0), dim=1)
            best_sample_idx = torch.argmin(distances)
            return class_features[best_sample_idx]
            
        elif self.init_strategy == 'enhanced_prototype':
            # 方法3：增强原型（添加小幅随机扰动）
            prototype = class_features.mean(0)
            noise = torch.randn_like(prototype) * self.noise_scale
            return prototype + noise

        else:
            # 默认：类原型
            return class_features.mean(0)

    def _compute_spatial_aware_class_weight(self, class_features, class_spatial_contexts):
        """
        计算空间感知的类别初始化权重（协同优化阶段1）

        Args:
            class_features: 类别的特征 [N, feature_dim]
            class_spatial_contexts: 类别的空间上下文 [N, spatial_context_dim]

        Returns:
            init_weight: 空间感知的初始化权重 [feature_dim]
        """
        if class_spatial_contexts is None:
            # 回退到标准方法
            return self._compute_class_init_weight(class_features)

        # 基础类原型
        base_prototype = class_features.mean(0)

        # 空间上下文的影响
        spatial_prototype = class_spatial_contexts.mean(0)  # [spatial_context_dim]

        # 优化的空间感知融合策略
        # 1. 计算空间上下文的统计特征
        spatial_mean = spatial_prototype.mean()
        spatial_std = spatial_prototype.std() + 1e-8

        # 2. 将空间上下文投影到特征维度（改进版本）
        if spatial_prototype.size(0) < base_prototype.size(0):
            # 使用线性插值而非简单重复
            indices = torch.linspace(0, spatial_prototype.size(0) - 1, base_prototype.size(0))
            spatial_weight = torch.zeros_like(base_prototype)
            for i, idx in enumerate(indices):
                idx_floor = int(idx.floor())
                idx_ceil = min(int(idx.ceil()), spatial_prototype.size(0) - 1)
                weight = idx - idx_floor
                spatial_weight[i] = (1 - weight) * spatial_prototype[idx_floor] + weight * spatial_prototype[idx_ceil]
        else:
            # 使用平均池化而非截断
            pool_size = spatial_prototype.size(0) // base_prototype.size(0)
            spatial_weight = F.avg_pool1d(
                spatial_prototype.unsqueeze(0).unsqueeze(0),
                kernel_size=pool_size,
                stride=pool_size
            ).squeeze().squeeze()
            if spatial_weight.size(0) < base_prototype.size(0):
                # 补齐剩余维度
                remaining = base_prototype.size(0) - spatial_weight.size(0)
                spatial_weight = torch.cat([spatial_weight, spatial_prototype[-remaining:]], dim=0)

        # 3. 多层次空间感知融合
        # 归一化空间权重
        spatial_weight_norm = torch.tanh(spatial_weight)  # 使用tanh获得更平滑的权重

        # 基于空间统计的自适应调制强度
        modulation_strength = 0.15 * torch.sigmoid(spatial_std)  # 根据空间变化调整强度

        # 空间感知的原型融合（多种策略）
        # 策略1：加性调制
        additive_prototype = base_prototype + modulation_strength * spatial_weight_norm

        # 策略2：乘性调制
        multiplicative_prototype = base_prototype * (1.0 + modulation_strength * spatial_weight_norm)

        # 策略3：混合调制
        alpha = 0.6  # 加性权重
        spatial_aware_prototype = alpha * additive_prototype + (1 - alpha) * multiplicative_prototype

        return spatial_aware_prototype
    
    def get_init_stats(self):
        """获取初始化统计信息"""
        return {
            'strategy': self.init_strategy,
            'noise_scale': self.noise_scale,
            'use_fallback': self.use_old_weight_fallback,
            'total_classes': self.out_features,
            'feature_dim': self.in_features
        }


class NewClassAwareConfig:
    """新类感知分类器配置类"""
    
    def __init__(self, args):
        self.enable_new_class_aware = args.get('enable_new_class_aware_classifier', False)
        self.init_strategy = args.get('new_class_init_strategy', 'class_prototype')
        self.noise_scale = args.get('new_class_noise_scale', 0.1)
        self.use_fallback = args.get('new_class_use_fallback', True)
        
    def create_classifier(self, in_features, out_features, **kwargs):
        """创建新类感知分类器"""
        if not self.enable_new_class_aware:
            return CosineLinear(in_features, out_features, **kwargs)
            
        classifier = NewClassAwareClassifier(in_features, out_features, **kwargs)
        classifier.init_strategy = self.init_strategy
        classifier.noise_scale = self.noise_scale
        classifier.use_old_weight_fallback = self.use_fallback
        
        return classifier


def create_new_class_aware_classifier(in_features, out_features, args=None, **kwargs):
    """
    工厂函数：创建新类感知分类器
    
    Args:
        in_features: 输入特征维度
        out_features: 输出类别数
        args: 配置参数
        **kwargs: 其他参数
        
    Returns:
        分类器实例
    """
    if args and args.get('enable_new_class_aware_classifier', False):
        config = NewClassAwareConfig(args)
        return config.create_classifier(in_features, out_features, **kwargs)
    else:
        return CosineLinear(in_features, out_features, **kwargs)
