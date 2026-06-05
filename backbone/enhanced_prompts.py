"""
空间感知提示编码器 (Spatial-Aware Prompt Encoder)

基于temp.md中的设计理念实现
功能：在ASP的提示编码层面增强空间感知能力，深度集成到提示生成过程

更新：集成统一空间计算中心，避免重复计算（阶段4任务4.1）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 统一空间计算结果类
class SpatialComputeResult:
    """统一空间计算结果"""
    def __init__(self, pos_encoded=None, spatial_relations=None, raw_features=None,
                 relative_positions=None, adaptive_weights=None):
        self.pos_encoded = pos_encoded
        self.spatial_relations = spatial_relations
        self.raw_features = raw_features
        self.relative_positions = relative_positions
        self.adaptive_weights = adaptive_weights


class SpatialAwareModule:
    """空间感知模块基类"""
    def forward_with_shared_spatial(self, spatial_result):
        """使用共享空间计算结果的前向传播"""
        raise NotImplementedError("子类必须实现此方法")


class PositionalEncoding2D(nn.Module):
    """
    2D位置编码（优化版本，支持缓存）
    为提示tokens添加2D位置编码，使其感知在特征图中的绝对空间位置
    """

    def __init__(self, d_model, max_len=196):  # 14x14 = 196 patches for 224x224 image
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 预计算并缓存2D位置编码
        pe = self._compute_positional_encoding(d_model, max_len)
        self.register_buffer('pe', pe.unsqueeze(0))

        # 缓存不同序列长度的位置编码
        self._cached_pe = {}

    def _compute_positional_encoding(self, d_model, max_len):
        """预计算2D位置编码"""
        pe = torch.zeros(max_len, d_model)

        # 假设14x14的patch grid
        grid_size = int(math.sqrt(max_len))

        for pos in range(max_len):
            # 计算2D坐标
            y = pos // grid_size
            x = pos % grid_size

            # 为x和y坐标分别编码
            for i in range(0, d_model//4):
                # X坐标编码
                pe[pos, 4*i] = math.sin(x / (10000 ** (4*i / d_model)))
                pe[pos, 4*i + 1] = math.cos(x / (10000 ** (4*i / d_model)))
                # Y坐标编码
                pe[pos, 4*i + 2] = math.sin(y / (10000 ** (4*i / d_model)))
                pe[pos, 4*i + 3] = math.cos(y / (10000 ** (4*i / d_model)))

        return pe

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + 2D positional encoding
        """
        seq_len = x.size(1)
        device = x.device

        # 检查序列长度是否超出预计算范围
        if seq_len > self.max_len:
            # 动态计算更大的位置编码
            pe_extended = self._compute_positional_encoding(self.d_model, seq_len)
            pe_extended = pe_extended.unsqueeze(0).to(device)
            return x + pe_extended

        # 检查缓存
        cache_key = (seq_len, device)
        if cache_key not in self._cached_pe:
            # 缓存当前序列长度的位置编码
            self._cached_pe[cache_key] = self.pe[:, :seq_len, :].to(device)

        return x + self._cached_pe[cache_key]


class LearnablePositionalEncoding2D(nn.Module):
    """
    可学习的2D位置编码（优化版本，支持缓存）
    使用可训练的参数来学习最优的位置编码
    """

    def __init__(self, d_model, max_len=196):
        super(LearnablePositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 创建可学习的位置编码参数
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # 缓存不同序列长度的位置编码
        self._cached_pe = {}

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + learnable positional encoding
        """
        seq_len = x.size(1)
        device = x.device

        # 检查序列长度是否超出预计算范围
        if seq_len > self.max_len:
            # 对于超出范围的序列，只使用前max_len个位置的编码，其余位置为0
            pe_extended = torch.zeros(1, seq_len, self.d_model, device=device)
            pe_extended[:, :self.max_len, :] = self.pe.to(device)
            return x + pe_extended

        # 检查缓存
        cache_key = (seq_len, device)
        if cache_key not in self._cached_pe:
            # 缓存当前序列长度的位置编码
            self._cached_pe[cache_key] = self.pe[:, :seq_len, :].to(device)

        return x + self._cached_pe[cache_key]


class AdaptivePositionalEncoding2D(nn.Module):
    """
    自适应2D位置编码（基于temp.md设计）
    结合固定编码和可学习缩放因子，支持高效缓存
    """

    def __init__(self, d_model, max_len=196):
        super(AdaptivePositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 可学习的缩放因子
        self.learnable_scale = nn.Parameter(torch.ones(1))

        # 预计算固定的位置编码
        pe = self._compute_positional_encoding(d_model, max_len)
        self.register_buffer('pe', pe.unsqueeze(0))

        # 缓存不同序列长度和设备的位置编码
        self._cached_pe = {}

    def _compute_positional_encoding(self, d_model, max_len):
        """预计算2D位置编码"""
        pe = torch.zeros(max_len, d_model)
        grid_size = int(math.sqrt(max_len))

        for pos in range(max_len):
            y = pos // grid_size
            x = pos % grid_size

            for i in range(0, d_model//4):
                # X坐标编码
                pe[pos, 4*i] = math.sin(x / (10000 ** (4*i / d_model)))
                pe[pos, 4*i + 1] = math.cos(x / (10000 ** (4*i / d_model)))
                # Y坐标编码
                pe[pos, 4*i + 2] = math.sin(y / (10000 ** (4*i / d_model)))
                pe[pos, 4*i + 3] = math.cos(y / (10000 ** (4*i / d_model)))

        return pe

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + adaptive positional encoding
        """
        seq_len = x.size(1)
        device = x.device

        # 检查序列长度是否超出预计算范围
        if seq_len > self.max_len:
            # 动态计算更大的位置编码
            pe_extended = self._compute_positional_encoding(self.d_model, seq_len)
            pe_extended = pe_extended.unsqueeze(0).to(device)
            return x + self.learnable_scale * pe_extended

        # 检查缓存
        cache_key = (seq_len, device)
        if cache_key not in self._cached_pe:
            # 缓存当前序列长度的位置编码
            self._cached_pe[cache_key] = self.pe[:, :seq_len, :].to(device)

        return x + self.learnable_scale * self._cached_pe[cache_key]


class RelativePositionAttention(nn.Module):
    """
    相对位置注意力机制
    允许提示之间、以及提示与图像patch之间动态地根据其相对位置调整注意力权重
    """
    
    def __init__(self, d_model, num_heads=8, max_relative_position=14):
        super(RelativePositionAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_relative_position = max_relative_position

        # 查询、键、值投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 相对位置嵌入
        self.relative_position_k = nn.Parameter(
            torch.randn(2 * max_relative_position - 1, self.head_dim)
        )
        self.relative_position_v = nn.Parameter(
            torch.randn(2 * max_relative_position - 1, self.head_dim)
        )

        self.dropout = nn.Dropout(0.1)

        # 缓存相对位置索引
        self._cached_relative_positions = {}
        
    def _get_relative_positions(self, seq_len):
        """计算相对位置索引（支持缓存）"""
        # 检查缓存
        if seq_len in self._cached_relative_positions:
            return self._cached_relative_positions[seq_len]

        grid_size = int(math.sqrt(seq_len))
        positions = []

        for i in range(seq_len):
            y1, x1 = i // grid_size, i % grid_size
            row = []
            for j in range(seq_len):
                y2, x2 = j // grid_size, j % grid_size
                # 计算相对位置（曼哈顿距离）
                rel_pos = abs(y1 - y2) + abs(x1 - x2)
                rel_pos = min(rel_pos, self.max_relative_position - 1)
                row.append(rel_pos)
            positions.append(row)

        # 缓存结果
        result = torch.tensor(positions, dtype=torch.long)
        self._cached_relative_positions[seq_len] = result
        return result
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            相对位置注意力增强的特征
        """
        batch_size, seq_len, d_model = x.shape
        
        # 投影到Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加相对位置偏置
        rel_pos_indices = self._get_relative_positions(seq_len).to(x.device)
        rel_pos_k = self.relative_position_k[rel_pos_indices]  # [seq_len, seq_len, head_dim]
        
        # 计算相对位置注意力
        rel_scores = torch.einsum('bhid,ijd->bhij', q, rel_pos_k)
        scores = scores + rel_scores
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, v)
        
        # 添加相对位置到值
        rel_pos_v = self.relative_position_v[rel_pos_indices]  # [seq_len, seq_len, head_dim]
        rel_attn_output = torch.einsum('bhij,ijd->bhid', attn_weights, rel_pos_v)
        attn_output = attn_output + rel_attn_output
        
        # 重塑并投影输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.out_proj(attn_output)
        
        return output


class SpatialAwarePromptEncoder(nn.Module, SpatialAwareModule):
    """
    空间感知提示编码器
    添加2D位置编码和相对位置注意力，深度集成到ASP的提示生成过程

    更新：支持统一空间计算，避免重复计算（阶段4任务4.1）
    """

    def __init__(self, args, depth, prompt_length, prompt_features=768):
        super(SpatialAwarePromptEncoder, self).__init__()

        self.args = args
        self.depth = depth
        self.prompt_length = prompt_length
        self.prompt_features = prompt_features

        # 空间感知配置
        self.spatial_aware_prompts = args.get('spatial_aware_prompts', False) if args else False

        # 统一空间计算支持
        self.use_unified_spatial = args.get('use_unified_spatial', False) if args else False

        # 多尺度光谱特征融合配置（方案一）
        self.enable_multiscale_spectral_fusion = args.get('enable_multiscale_spectral_fusion', False) if args else False
        if self.enable_multiscale_spectral_fusion:
            from .multiscale_spectral_fusion import MultiScaleSpectralFusion
            self.multiscale_spectral_fusion = MultiScaleSpectralFusion(
                feature_dim=prompt_features,
                args=args
            )
            print(f"✅ 多尺度光谱特征融合已启用: 参数数量={self.multiscale_spectral_fusion.get_parameter_count()}")
        else:
            self.multiscale_spectral_fusion = None

        # 轻量级特征调制配置（通过JSON控制）
        self.enable_feature_modulation = args.get('enable_feature_modulation', False) if args else False
        # 自适应调制v2已移除
        self.enable_feature_modulation_v1pro = args.get('enable_feature_modulation_v1pro', False) if args else False

        if self.enable_feature_modulation:
            from .lightweight_feature_modulation import LightweightFeatureModulation
            self.feature_modulation = LightweightFeatureModulation(
                feature_dim=prompt_features,
                modulation_strength=args.get('modulation_strength', 0.05)
            )
            print(f"✅ 轻量级特征调制v1已启用: strength={args.get('modulation_strength', 0.05)}, 参数数量={self.feature_modulation.get_parameter_count()}")
        # 自适应调制v2已移除（效果不佳，-0.93%性能下降）
        else:
            self.feature_modulation = None
        
        if self.spatial_aware_prompts:
            # 2D位置编码
            self.spatial_position_encoding = args.get('spatial_position_encoding', True) if args else True
            if self.spatial_position_encoding:
                # 根据position_encoding_type创建不同的位置编码
                position_encoding_type = args.get('position_encoding_type', 'sinusoidal_2d') if args else 'sinusoidal_2d'

                if position_encoding_type == 'standard_1d':
                    # A1.1: 标准1D位置编码（实际上不使用空间感知）
                    self.pos_embedding = None
                elif position_encoding_type == 'sinusoidal_2d':
                    # A1.2: 2D正弦位置编码（优化版本）
                    self.pos_embedding = PositionalEncoding2D(prompt_features)
                elif position_encoding_type == 'learnable_2d':
                    # A1.3: 2D可学习位置编码（优化版本）
                    self.pos_embedding = LearnablePositionalEncoding2D(prompt_features)
                elif position_encoding_type == 'adaptive_2d':
                    # A1.4: 自适应2D位置编码（新增，基于temp.md设计）
                    self.pos_embedding = AdaptivePositionalEncoding2D(prompt_features)
                else:
                    print(f"Warning: Unknown position_encoding_type '{position_encoding_type}', using 'adaptive_2d'")
                    self.pos_embedding = AdaptivePositionalEncoding2D(prompt_features)
            
            # 相对位置注意力
            self.relative_position_attention = args.get('relative_position_attention', True) if args else True
            if self.relative_position_attention:
                self.rel_pos_attention = RelativePositionAttention(prompt_features, num_heads=8)
            
            # 空间关系建模
            self.spatial_relation_weight = args.get('spatial_relation_weight', 0.1) if args else 0.1

            # 根据spatial_mlp_type创建不同的MLP结构
            spatial_mlp_type = args.get('spatial_mlp_type', 'compress_expand') if args else 'compress_expand'

            if spatial_mlp_type == 'none':
                # A3.1: 无空间关系MLP
                self.spatial_relation_mlp = None
            elif spatial_mlp_type == 'linear':
                # A3.3: 线性变换结构
                self.spatial_relation_mlp = nn.Linear(prompt_features, prompt_features)
            elif spatial_mlp_type == 'compress_expand':
                # A3.2: 压缩-扩展结构 (默认)
                self.spatial_relation_mlp = nn.Sequential(
                    nn.Linear(prompt_features, prompt_features // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(prompt_features // 2, prompt_features)
                )
            else:
                # 未知类型，使用默认的压缩-扩展结构
                print(f"Warning: Unknown spatial_mlp_type '{spatial_mlp_type}', using 'compress_expand'")
                self.spatial_relation_mlp = nn.Sequential(
                    nn.Linear(prompt_features, prompt_features // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(prompt_features // 2, prompt_features)
                )
        else:
            # 当spatial_aware_prompts=False时，初始化默认属性
            self.spatial_position_encoding = False
            self.relative_position_attention = False
            self.pos_embedding = None
            self.rel_pos_attention = None
            self.spatial_relation_mlp = None
            self.spatial_relation_weight = 0.0

        # 基础提示编码器
        self.base_encoder = nn.Sequential(
            nn.Linear(prompt_features * 2, prompt_features),
            nn.LayerNorm(prompt_features),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_features, prompt_length * prompt_features)
        )

        # 协同优化：空间上下文传递机制（阶段1）
        self.enable_spatial_context_transfer = args.get('enable_spatial_context_transfer', False) if args else False
        if self.enable_spatial_context_transfer:
            self.spatial_context_dim = args.get('spatial_context_dim', 128) if args else 128
            # 空间上下文投影层（极少参数）
            self.spatial_context_projection = nn.Linear(prompt_features, self.spatial_context_dim)
            print(f"✅ 空间上下文传递已启用: context_dim={self.spatial_context_dim}, 参数数量={self.spatial_context_dim * prompt_features + self.spatial_context_dim}")
        else:
            self.spatial_context_projection = None
            self.spatial_context_dim = 0
    
    def forward(self, tip_features, image_features, spatial_coords=None):
        """
        前向传播
        
        Args:
            tip_features: TIP特征 [batch_size, feature_dim]
            image_features: 图像特征 [batch_size, feature_dim]
            spatial_coords: 空间坐标（可选）
            
        Returns:
            spatial_prompts: 空间感知的提示 [batch_size, depth, prompt_length, prompt_features]
        """
        batch_size = tip_features.size(0)
        
        # 基础提示生成
        combined_features = torch.cat([tip_features, image_features], dim=1)
        base_prompts = self.base_encoder(combined_features)
        base_prompts = base_prompts.view(batch_size, self.prompt_length, self.prompt_features)
        
        if self.spatial_aware_prompts:
            # 添加2D位置编码
            if self.spatial_position_encoding and self.pos_embedding is not None:
                pos_encoded_prompts = self.pos_embedding(base_prompts)
            else:
                pos_encoded_prompts = base_prompts
            
            # 应用相对位置注意力
            if self.relative_position_attention:
                spatial_enhanced_prompts = self.rel_pos_attention(pos_encoded_prompts)
            else:
                spatial_enhanced_prompts = pos_encoded_prompts
            
            # 空间关系建模
            if self.spatial_relation_mlp is not None:
                # 有MLP结构时进行空间关系建模
                spatial_relations = self.spatial_relation_mlp(spatial_enhanced_prompts)
                final_prompts = spatial_enhanced_prompts + self.spatial_relation_weight * spatial_relations
            else:
                # 无MLP结构时直接使用空间增强的提示
                final_prompts = spatial_enhanced_prompts
            
        else:
            final_prompts = base_prompts
        
        # 扩展到所有深度
        spatial_prompts = final_prompts.unsqueeze(1).expand(-1, self.depth, -1, -1)

        # 应用多尺度光谱特征融合（在轻量级调制之前）
        if self.enable_multiscale_spectral_fusion and hasattr(self, 'multiscale_spectral_fusion') and self.multiscale_spectral_fusion is not None:
            spatial_prompts = self.multiscale_spectral_fusion(spatial_prompts, training=self.training)

        # 应用轻量级特征调制（如果启用v1或v1pro）
        if (self.enable_feature_modulation or self.enable_feature_modulation_v1pro) and hasattr(self, 'feature_modulation') and self.feature_modulation is not None:
            spatial_prompts = self.feature_modulation(spatial_prompts, training=self.training)

        return spatial_prompts

    def forward_without_feature_modulation(self, tip_features, image_features, spatial_coords=None):
        """Generate SAPM prompts before lightweight/spatial-context modulation."""
        old_feature_modulation = self.feature_modulation
        old_enable_feature_modulation = self.enable_feature_modulation
        old_enable_feature_modulation_v1pro = self.enable_feature_modulation_v1pro
        try:
            self.feature_modulation = None
            self.enable_feature_modulation = False
            self.enable_feature_modulation_v1pro = False
            return self.forward(tip_features, image_features, spatial_coords)
        finally:
            self.feature_modulation = old_feature_modulation
            self.enable_feature_modulation = old_enable_feature_modulation
            self.enable_feature_modulation_v1pro = old_enable_feature_modulation_v1pro

    def _apply_spatial_context_modulation(self, spatial_prompts, spatial_context):
        """应用空间上下文感知的特征调制（协同优化阶段1）"""
        if (self.enable_feature_modulation or self.enable_feature_modulation_v1pro) and hasattr(self, 'feature_modulation') and self.feature_modulation is not None:
            # 检查特征调制模块是否支持空间上下文
            if hasattr(self.feature_modulation, 'forward_with_spatial_context'):
                return self.feature_modulation.forward_with_spatial_context(spatial_prompts, spatial_context, training=self.training)
            else:
                # 回退到原始调制
                return self.feature_modulation(spatial_prompts, training=self.training)
        else:
            return spatial_prompts

    def generate_spatial_context(self, tip_features, image_features):
        """
        生成空间上下文向量（协同优化阶段1）

        Args:
            tip_features: TIP特征 [batch_size, feature_dim]
            image_features: 图像特征 [batch_size, feature_dim]

        Returns:
            spatial_context: 空间上下文向量 [batch_size, spatial_context_dim] 或 None
        """
        if not self.enable_spatial_context_transfer or self.spatial_context_projection is None:
            return None

        # 融合TIP和图像特征
        combined_features = torch.cat([tip_features, image_features], dim=1)

        # 通过基础编码器获取中间特征
        intermediate_features = self.base_encoder[:-1](combined_features)  # 除了最后一层

        # 如果启用了空间感知，使用空间增强的特征
        if self.spatial_aware_prompts:
            # 简化的空间增强（用于上下文生成）
            if self.spatial_position_encoding and self.pos_embedding is not None:
                # 重塑为适合位置编码的形状
                reshaped_features = intermediate_features.unsqueeze(1).expand(-1, self.prompt_length, -1)
                pos_enhanced = self.pos_embedding(reshaped_features)
                # 取平均作为全局空间特征
                spatial_features = pos_enhanced.mean(dim=1)
            else:
                spatial_features = intermediate_features
        else:
            spatial_features = intermediate_features

        # 投影到空间上下文维度
        spatial_context = self.spatial_context_projection(spatial_features)

        return spatial_context

    def forward_with_context(self, tip_features, image_features, spatial_coords=None):
        """
        带空间上下文输出的前向传播（协同优化阶段1）

        Args:
            tip_features: TIP特征 [batch_size, feature_dim]
            image_features: 图像特征 [batch_size, feature_dim]
            spatial_coords: 空间坐标（可选）

        Returns:
            tuple: (spatial_prompts, spatial_context)
                - spatial_prompts: 空间感知的提示 [batch_size, depth, prompt_length, prompt_features]
                - spatial_context: 空间上下文向量 [batch_size, spatial_context_dim] 或 None
        """
        batch_size = tip_features.size(0)

        # 生成空间上下文（在特征调制之前）
        spatial_context = self.generate_spatial_context(tip_features, image_features)

        # 基础提示生成（复制forward方法的逻辑，但使用空间上下文感知调制）
        combined_features = torch.cat([tip_features, image_features], dim=1)
        base_prompts = self.base_encoder(combined_features)
        base_prompts = base_prompts.view(batch_size, self.prompt_length, self.prompt_features)

        if self.spatial_aware_prompts:
            # 添加2D位置编码
            if self.spatial_position_encoding and self.pos_embedding is not None:
                pos_encoded_prompts = self.pos_embedding(base_prompts)
            else:
                pos_encoded_prompts = base_prompts

            # 应用相对位置注意力
            if self.relative_position_attention:
                spatial_enhanced_prompts = self.rel_pos_attention(pos_encoded_prompts)
            else:
                spatial_enhanced_prompts = pos_encoded_prompts

            # 空间关系建模
            if self.spatial_relation_mlp is not None:
                spatial_relations = self.spatial_relation_mlp(spatial_enhanced_prompts)
                final_prompts = spatial_enhanced_prompts + self.spatial_relation_weight * spatial_relations
            else:
                final_prompts = spatial_enhanced_prompts
        else:
            final_prompts = base_prompts

        # 扩展到所有深度
        spatial_prompts = final_prompts.unsqueeze(1).expand(-1, self.depth, -1, -1)

        # 应用多尺度光谱特征融合（在轻量级调制之前）
        if self.enable_multiscale_spectral_fusion and hasattr(self, 'multiscale_spectral_fusion') and self.multiscale_spectral_fusion is not None:
            spatial_prompts = self.multiscale_spectral_fusion(spatial_prompts, training=self.training)

        # 协同优化：应用空间上下文感知的特征调制
        spatial_prompts = self._apply_spatial_context_modulation(spatial_prompts, spatial_context)

        return spatial_prompts, spatial_context

    def forward_with_shared_spatial(self, spatial_result: SpatialComputeResult,
                                   tip_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        使用统一空间计算结果的前向传播

        Args:
            spatial_result: 统一空间计算结果
            tip_features: TIP特征 [batch_size, feature_dim]
            image_features: 图像特征 [batch_size, feature_dim]

        Returns:
            spatial_prompts: 空间感知的提示 [batch_size, depth, prompt_length, prompt_features]
        """
        batch_size = tip_features.size(0)

        # 基础提示生成
        combined_features = torch.cat([tip_features, image_features], dim=1)
        base_prompts = self.base_encoder(combined_features)
        base_prompts = base_prompts.view(batch_size, self.prompt_length, self.prompt_features)

        if self.spatial_aware_prompts and self.use_unified_spatial:
            # 使用统一空间计算结果
            # 将基础提示与空间编码结果融合
            if spatial_result.pos_encoded is not None:
                # 调整维度以匹配提示长度
                if spatial_result.pos_encoded.size(1) != self.prompt_length:
                    # 如果维度不匹配，使用自适应池化调整
                    spatial_features = F.adaptive_avg_pool1d(
                        spatial_result.pos_encoded.transpose(1, 2),
                        self.prompt_length
                    ).transpose(1, 2)
                else:
                    spatial_features = spatial_result.pos_encoded

                # 融合空间特征
                pos_encoded_prompts = base_prompts + spatial_features
            else:
                pos_encoded_prompts = base_prompts

            # 使用空间关系信息
            if spatial_result.spatial_relations is not None:
                # 调整维度以匹配提示长度
                if spatial_result.spatial_relations.size(1) != self.prompt_length:
                    spatial_relations = F.adaptive_avg_pool1d(
                        spatial_result.spatial_relations.transpose(1, 2),
                        self.prompt_length
                    ).transpose(1, 2)
                else:
                    spatial_relations = spatial_result.spatial_relations

                # 应用空间关系权重（支持自适应权重）
                if spatial_result.adaptive_weights and 'relation_weight' in spatial_result.adaptive_weights:
                    # 使用自适应权重
                    adaptive_relation_weight = spatial_result.adaptive_weights['relation_weight'].unsqueeze(1)  # [B, 1, 1]
                    final_prompts = pos_encoded_prompts + adaptive_relation_weight * spatial_relations
                else:
                    # 使用固定权重
                    final_prompts = pos_encoded_prompts + self.spatial_relation_weight * spatial_relations
            else:
                final_prompts = pos_encoded_prompts

        elif self.spatial_aware_prompts:
            # 回退到原始的空间感知计算
            return self.forward(tip_features, image_features)
        else:
            final_prompts = base_prompts

        # 扩展到所有深度
        spatial_prompts = final_prompts.unsqueeze(1).expand(-1, self.depth, -1, -1)

        # 应用多尺度光谱特征融合（在轻量级调制之前）
        if self.enable_multiscale_spectral_fusion and hasattr(self, 'multiscale_spectral_fusion') and self.multiscale_spectral_fusion is not None:
            spatial_prompts = self.multiscale_spectral_fusion(spatial_prompts, training=self.training)

        # 应用轻量级特征调制（如果启用v1或v1pro）
        if (self.enable_feature_modulation or self.enable_feature_modulation_v1pro) and hasattr(self, 'feature_modulation') and self.feature_modulation is not None:
            spatial_prompts = self.feature_modulation(spatial_prompts, training=self.training)

        return spatial_prompts

    def set_task_id(self, task_id):
        """设置当前任务ID（用于自适应调制）"""
        if hasattr(self, 'feature_modulation') and self.feature_modulation is not None:
            if hasattr(self.feature_modulation, 'set_task_id'):
                self.feature_modulation.set_task_id(task_id)
                print(f"🔧 自适应调制任务ID已设置: {task_id}")

    def get_modulation_stats(self):
        """获取调制统计信息"""
        if hasattr(self, 'feature_modulation') and self.feature_modulation is not None:
            if hasattr(self.feature_modulation, 'get_modulation_stats'):
                return self.feature_modulation.get_modulation_stats()
            elif hasattr(self.feature_modulation, 'get_parameter_count'):
                return {'parameter_count': self.feature_modulation.get_parameter_count()}
        return {'parameter_count': 0}


class SpatialPromptConfig:
    """空间提示配置类"""
    
    def __init__(self):
        # 空间感知提示配置
        self.spatial_aware_prompts = True
        self.spatial_position_encoding = True
        self.relative_position_attention = True
        self.spatial_relation_weight = 0.1
        
        # 提示基础配置
        self.prompt_length = 6
        self.prompt_features = 768
        self.depth = 12  # ViT-B/16的深度
