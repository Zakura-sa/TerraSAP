import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 设置绘图风格 (全局设置)
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',    # 对应原图的无衬线字体 (如Arial/Helvetica)
    'axes.labelsize': 16,           # 坐标轴标签字号
    'xtick.labelsize': 14,          # X轴刻度字号
    'ytick.labelsize': 14,          # Y轴刻度字号
    'axes.linewidth': 1.2,          # 坐标轴线宽
    'hatch.linewidth': 0.5,
    'legend.fontsize': 13,
    'legend.frameon': False,        # 图例无边框
    'figure.autolayout': False,     # 手动控制布局
})

# 颜色定义 (提取自原图风格: Tableau/Seaborn Muted 风格)
colors = {
    'NWPU':  '#4E79A7', # 蓝色
    'MSTAR': '#59A14F', # 绿色
    'UCM':   '#F28E2B'  # 橙色
}

# ==========================================
# 2. Embedded values for the parameter-sensitivity figure
# ==========================================
# 结构: '文件名': { 'x_label': 'X轴标题', 'ticks': [刻度], 'data': {'NWPU': [], 'MSTAR': [], 'UCM': []} }

configs = {
    'sensitivity_modulation_strength': {
        'x_title': 'Modulation Strength',
        'ticks': ['0', '0.1', '0.2'],
        'data': {
            'NWPU':  [83.91, 84.52, 84.14],
            'MSTAR': [73.68, 76.70, 74.23],
            'UCM':   [83.85, 84.41, 84.02]
        }
    },
    'sensitivity_spatial_relation_weight': {
        'x_title': 'Spatial Relation Weight',
        'ticks': ['0.2', '0.35', '0.5'],
        'data': {
            'NWPU':  [83.79, 84.52, 84.11],
            'MSTAR': [71.28, 76.70, 71.37],
            'UCM':   [83.90, 84.41, 84.15]
        }
    },
    'sensitivity_avg_alpha': {
        'x_title': 'Average Alpha',
        'ticks': ['0.3', '0.6', '0.75', '0.85', '0.95'],
        'data': {
            'NWPU':  [83.95, 84.11, 83.99, 83.83, 84.52],
            # 注意: 0.85处MSTAR为61.54，低于Y轴下限65，原图中此处不显示柱子，符合预期
            'MSTAR': [72.02, 71.06, 76.70, 61.54, 71.39], 
            'UCM':   [83.50, 83.90, 84.41, 83.80, 83.95]
        }
    },
    'sensitivity_dual_ema_fusion_weight': {
        'x_title': 'Dual EMA Fusion Weight',
        'ticks': ['0.35', '0.45', '0.55', '0.65'],
        'data': {
            'NWPU':  [84.52, 83.92, 84.10, 83.76],
            'MSTAR': [69.63, 76.70, 67.76, 70.41],
            'UCM':   [84.10, 84.41, 84.00, 83.85]
        }
    },
    'sensitivity_EMA_beta': {
        'x_title': 'EMA Beta',
        'ticks': ['0.98', '0.99', '0.995', '0.997'],
        'data': {
            'NWPU':  [84.10, 84.52, 84.03, 83.65],
            'MSTAR': [76.70, 71.19, 67.95, 71.18],
            'UCM':   [83.95, 84.41, 84.00, 83.70]
        }
    }
}

# ==========================================
# 3. 绘图函数
# ==========================================
def draw_bar_chart(filename, config):
    # 数据提取
    labels = config['ticks']
    nwpu_means = config['data']['NWPU']
    mstar_means = config['data']['MSTAR']
    ucm_means = config['data']['UCM']

    x = np.arange(len(labels))  # 标签位置
    width = 0.20  # 柱状图宽度，原图较窄

    # 创建画布，长宽比模仿原图 (宽矮型)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    # 绘制柱子 (注意顺序：NWPU, MSTAR, UCM)
    # zorder=3 确保柱子在网格线前方
    rects1 = ax.bar(x - width, nwpu_means, width, label='NWPU', 
                    color=colors['NWPU'], edgecolor='black', linewidth=0.6, zorder=3)
    rects2 = ax.bar(x, mstar_means, width, label='MSTAR', 
                    color=colors['MSTAR'], edgecolor='black', linewidth=0.6, zorder=3)
    rects3 = ax.bar(x + width, ucm_means, width, label='UCM', 
                    color=colors['UCM'], edgecolor='black', linewidth=0.6, zorder=3)

    # Y轴设置
    ax.set_ylabel('Avg Top-1 (%)')
    ax.set_ylim(65, 87) # 依据原图，Y轴范围固定在 65-87 之间
    
    # X轴设置
    ax.set_xlabel(config['x_title'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # 网格线设置 (水平，虚线，灰色，位于底层)
    ax.yaxis.grid(True, linestyle=':', alpha=0.6, color='lightgray', zorder=0, linewidth=1.5)
    ax.set_axisbelow(True) # 确保网格线在柱子后面

    # 图例设置 (位于顶部，水平排列，无边框)
    # bbox_to_anchor 用于微调图例位置，使其像原图一样悬浮在图表上方
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), 
              ncol=3, columnspacing=3.5, handletextpad=0.5)

    # 调整布局，防止标签被截断
    plt.tight_layout()
    
    # 调整顶部边距以容纳图例
    plt.subplots_adjust(top=0.88) 

    # 保存文件
    save_name = f"{filename}.pdf"
    plt.savefig(save_name, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Generated: {save_name}")
    plt.close()

# ==========================================
# 4. 批量生成
# ==========================================
if __name__ == "__main__":
    for fname, cfg in configs.items():
        draw_bar_chart(fname, cfg)