import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 全局样式设置 (IEEE 风格)
# ==========================================
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif', # 如果系统有 Times New Roman，建议改为 'Times New Roman'
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 5),
    'lines.linewidth': 2.5,
    'grid.alpha': 0.6
})

# ==========================================
# 2. 数据与样式定义
# ==========================================

# 颜色代码 (参考原图近似色)
colors = {
    'red_dark': '#d62728',   # TerraSAP
    'red_light': '#d65f5f',  # RSSC (Salmon)
    'orange': '#ff7f0e',     # ASP
    'pink': '#ffb6c1',       # LIMIT
    'purple': '#9467bd',     # BiDist
    'purple_light': '#c5b0d5', # SAVC
    'blue_light': '#9edae5', # CPL (Light Blue)
    'cyan': '#17becf'        # C-FSCIL
}

# ----------------------------
# NWPU-RESISC45 数据
# ----------------------------
nwpu_sessions = ['1', '2', '3', '4', '5']
nwpu_data = {
    'TerraSAP': {'y': [93.87, 86.80, 82.39, 81.12, 78.43], 'c': colors['red_dark'],  'm': '*', 'mfc': colors['red_dark'], 'lw': 3.5, 'ms': 11}, # 实心
    'ASP':      {'y': [94.55, 86.79, 80.46, 77.03, 76.16], 'c': colors['orange'],    'm': 'X', 'mfc': 'white',            'lw': 2.5, 'ms': 9},  # 空心X
    'RSSC':     {'y': [94.80, 85.90, 82.80, 71.80, 71.80], 'c': colors['red_light'], 'm': 'o', 'mfc': 'white',            'lw': 2.5, 'ms': 9},  # 空心圆
    'LIMIT':    {'y': [94.65, 85.90, 79.50, 71.80, 70.60], 'c': colors['pink'],      'm': 's', 'mfc': 'white',            'lw': 2.5, 'ms': 9},  # 空心方
    'BiDist':   {'y': [94.40, 82.50, 78.40, 76.30, 69.10], 'c': colors['purple'],    'm': '^', 'mfc': 'white',            'lw': 2.5, 'ms': 9},  # 空心三角
}
# 原图图例顺序: TerraSAP -> ASP -> RSSC -> LIMIT -> BiDist
nwpu_legend_order = ['TerraSAP', 'ASP', 'RSSC', 'LIMIT', 'BiDist']

# ----------------------------
# UCM 数据
# ----------------------------
ucm_sessions = ['1', '2', '3', '4']
ucm_data = {
    'RSSC':     {'y': [98.30, 87.40, 80.30, 77.00], 'c': colors['red_light'],  'm': 'o', 'mfc': 'white',            'lw': 2.5, 'ms': 9},
    'TerraSAP': {'y': [98.33, 85.10, 79.62, 74.61], 'c': colors['red_dark'],   'm': '*', 'mfc': colors['red_dark'], 'lw': 3.5, 'ms': 11},
    'LIMIT':    {'y': [97.90, 82.70, 78.20, 71.40], 'c': colors['pink'],       'm': 's', 'mfc': 'white',            'lw': 2.5, 'ms': 9},
    'BiDist':   {'y': [97.80, 81.50, 78.80, 70.50], 'c': colors['purple'],     'm': '^', 'mfc': 'white',            'lw': 2.5, 'ms': 9},
    'SAVC':     {'y': [97.70, 82.00, 75.10, 72.72], 'c': colors['purple_light'],'m': 'P', 'mfc': 'white',            'lw': 2.5, 'ms': 9}, # P for Plus(filled)
}
# 原图图例顺序: RSSC -> TerraSAP -> LIMIT -> BiDist -> SAVC
ucm_legend_order = ['RSSC', 'TerraSAP', 'LIMIT', 'BiDist', 'SAVC']

# ----------------------------
# MSTAR 数据
# ----------------------------
mstar_sessions = ['1', '2', '3', '4', '5', '6', '7']
mstar_data = {
    'CPL':      {'y': [99.89, 90.60, 76.96, 72.92, 70.39, 66.71, 60.36], 'c': colors['blue_light'], 'm': 'v', 'mfc': 'white',            'lw': 2.5, 'ms': 9},
    'TerraSAP': {'y': [94.79, 86.82, 80.70, 76.32, 70.45, 66.08, 61.71], 'c': colors['red_dark'],   'm': '*', 'mfc': colors['red_dark'], 'lw': 3.5, 'ms': 11},
    'RSSC':     {'y': [95.20, 88.20, 82.50, 72.00, 66.40, 63.20, 57.78], 'c': colors['red_light'],  'm': 'o', 'mfc': 'white',            'lw': 2.5, 'ms': 9},
    'BiDist':   {'y': [94.10, 88.90, 77.00, 70.30, 64.00, 61.10, 58.68], 'c': colors['purple'],     'm': '^', 'mfc': 'white',            'lw': 2.5, 'ms': 9},
    'C-FSCIL':  {'y': [92.90, 87.50, 74.40, 68.50, 62.90, 57.20, 52.69], 'c': colors['cyan'],       'm': '2', 'mfc': 'white',            'lw': 2.5, 'ms': 10}, # Tri-up marker
}
# 原图图例顺序: CPL -> TerraSAP -> RSSC -> BiDist -> C-FSCIL
mstar_legend_order = ['CPL', 'TerraSAP', 'RSSC', 'BiDist', 'C-FSCIL']


# ==========================================
# 3. 核心绘图函数
# ==========================================
def plot_sota_reproduced(sessions, data_dict, legend_order, title, filename, y_lim=None):
    plt.figure(figsize=(8, 5))
    
    # 临时存储 handles 用于自定义图例顺序
    handles_map = {}

    # 绘制策略：为了防止关键线条(TerraSAP)被遮挡，我们先把非TerraSAP画完，最后画TerraSAP
    # 但这不会影响图例顺序，图例顺序由 legend_order 决定
    plot_sequence = [k for k in data_dict.keys() if k != 'TerraSAP'] + ['TerraSAP']
    
    for label in plot_sequence:
        if label not in data_dict: continue # 安全检查
        d = data_dict[label]
        
        # 绘图
        line, = plt.plot(sessions, d['y'], 
                 label=label, 
                 color=d['c'], 
                 marker=d['m'], 
                 linewidth=d['lw'], 
                 markersize=d['ms'],
                 markerfacecolor=d['mfc'], 
                 markeredgewidth=2) # 边框加粗，确保空心效果明显
        
        handles_map[label] = line

    # 坐标轴设置
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Session', fontweight='normal') # 论文中X轴标签通常不是特别粗
    plt.ylabel('Accuracy (%)', fontweight='normal')
    plt.grid(True, linestyle='-', alpha=0.6)
    
    if y_lim:
        plt.ylim(y_lim)

    # --- 关键修改：强制图例顺序 ---
    # 根据传入的 legend_order 列表，从 handles_map 中提取对应的句柄
    sorted_handles = [handles_map[label] for label in legend_order if label in handles_map]
    sorted_labels = [label for label in legend_order if label in handles_map]

    # 设置图例：右上角，带边框，不透明背景
    plt.legend(sorted_handles, sorted_labels, 
               loc='upper right', 
               frameon=True, 
               framealpha=1,      # 不透明背景
               edgecolor='#cccccc', # 浅灰边框
               fancybox=True)     # 圆角边框

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Generated: {filename}")

# ==========================================
# 4. 生成图片
# ==========================================

# 1. NWPU-RESISC45
plot_sota_reproduced(
    nwpu_sessions, nwpu_data, nwpu_legend_order,
    'NWPU-RESISC45 Top-1 Accuracy', 
    'sota_nwpu_final.pdf'
)

# 2. UCM
plot_sota_reproduced(
    ucm_sessions, ucm_data, ucm_legend_order,
    'UCM Top-1 Accuracy', 
    'sota_ucm_final.pdf'
)

# 3. MSTAR
plot_sota_reproduced(
    mstar_sessions, mstar_data, mstar_legend_order,
    'MSTAR Top-1 Accuracy', 
    'sota_mstar_final.pdf'
)