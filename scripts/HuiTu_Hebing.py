import matplotlib.pyplot as plt
import numpy as np
from pdf2image import convert_from_path
import os

# ================= 配置区域 =================
# 输出分辨率设置
TARGET_DPI = 600

FIGURES_CONFIG = {
    "fig2": {
        "files": [
            "NWPU_RESISC45_samples.pdf", 
            "UCMerced_samples.pdf", 
            "MSTAR_samples.pdf"
        ],
        "layout": (3, 1), # 3行1列
        "labels": [
            "(a) NWPU-RESISC45 dataset samples", 
            "(b) UCM dataset samples", 
            "(c) MSTAR dataset samples"
        ],
        "figsize": (10, 18) 
    },
    "fig3": {
        "files": [
            "sota_nwpu_final.pdf", 
            "sota_ucm_final.pdf", 
            "sota_mstar_final.pdf"
        ],
        "layout": (1, 3), # 1行3列
        "labels": [
            "(a) NWPU-RESISC45", 
            "(b) UCM", 
            "(c) MSTAR"
        ],
        "figsize": (20, 6)
    },
    "fig6": {
        "files": [
            "sensitivity_avg_alpha.pdf",
            "sensitivity_dual_ema_fusion_weight.pdf",
            "sensitivity_EMA_beta.pdf",
            "sensitivity_modulation_strength.pdf",
            None,  # 占位符，保持中间留白
            "sensitivity_spatial_relation_weight.pdf"
        ],
        "layout": (2, 3), # 2行3列
        "labels": [
            "(a) Avg Alpha", 
            "(b) Dual EMA fusion weight", 
            "(c) EMA Beta",
            "(d) Modulation strength", 
            None,
            "(e) Spatial relation weight"
        ],
        # 【关键修改】减小高度：从 (20, 12) 改为 (20, 8)
        # 这会迫使第一行和第二行靠得更近，消除中间过大的空白
        "figsize": (20, 8) 
    }
}

def pdf_to_high_res_image(pdf_path):
    """将PDF第一页转换为600dpi的高清numpy图像数组"""
    if not os.path.exists(pdf_path):
        print(f"警告: 文件未找到 {pdf_path}, 生成空白占位图。")
        return np.ones((600, 600, 3), dtype=np.uint8) * 255
    
    try:
        images = convert_from_path(pdf_path, dpi=TARGET_DPI)
        return np.array(images[0])
    except Exception as e:
        print(f"处理 {pdf_path} 时出错: {e}")
        return np.ones((600, 600, 3), dtype=np.uint8) * 255

def process_figures():
    for fig_name, config in FIGURES_CONFIG.items():
        print(f"正在处理 {fig_name} (Target: PDF, {TARGET_DPI} DPI)...")
        
        rows, cols = config['layout']
        files = config['files']
        labels = config['labels']
        
        # 创建画布
        # constrained_layout=True 会自动尽可能紧凑地排列子图
        fig, axes = plt.subplots(rows, cols, figsize=config['figsize'], constrained_layout=True)
        
        # 展平 axes 方便遍历
        if rows * cols > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
            
        # 遍历处理子图
        for i, ax in enumerate(axes_flat):
            if i < len(files):
                img_path = files[i]
                label_text = labels[i]

                # 检查是否为占位符 (None)
                if img_path is None:
                    ax.axis('off') 
                    continue       
                
                print(f"  -> 读取子图: {img_path}")
                img_data = pdf_to_high_res_image(img_path)
                
                ax.imshow(img_data)
                
                # 添加底部标签
                # 【修改说明】：
                # 1. fontname='Times New Roman' 设置字体为新罗马
                # 2. fontsize=16 控制字体大小，修改这个数字即可改变大小
                ax.set_xlabel(label_text, fontsize=26, labelpad=15, fontname='Times New Roman')
                
                # 移除坐标轴和边框
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                # 隐藏多余的空白格子
                ax.axis('off')

        output_file = f"{fig_name}.pdf"
        
        # 保存为 PDF
        plt.savefig(output_file, dpi=TARGET_DPI, bbox_inches='tight')
        
        print(f"成功生成: {output_file}")
        plt.close(fig)

if __name__ == "__main__":
    if not any(os.path.exists(f) for f in FIGURES_CONFIG['fig2']['files'] if f):
        print("提示：请确保所有源PDF文件都在当前目录下，否则将生成空白图。")
    
    process_figures()