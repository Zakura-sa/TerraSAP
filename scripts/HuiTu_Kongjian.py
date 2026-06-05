import matplotlib.pyplot as plt
from PIL import Image
import os

# ================= 1. 基础配置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
DPI = 600
FONT_SIZE = 14

# 顶部裁剪比例 (去除旧标题)
CROP_TOP_RATIO = 0.18 

# 文字垂直悬浮高度 (负数表示在图片上方)
# -0.01 表示悬浮在图片顶端上方 1% 的位置 (比之前的 -0.02 更贴近图片)
TEXT_Y_OFFSET = -0.01

# ================= 2. 间距控制 (新功能) =================
# 行间距 (上下两行图片的距离)。数值越小，两行靠得越近。
# 建议范围: 0.10 ~ 0.20 (原图看起来比较紧凑，这里设为 0.15)
ROW_SPACING = 0.15

# 列间距 (左右两列图片的距离)。数值越小，左右靠得越近。
# 建议范围: 0.01 ~ 0.05
COL_SPACING = 0.02

# ================= 3. 文字左右微调区 =================
# 左侧列 (索引 0 和 3): 负数左移，正数右移
LEFT_COL_X_OFFSET = -120

# 右侧列 (索引 2 和 5): 负数左移，正数右移
RIGHT_COL_X_OFFSET = 120

# 定义处理任务
tasks = [
    {
        "filename": "Bridge_GradCAM_blocks10.png",
        "rows": 2, "cols": 3,
        "titles": [
            "Original Image", "With Spatial Awareness\nGradCAM(block 10)", "Without Spatial Awareness\nGradCAM(block 10)",
            "Attention Difference", "With Spatial Awareness\n(Heatmap Only)", "Without Spatial Awareness\n(Heatmap Only)"
        ]
    },
    {
        "filename": "Bridge_ScoreCAM_blocks10.png",
        "rows": 2, "cols": 3,
        "titles": [
            "Original Image", "With Spatial Awareness\nScoreCAM(block 10)", "Without Spatial Awareness\nScoreCAM(blocks 10)",
            "Attention Difference", "With Spatial Awareness\n(Heatmap Only)", "Without Spatial Awareness\n(Heatmap Only)"
        ]
    }
]

def process_image(task):
    filename = task['filename']
    rows = task['rows']
    cols = task['cols']
    titles = task['titles']

    if not os.path.exists(filename):
        print(f"错误: 找不到文件 {filename}")
        return

    # 读取原图
    img = Image.open(filename)
    img_w, img_h = img.size
    cell_w = img_w // cols
    cell_h = img_h // rows
    
    # 裁剪高度
    crop_pixels = int(cell_h * CROP_TOP_RATIO)
    final_cell_h = cell_h - crop_pixels
    
    # 动态计算画布尺寸 (系数从 1.3 调小到 1.2，减少整体白边)
    fig_width = 12 
    aspect_ratio = (final_cell_h * rows) / (cell_w * cols)
    fig_height = fig_width * aspect_ratio * 1.2 
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    print(f"正在处理: {filename}...")

    for i in range(rows * cols):
        ax = axes[i]
        
        # 计算格子位置
        r, c = divmod(i, cols)
        left = c * cell_w
        top = r * cell_h
        right = left + cell_w
        bottom = top + cell_h
        
        # 裁剪图片
        cell_img = img.crop((left, top, right, bottom))
        content_img = cell_img.crop((0, crop_pixels, cell_img.width, cell_img.height))
        
        # 显示图片
        ax.imshow(content_img)
        
        if i < len(titles):
            w = content_img.width
            h = content_img.height
            
            # === 计算 X 坐标 ===
            x_pos = w / 2 # 默认居中
            
            # 1. 左侧列修正 (索引 0, 3)
            if i == 0 or i == 3:
                x_pos += LEFT_COL_X_OFFSET
            
            # 2. 右侧列修正 (索引 2, 5)
            elif i == 2 or i == 5:
                x_pos += RIGHT_COL_X_OFFSET

            # 绘制文字
            ax.text(
                x=x_pos, 
                y=h * TEXT_Y_OFFSET, 
                s=titles[i], 
                fontname='Times New Roman', 
                fontsize=FONT_SIZE, 
                ha='center', 
                va='bottom'
            )
        
        ax.axis('off')

    # === 布局收紧 ===
    # wspace: 列间距, hspace: 行间距
    plt.subplots_adjust(
        left=0.01, right=0.99, 
        top=0.95, bottom=0.05, 
        wspace=COL_SPACING, 
        hspace=ROW_SPACING
    )
    
    output_filename = filename.replace(".png", ".pdf")
    plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')
    print(f"已保存: {output_filename}")
    plt.close()

if __name__ == "__main__":
    for task in tasks:
        process_image(task)