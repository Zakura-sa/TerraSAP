import matplotlib.pyplot as plt
from PIL import Image
import os

# ================= 配置区域 =================
# 1. 设置 IEEE 风格字体 (Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 2. 输出分辨率
DPI = 600

# 3. 图片处理任务列表
tasks = [
    {
        "filename": "UCMerced_samples.png",
        "rows": 2, "cols": 4,
        "crop_top": 0.12, 
        "titles": [
            "Airplane", "Buildings", "Forest", "Harbor",
            "Intersection", "Parking Lot", "River", "Tennis Court"
        ]
    },
    {
        "filename": "NWPU_RESISC45_samples.png",
        "rows": 2, "cols": 4,
        "crop_top": 0.12,
        "titles": [
            "Airplane", "Airport", "Bridge", "Church",
            "Forest", "Harbor", "Lake", "Mountain"
        ]
    },
    {
        "filename": "MSTAR_samples.png",
        "rows": 2, "cols": 4,
        "crop_top": 0.12,
        "titles": [
            "T72", "BMP2", "BTR70", "BTR60",
            "2S1", "D7", "ZSU234", "ZIL131"
        ]
    },
]

def process_and_rebuild_image(task):
    filename = task['filename']
    rows = task['rows']
    cols = task['cols']
    crop_top_ratio = task['crop_top']
    titles = task['titles']

    if not os.path.exists(filename):
        print(f"错误: 找不到文件 {filename}，请确保图片在当前目录下。")
        return

    # 读取原图
    img = Image.open(filename)
    img_w, img_h = img.size
    
    # 计算每个小格子的尺寸
    cell_w = img_w // cols
    cell_h = img_h // rows

    # 创建绘图对象
    # 修改1: 减小figsize的高度系数，使画布更紧凑
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.2))
    axes = axes.flatten()

    print(f"正在处理: {filename}...")

    for i in range(rows * cols):
        ax = axes[i]
        
        # 计算裁剪区域
        r, c = divmod(i, cols)
        left = c * cell_w
        top = r * cell_h
        right = left + cell_w
        bottom = top + cell_h
        
        # 1. 切割出小图
        cell_img = img.crop((left, top, right, bottom))
        
        # 2. 去除旧标题
        crop_pixels = int(cell_img.height * crop_top_ratio)
        content_img = cell_img.crop((0, crop_pixels, cell_img.width, cell_img.height))
        
        # 3. 显示图片
        ax.imshow(content_img)
        
        # 4. 设置新标题
        if i < len(titles):
            w = content_img.width
            h = content_img.height
            
            # 修改2: 让文字更贴近图片边缘 (从 -0.05 改为 -0.02)
            ax.text(
                x=w/2, 
                y=-h * 0.02, 
                s=titles[i], 
                fontname='Times New Roman', 
                fontsize=14, 
                ha='center',
                va='bottom'
            )
        
        # 移除坐标轴
        ax.axis('off')

    # 修改3: 大幅减小 h_pad (行间距) 和 w_pad (列间距)
    # h_pad=0.3 可以显著减少行与行之间的空白
    plt.tight_layout(h_pad=0.3, w_pad=0.1)
    
    # 保存为 PDF
    output_filename = filename.replace(".png", ".pdf")
    if output_filename == filename:
         output_filename = filename + "_fixed.pdf"

    # bbox_inches='tight' 会自动切除周围多余白边
    plt.savefig(output_filename, dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    print(f"已保存: {output_filename}")
    plt.close()

# ================= 主程序 =================
if __name__ == "__main__":
    for task in tasks:
        try:
            process_and_rebuild_image(task)
        except Exception as e:
            print(f"处理 {task['filename']} 时出错: {e}")

    print("全部处理完成。")