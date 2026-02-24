import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 配置 =================
DATA_ROOT = "synthetic_EOG"  # 数据集根目录
CLASSES = ["Blink", "Left", "Right", "Up", "Down", "Rest"]
SAMPLES_PER_CLASS = 3  # 每个类别抽取的样本数

# ================= 工具函数 =================

def load_csv_robust(fpath):
    """
    鲁棒的 CSV 读取函数，能够处理不同的列名格式
    """
    try:
        df = pd.read_csv(fpath)
        cols = df.columns
        
        # 尝试匹配常见的列名格式
        if 'H' in cols and 'V' in cols:
            return df['H'].values, df['V'].values
        elif 'EOG_H' in cols and 'EOG_V' in cols:
            return df['EOG_H'].values, df['EOG_V'].values
        elif 'data 0' in cols and 'data 1' in cols:
            return df['data 0'].values, df['data 1'].values
        else:
            # 如果找不到名字，尝试按位置读取前两列
            if df.shape[1] >= 2:
                return df.iloc[:, 0].values, df.iloc[:, 1].values
            return None, None
    except Exception as e:
        print(f"读取失败 {fpath}: {e}")
        return None, None

def visualize_dataset():
    if not os.path.exists(DATA_ROOT):
        print(f"[错误] 找不到目录: {DATA_ROOT}")
        print("请确保脚本与 synthetic_EOG 文件夹在同一级，或修改 DATA_ROOT 路径。")
        return

    # 创建画布: 6行 x 3列
    fig, axes = plt.subplots(nrows=len(CLASSES), ncols=SAMPLES_PER_CLASS, 
                             figsize=(15, 12), constrained_layout=True)
    
    fig.suptitle(f'Random Samples from {DATA_ROOT}', fontsize=16)

    for i, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_ROOT, cls_name)
        
        # 1. 检查文件夹是否存在
        if not os.path.exists(cls_dir):
            print(f"[警告] 文件夹不存在: {cls_dir}，跳过...")
            continue
            
        # 2. 获取所有 CSV 文件
        files = [f for f in os.listdir(cls_dir) if f.endswith('.csv')]
        
        if not files:
            print(f"[警告] {cls_name} 文件夹为空")
            continue
            
        # 3. 随机抽取 3 个 (如果不足 3 个，则取全部)
        count = min(len(files), SAMPLES_PER_CLASS)
        selected_files = random.sample(files, count)
        
        # 4. 绘图
        for j in range(SAMPLES_PER_CLASS):
            ax = axes[i, j]
            
            # 如果该类别文件不足3个，剩下的子图隐藏掉
            if j >= len(selected_files):
                ax.axis('off')
                continue
                
            fname = selected_files[j]
            fpath = os.path.join(cls_dir, fname)
            
            h_data, v_data = load_csv_robust(fpath)
            
            if h_data is not None:
                # 绘制波形
                ax.plot(h_data, label='Horz', color='#1f77b4', linewidth=1.2)
                ax.plot(v_data, label='Vert', color='#ff7f0e', linewidth=1.2)
                
                # 设置标题 (文件名)
                ax.set_title(fname, fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # 只在第一列显示 Y 轴标签 (类别名)
                if j == 0:
                    ax.set_ylabel(cls_name, fontsize=14, fontweight='bold', rotation=0, labelpad=40)
                
                # 只在第一行的第一个图显示图例，避免杂乱
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize='small')
            else:
                ax.text(0.5, 0.5, "Data Error", ha='center', va='center')

    print("绘图完成，正在显示窗口...")
    plt.show()

if __name__ == "__main__":
    visualize_dataset()