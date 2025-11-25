import os
import matplotlib.pyplot as plt
import numpy as np


def read_yolo_labels(labels_dir):
    """读取YOLO标签文件并提取宽度"""
    widths = []
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(labels_dir, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        widths.append(float(parts[3]))
    return widths


def plot_width_distribution_en(widths, title, save_path=None):
    """使用英文标签绘制分布图"""
    plt.figure(figsize=(10, 6))

    widths_array = np.array(widths)
    mean_width = np.mean(widths_array)
    median_width = np.median(widths_array)
    std_width = np.std(widths_array)

    # 绘制直方图
    n, bins, patches = plt.hist(widths_array, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

    # 英文统计信息
    stats_text = f'Mean: {mean_width:.4f}\nMedian: {median_width:.4f}\nStd: {std_width:.4f}\nTotal: {len(widths)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)

    plt.xlabel('Bounding Box Width (normalized)')
    plt.ylabel('Frequency')
    plt.title(f'{title} - Bounding Box Width Distribution')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return mean_width, median_width, std_width


def analyze_dataset_en(base_path):
    """英文版本的数据集分析"""
    splits = ['train', 'test', 'val']
    results = {}

    for split in splits:
        labels_path = os.path.join(base_path, 'labels', split)

        if os.path.exists(labels_path):
            print(f"Processing {split} set...")
            widths = read_yolo_labels(labels_path)

            if widths:
                save_path = f'{split}_bbox_width_distribution.png'
                mean, median, std = plot_width_distribution_en(widths, split.upper(), save_path)

                results[split] = {
                    'count': len(widths),
                    'mean': mean,
                    'median': median,
                    'std': std
                }

                print(f"{split.upper()} Set Analysis:")
                print(f"  Bounding boxes: {len(widths)}")
                print(f"  Width range: [{min(widths):.4f}, {max(widths):.4f}]")
                print(f"  Mean width: {mean:.4f}")
                print(f"  Median width: {median:.4f}")
                print(f"  Std: {std:.4f}")
                print("-" * 50)

    return results


# 使用英文版本
if __name__ == "__main__":
    dataset_path = "dataset\\1p"  # 修改为你的数据集路径
    results = analyze_dataset_en(dataset_path)

    print("\n" + "=" * 50)
    print("Dataset Summary:")
    print("=" * 50)
    for split, stats in results.items():
        print(f"{split.upper()}: {stats['count']} boxes, "
              f"mean width: {stats['mean']:.4f} ± {stats['std']:.4f}")