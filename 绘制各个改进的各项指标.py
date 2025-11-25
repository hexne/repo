import os
import pandas as pd
import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_metric(dirs, metric_name, title, ylabel, save_name):
    plt.figure(figsize=(10, 6))
    for d in dirs:
        results_path = os.path.join(d, "results.csv")
        if not os.path.exists(results_path):
            continue

        try:
            df = pd.read_csv(results_path)
            if metric_name not in df.columns:
                print(f"{metric_name} not found in {results_path}")
                continue

            label = os.path.basename(os.path.dirname(d))
            plt.plot(df.index, df[metric_name], label=label)
        except Exception as e:
            print(f"Error reading {results_path}: {e}")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_name}.png")
    plt.close()


def draw_precision(dirs):
    draw_metric(dirs, "metrics/precision(B)", "Precision Curve", "Precision", "BoxP_curve")


def draw_recall(dirs):
    draw_metric(dirs, "metrics/recall(B)", "Recall Curve", "Recall", "BoxR_curve")


def draw_map5(dirs):
    draw_metric(dirs, "metrics/mAP50(B)", "mAP@0.5 Curve", "mAP@0.5", "BoxF_curve")


def draw_map95(dirs):
    draw_metric(dirs, "metrics/mAP50-95(B)", "mAP@0.5:0.95 Curve", "mAP@0.5:0.95", "BoxF95_curve")


def draw_f1(dirs):
    plt.figure(figsize=(10, 6))
    for d in dirs:
        results_path = os.path.join(d, "results.csv")
        if not os.path.exists(results_path):
            continue

        try:
            df = pd.read_csv(results_path)
            if "metrics/precision(B)" not in df.columns or "metrics/recall(B)" not in df.columns:
                print(f"Missing precision/recall in {results_path}")
                continue

            precision = df["metrics/precision(B)"]
            recall = df["metrics/recall(B)"]
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            label = os.path.basename(os.path.dirname(d))
            plt.plot(df.index, f1, label=label)
        except Exception as e:
            print(f"Error reading {results_path}: {e}")

    plt.title("F1 Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("BoxF1_curve.png")
    plt.close()


if __name__ == '__main__':
    base_dir = r"C:\Users\hexne\Desktop\results"
    dirs = [os.path.join(base_dir, f, "1") for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f, "5"))]

    draw_precision(dirs)
    draw_recall(dirs)
    draw_map5(dirs)
    draw_map95(dirs)
    draw_f1(dirs)
