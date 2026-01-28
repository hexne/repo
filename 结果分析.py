import os
from pathlib import Path
import numpy as np
import torch
from tifffile import imread
from ultralytics import YOLO
import matplotlib.pyplot as plt

# -----------------------------
# IoU 计算函数
# -----------------------------
def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# -----------------------------
# 检测分析：统计 TP / FP / FN
# -----------------------------
def analyze_detections(predictions, ground_truths, iou_thr=0.5):
    predictions = sorted(predictions, key=lambda x: -x[1])  # 按置信度排序
    matched = {k: np.zeros(len(v)) for k,v in ground_truths.items()}

    tp, fp, fn = 0, 0, 0
    tp_list, fp_list, fn_list = [], [], []

    for img_id, conf, box in predictions:
        gts = ground_truths.get(img_id, [])
        ious = [iou(box, gt) for gt in gts]
        if len(ious) > 0 and max(ious) >= iou_thr:
            j = np.argmax(ious)
            if matched[img_id][j] == 0:
                matched[img_id][j] = 1
                tp += 1
                tp_list.append((conf, box))  # 保存 TP
            else:
                fp += 1
                fp_list.append((conf, box))
        else:
            fp += 1
            fp_list.append((conf, box))

    for img_id, gts in ground_truths.items():
        for j, gt in enumerate(gts):
            if matched[img_id][j] == 0:
                fn += 1
                fn_list.append(gt)

    return tp, fp, fn, tp_list, fp_list, fn_list

# -----------------------------
# 绘图函数（增加保存功能）
# -----------------------------
def plot_results(tp_list, fp_list, fn_list, precision, recall, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    # 1. TP 置信度直方图
    tp_confidences = [conf for conf, _ in tp_list]
    plt.figure(figsize=(6,4))
    plt.hist(tp_confidences, bins=10, color='skyblue', edgecolor='black')
    plt.title(f"TP Confidence Distribution\nPrecision={precision:.3f}, Recall={recall:.3f}")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tp_confidence_hist.png"))
    plt.close()

    # 2. FP 尺寸分布 (宽度)
    fp_widths = [(box[2]-box[0]) for _, box in fp_list]
    plt.figure(figsize=(6,4))
    plt.hist(fp_widths, bins=10, color='salmon', edgecolor='black')
    plt.title("FP Size Distribution (Width)")
    plt.xlabel("Width")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fp_width_hist.png"))
    plt.close()

    # 3. FN 尺寸分布 (宽度)
    fn_widths = [(box[2]-box[0]) for box in fn_list]
    plt.figure(figsize=(6,4))
    plt.hist(fn_widths, bins=10, color='lightgreen', edgecolor='black')
    plt.title("FN Size Distribution (Width)")
    plt.xlabel("Width")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fn_width_hist.png"))
    plt.close()

    # 4. TP 尺寸分布 (宽度)
    tp_widths = [(box[2]-box[0]) for _, box in tp_list]
    plt.figure(figsize=(6,4))
    plt.hist(tp_widths, bins=10, color='orange', edgecolor='black')
    plt.title("TP Size Distribution (Width)")
    plt.xlabel("Width")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tp_width_hist.png"))
    plt.close()

    print(f"所有图像已保存到文件夹: {save_dir}")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    model = YOLO("best.pt")
    images_dir = "dataset/5/images/test"
    labels_dir = "dataset/5/labels/test"

    file_names = [f.stem for f in Path(images_dir).iterdir() if f.is_file()]

    predictions = []   # (image_id, conf, [x1,y1,x2,y2])
    ground_truths = {} # {image_id: [[x1,y1,x2,y2], ...]}

    for file_name in file_names:
        image_path = f"{images_dir}/{file_name}.tiff"
        label_path = f"{labels_dir}/{file_name}.txt"

        # 读取图像
        img = imread(image_path)
        img = np.transpose(img, (2, 0, 1))  # (5, 512, 512)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0)

        # 推理
        boxes = model.predict(tensor, verbose=False)[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i])
            predictions.append((file_name, conf, [x1, y1, x2, y2]))

        # 读取真值
        gts = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.strip().split())
                    x1 = (cx - w/2) * img.shape[2]
                    y1 = (cy - h/2) * img.shape[1]
                    x2 = (cx + w/2) * img.shape[2]
                    y2 = (cy + h/2) * img.shape[1]
                    gts.append([x1, y1, x2, y2])
        ground_truths[file_name] = gts

    # 分析
    tp, fp, fn, tp_list, fp_list, fn_list = analyze_detections(predictions, ground_truths)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"Precision={precision:.3f}, Recall={recall:.3f}")

    # 绘制并保存图像到指定文件夹
    plot_results(tp_list, fp_list, fn_list, precision, recall, save_dir="结果分析/fl损失函数")
