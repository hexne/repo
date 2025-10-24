from ultralytics import YOLO
import numpy as np
import json
from tifffile import tifffile
import os


def train_and_evaluate():
    # im = tifffile.imread('/home/hexne/yolo修改/1/images/train/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_1.tif')
    # shape = im.shape
    # print(shape)
    # exit(0)
    # 训练模型
    model = YOLO('yolo11n.pt')

    model.train(
        data='7v/dataset.yaml',
        imgsz=512,
        epochs=300,
        batch=32,
        workers=8,
        device=0,
        name='1',
        lr0=0.01 / 2,
#        deterministic=False
    )
    print("✅ 训练完成!")

    # 验证最佳模型
    print("\n🔍 开始在测试集上验证最佳模型...")
    # 获取最新 runs/detect 子目录
    import glob

    detect_dirs = sorted(glob.glob('runs/detect/*'), key=os.path.getmtime)
    latest_dir = detect_dirs[-1] if detect_dirs else None
    best_model_path = os.path.join(latest_dir, 'weights', 'best.pt')

    # best_model_path = 'runs/detect/1/weights/best.pt'
    best = YOLO(best_model_path)

    # 进行验证
    val_results = best.val(split='test')

    # 显示主要指标
    print("\n" + "=" * 50)
    print("📊 测试集评估结果")
    print("=" * 50)

    # 检测任务指标
    if hasattr(val_results, 'box'):
        print(f"📦 检测任务指标:")
        print(f"   mAP@0.5:     {val_results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {val_results.box.map:.4f}")

        # 修正：精确度和召回率是数组，取平均值
        if hasattr(val_results.box, 'p'):
            precision = np.mean(val_results.box.p) if isinstance(val_results.box.p, np.ndarray) else val_results.box.p
            recall = np.mean(val_results.box.r) if isinstance(val_results.box.r, np.ndarray) else val_results.box.r
            print(f"   精确度 (Precision): {precision:.4f}")
            print(f"   召回率 (Recall):    {recall:.4f}")

    # 分割任务指标
    if hasattr(val_results, 'seg'):
        print(f"🎯 分割任务指标:")
        print(f"   掩码 mAP@0.5:     {val_results.seg.map50:.4f}")
        print(f"   掩码 mAP@0.5:0.95: {val_results.seg.map:.4f}")

        # 计算并显示IoU和Dice系数
        if hasattr(val_results.seg, 'miou'):
            print(f"   平均IoU: {val_results.seg.miou:.4f}")
            # Dice系数 = 2 * IoU / (300_基准 + IoU)
            dice = 2 * val_results.seg.miou / (1 + val_results.seg.miou)
            print(f"   Dice系数: {dice:.4f}")

    # 计算IoU和Dice系数
    print(f"\n🎯 IoU和Dice系数计算:")

    # 方法1: 使用mAP50作为IoU的近似值
    if hasattr(val_results, 'box'):
        iou_approx = val_results.box.map50
        dice_approx = 2 * iou_approx / (1 + iou_approx)
        print(f"   基于检测mAP50的估算:")
        print(f"   IoU ≈ {iou_approx:.4f}")
        print(f"   Dice系数 ≈ {dice_approx:.4f}")

    # 方法2: 如果你需要更精确的IoU，可以手动计算
    print(f"\n📈 详细类别指标:")
    if hasattr(val_results, 'names') and hasattr(val_results, 'box'):
        if hasattr(val_results.box, 'ap50') and isinstance(val_results.box.ap50, np.ndarray):
            for i, class_name in val_results.names.items():
                if i < len(val_results.box.ap50):
                    ap50 = val_results.box.ap50[i]
                    class_iou = ap50  # 使用AP50作为该类别的IoU近似
                    class_dice = 2 * class_iou / (1 + class_iou) if class_iou > 0 else 0
                    print(f"   {class_name}: AP50={ap50:.4f}, IoU≈{class_iou:.4f}, Dice≈{class_dice:.4f}")

    # 保存详细结果到文件
    save_detailed_results(val_results, best_model_path)

    return val_results, best


def save_detailed_results(val_results, model_path):
    """保存详细结果到文本文件"""
    results_dir = os.path.dirname(model_path)
    results_file = os.path.join(results_dir, 'test_results.txt')

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("YOLOv8 测试集评估结果\n")
        f.write("=" * 50 + "\n\n")

        # 检测指标
        if hasattr(val_results, 'box'):
            f.write("检测任务指标:\n")
            f.write(f"mAP@0.5:     {val_results.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {val_results.box.map:.4f}\n")

            # 修正：处理精确度和召回率数组
            if hasattr(val_results.box, 'p'):
                precision = np.mean(val_results.box.p) if isinstance(val_results.box.p,
                                                                     np.ndarray) else val_results.box.p
                recall = np.mean(val_results.box.r) if isinstance(val_results.box.r, np.ndarray) else val_results.box.r
                f.write(f"精确度 (Precision): {precision:.4f}\n")
                f.write(f"召回率 (Recall):    {recall:.4f}\n\n")

        # 分割指标
        if hasattr(val_results, 'seg'):
            f.write("分割任务指标:\n")
            f.write(f"掩码 mAP@0.5:     {val_results.seg.map50:.4f}\n")
            f.write(f"掩码 mAP@0.5:0.95: {val_results.seg.map:.4f}\n")

            if hasattr(val_results.seg, 'miou'):
                f.write(f"平均IoU: {val_results.seg.miou:.4f}\n")
                dice = 2 * val_results.seg.miou / (1 + val_results.seg.miou)
                f.write(f"Dice系数: {dice:.4f}\n\n")

        # IoU和Dice计算
        f.write("IoU和Dice系数:\n")
        if hasattr(val_results, 'box'):
            iou_approx = val_results.box.map50
            dice_approx = 2 * iou_approx / (1 + iou_approx)
            f.write(f"基于mAP50的IoU估算: {iou_approx:.4f}\n")
            f.write(f"基于mAP50的Dice系数估算: {dice_approx:.4f}\n\n")

        # 各类别详细结果
        if hasattr(val_results, 'names') and hasattr(val_results, 'box'):
            f.write("各类别详细指标:\n")
            if hasattr(val_results.box, 'ap50') and isinstance(val_results.box.ap50, np.ndarray):
                for i, class_name in val_results.names.items():
                    if i < len(val_results.box.ap50):
                        ap50 = val_results.box.ap50[i]
                        class_iou = ap50
                        class_dice = 2 * class_iou / (1 + class_iou) if class_iou > 0 else 0
                        f.write(f"  {class_name}:\n")
                        f.write(f"    AP50: {ap50:.4f}\n")
                        f.write(f"    IoU≈: {class_iou:.4f}\n")
                        f.write(f"    Dice≈: {class_dice:.4f}\n")

    print(f"📄 详细结果已保存至: {results_file}")


def calculate_exact_iou_dice(model, test_images_path):
    """如果需要更精确的IoU和Dice，可以使用这个方法"""
    print(f"\n🎯 计算精确的IoU和Dice系数...")

    try:
        # 这里可以实现精确的IoU和Dice计算
        # 需要获取预测框和真实框的详细数据
        print("⚠️  注意：精确计算需要分割模型和分割标注")
        print("💡 当前使用的是检测模型，IoU和Dice基于mAP50估算")

    except Exception as e:
        print(f"计算精确指标时出错: {e}")


if __name__ == "__main__":
    # 执行训练和评估
    val_results, best_model = train_and_evaluate()

