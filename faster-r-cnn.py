import os
import cv2
import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo

def calc_max_iter(batch_size, epochs):
    iters_per_epoch = 824 // batch_size
    max_iter = iters_per_epoch * epochs
    return max_iter

def main():
    # 修改为你的类别列表
    classes = ["nodule"]

    # 注册数据集
    register_coco_instances("my_train", {}, "dataset/coco/annotations/train.json", "dataset/coco/images/train")
    register_coco_instances("my_val", {}, "dataset/coco/annotations/val.json", "dataset/coco/images/val")
    register_coco_instances("my_test", {}, "dataset/coco/annotations/test.json", "dataset/coco/images/test")

    # 配置
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ("my_val",)
    cfg.DATALOADER.NUM_WORKERS = 0   # ⚠️ Windows 下必须设为 0，避免多进程报错
    BATCH_SIZE=16
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = calc_max_iter(BATCH_SIZE, 300)
    cfg.SOLVER.STEPS = []            # 避免 STPS > MAX_ITER 的警告
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

    # 输出目录
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 训练
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # -------------------
    # 🔎 在 test 集合上评估
    # -------------------
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluator = COCOEvaluator("my_test", cfg, False, output_dir="./eval_results/")
    val_loader = build_detection_test_loader(cfg, "my_test")
    metrics = inference_on_dataset(trainer.model, val_loader, evaluator)

    print("📊 测试集评估结果:")
    print(metrics)   # 包含 precision, recall, mAP@0.5, mAP@0.95 等指标

    # -------------------
    # ✅ 推理 + 可视化
    # -------------------
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    test_img = "dataset/coco/images/test/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_0.png"  # 修改为你的测试图片路径
    img = cv2.imread(test_img)
    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("my_train"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # ⚠️ Windows 下建议保存图片而不是 imshow
    cv2.imwrite("result.png", out.get_image()[:, :, ::-1])
    print("✅ 推理结果已保存到 result.png")

if __name__ == "__main__":
    main()
