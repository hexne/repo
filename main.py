import shutil

from ultralytics import YOLO
import json
from pathlib import Path

def train(model_name, c, batch = 32):
    lr = 0.01 * batch / 64
    model = YOLO(f"{model_name}.pt")
    model.train(
        data=f"{c}/dataset.yaml",
        imgsz=512,
        epochs=3,
        batch=batch,
        workers=8,
        device=0,
        name=f"{model_name}_{c}",
        lr0=lr,
    )
    return load_best(model_name, c)


def load_best(model_name, c):
    base_dir = Path("runs/detect")
    name_prefix = f"{model_name}_{c}"

    # 找到所有以 name_prefix 开头的子目录
    candidates = [d for d in base_dir.glob(f"{name_prefix}*") if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"❌ No runs found for {name_prefix}")

    latest_run = max(candidates, key=lambda d: d.stat().st_mtime)
    shutil.copytree(latest_run, f"results/{name_prefix}", dirs_exist_ok=True)
    best_path = latest_run / "weights" / "best.pt"

    print(f"📦 Loading best model from: {best_path}")
    model = YOLO(str(best_path))
    model._run_dir = name_prefix
    return model


def save_result(model):
    # 在验证集上评估
    metrics = model.val(split='test')
    results = metrics.results_dict

    # 获取 run 名称
    run_name = model._run_dir  # 使用我们手动保存的路径名
    result_path = Path("results") / f"{run_name}/{run_name}.json"
    result_path.parent.mkdir(exist_ok=True)

    # 保存为 JSON
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    models = ['yolo11n', 'yolo12n']
    for model in models:
        batch = 32 if model == "yolo11n" else 16
        for i in range(1, 3):
            save_result(train(model, str(i), batch))