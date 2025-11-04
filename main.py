import shutil

from ultralytics import YOLO
import json
import gc
import multiprocessing as mp
import torch
import time
from pathlib import Path


count = 0
def train(model_name, c):
    begin_time = time.time()
    model = YOLO("./ultralytics/cfg/models/11/yolo11.yaml")
    model.train(
        data=f"dataset/{c}/dataset.yaml",
        imgsz=512,
        epochs=300,
        batch=64,
        workers=1,
        device=0,
        name=f"{model_name}_{c}",
        patience=0
    )
    global count
    count= int(time.time() - begin_time)
    return load_best(model_name, c)


def load_best(model_name, c):
    base_dir = Path("runs/detect")
    name_prefix = f"{model_name}_{c}"

    # 找到所有以 name_prefix 开头的子目录
    candidates = [d for d in base_dir.glob(f"{name_prefix}*") if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found for {name_prefix}")

    latest_run = max(candidates, key=lambda d: d.stat().st_mtime)
    shutil.copytree(latest_run, f"results/{name_prefix}", dirs_exist_ok=True)
    best_path = latest_run / "weights" / "best.pt"

    print(f"Loading best model from: {best_path}")
    model = YOLO(str(best_path))
    model._run_dir = name_prefix
    return model


def save_result(model):
    # 在验证集上评估
    metrics = model.val(split='test')
    results = metrics.results_dict
    global count
    results['count_time_seconds'] = count

    # 获取 run 名称
    run_name = model._run_dir  # 使用我们手动保存的路径名
    result_path = Path("results") / f"{run_name}/{run_name}.json"
    result_path.parent.mkdir(exist_ok=True)

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

import multiprocessing as mp
import time

def train_worker(args):
    """在工作进程中运行训练"""
    model, epoch, batch_size, workers = args
    try:
        # 在新进程中重新导入

        print(f"进程开始训练: {model} epoch{epoch}")
        result = train(model, str(epoch), batch_size, workers)
        save_result(result)
        print(f"✓ 进程完成训练: {model} epoch{epoch}")
        return True
    except Exception as e:
        print(f"✗ 进程训练失败 {model} epoch{epoch}: {e}")
        return False

if __name__ == "__main__":
    for i in range(1, 11):
        print(f"\n{'='*60}")
        args = (i)
        process = mp.Process(target=train_worker, args=(args,))
        start_time = time.time()
        process.start()
        process.join()
        elapsed_time = time.time() - start_time

        # 检查进程退出状态
        if process.exitcode == 0:
            print(f"✅ 进程正常退出: channel{i} (耗时: {elapsed_time:.1f}秒)")
        else:
            print(f"❌ 进程异常退出: channel{i} (退出码: {process.exitcode})")

        process.close()

        print(f"🧹 进程资源已清理，内存完全释放")
        time.sleep(2)