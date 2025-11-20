import shutil

from ultralytics import YOLO
import json
import gc
import multiprocessing as mp
import torch
import time
from pathlib import Path

count = 0


def train(model_name, c, batch, worker):
    lr = 0.01 * batch / 64
    begin_time = time.time()
    model = YOLO(f"{model_name}.yaml")
    model.train(
        data=f"dataset/{c}/dataset.yaml",
        imgsz=512,
        epochs=300,
        batch=batch,
        workers=worker,
        device=0,
        name=f"{model_name}_{c}",
        lr0=lr,
        patience=0,
        amp=False,
        # deterministic=False
    )
    global count
    count = int(time.time() - begin_time)
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
    result = train(model, str(epoch), batch_size, workers)
    save_result(result)


if __name__ == "__main__":
    models = ['test4']
    batchs = [32, 32]
    workers = [4, 4]

    for model in models:
        index = 0
        for i in [5]:
            # 准备参数
            args = (model, i, batchs[index], workers[index])

            # 创建进程
            process = mp.Process(target=train_worker, args=(args,))

            # 记录开始时间
            start_time = time.time()

            # 启动进程
            process.start()

            # 等待进程完成（阻塞，不会并行）
            process.join()

            # 计算耗时
            elapsed_time = time.time() - start_time

            # 清理进程资源
            process.close()

            time.sleep(2)

            print(f"准备下一个训练任务...\n")
        index += 1