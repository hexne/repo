import shutil

from ultralytics import YOLO
import json
import gc
import multiprocessing as mp
import torch
import time
from pathlib import Path



def freeze_backbone(model):
    for name, param in model.model.named_parameters():
        if name.startswith("model.0"):  # backbone
            param.requires_grad = False


def check_freeze(model):
    frozen, trainable = [], []
    for name, p in model.model.named_parameters():
        if not p.requires_grad:
            frozen.append(name)
        else:
            trainable.append(name)
    print(f"Frozen layers: {len(frozen)}")
    print(f"Trainable layers: {len(trainable)}")
    return frozen, trainable

def partial_freeze_vit(model):
    vit_backbone = model.model.model[0]  # 正确访问 VitBackbone
    dino = vit_backbone.model            # DINOv3 模型

    # 1. 冻结 patch embedding
    if hasattr(dino, 'patch_embed'):
        for param in dino.patch_embed.parameters():
            param.requires_grad = False

    # 2. 冻结 positional embedding
    for attr in ['pos_embed', 'cls_token', 'storage_tokens', 'mask_token']:
        if hasattr(dino, attr):
            getattr(dino, attr).requires_grad = False

    # 3. 部分冻结 transformer blocks
    if hasattr(dino, 'blocks'):
        num_blocks = len(dino.blocks)
        unfreeze_blocks = min(4, num_blocks // 3)
        freeze_until = max(0, num_blocks - unfreeze_blocks)

        for i, block in enumerate(dino.blocks):
            freeze = (i < freeze_until)
            for param in block.parameters():
                param.requires_grad = not freeze

        print(f"冻结前 {freeze_until}/{num_blocks} 个 transformer block")

    # 4. 解冻最后的 norm 层
    if hasattr(dino, 'norm'):
        for param in dino.norm.parameters():
            param.requires_grad = True



count = 0
def train(c):
    begin_time = time.time()
    model = YOLO("cfg/dinovit.yaml")
    partial_freeze_vit(model)
    model.train(
        data=f"./dataset/{c}/dataset.yaml",
        imgsz=512,
        epochs=300,
        batch=32,
        workers=8,
        device=0,
        name=f"{c}",
        patience=0,
        freeze=0,
    )
    partial_freeze_vit(model)
    global count
    count= int(time.time() - begin_time)
    return load_best(c)


def load_best(c):
    base_dir = Path("runs/detect")
    name_prefix = f"{c}"

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
    save_result(train('1p'))