import time
import torch
from ultralytics import YOLO

def get_fps(path, img_size=512, num_runs=1000, ch=1, warmup=20):
    """
    Benchmark YOLO model FPS with synthetic input.
    """
    model = YOLO(path)
    dummy_input = torch.randn(1, ch, img_size, img_size).cuda()

    # Warm-up
    for _ in range(warmup):
        _ = model(dummy_input)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_runs):
        _ = model(dummy_input)

    torch.cuda.synchronize()
    end = time.perf_counter()

    return num_runs / (end - start)

if __name__ == "__main__":
    base_dir = "results"
    model_paths = [
        f"{base_dir}/finish_1/weights/best.pt",
        f"{base_dir}/finish_5/weights/best.pt",
        f"{base_dir}/WTB_1/weights/best.pt",
        f"{base_dir}/WTB_5/weights/best.pt",
        f"{base_dir}/PEM_1/weights/best.pt",
        f"{base_dir}/PEM_5/weights/best.pt",
    ]

    results = []
    for model_path in model_paths:
        ch = 5 if "5" in model_path else 1
        fps = get_fps(model_path, ch=ch)
        results.append(f"{model_path} fps is {fps:.2f}")

    for line in results:
        print(line)
