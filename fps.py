import time
import torch
from ultralytics import YOLO

def get_fps(path, img_size=640, num_runs=1000):
    """
    测试 YOLO 模型的 FPS (Frames Per Second)

    Args:
        path (str): 模型文件路径，例如 'runs/train/exp/weights/best.pt'
        img_size (int): 输入图像尺寸，默认 640
        num_runs (int): 测试次数，默认 1000

    Returns:
        float: 模型推理 FPS
    """
    # 加载模型
    model = YOLO(path)

    # 构造虚拟输入
    dummy_input = torch.randn(1, 10, img_size, img_size).cuda()

    # warm-up，避免第一次运行的初始化开销
    for _ in range(50):
        _ = model(dummy_input)

    # 正式计时
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_runs):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.perf_counter()

    fps = num_runs / (end - start)
    return fps

if __name__ == "__main__":
    model_path = "results/yolo26n_10/weights/best.pt"  # 换成你自己的模型路径
    fps = get_fps(model_path)
    print(f"fps is {fps:.2f}")
