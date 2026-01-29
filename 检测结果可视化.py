from ultralytics import YOLO


model = YOLO(r"results/yolo26n_1/weights/best.pt")
results = model.predict(
    source="dataset/1/images/test",
    save=True
)
print("推理完成，结果已保存到 runs/predict/exp/")

