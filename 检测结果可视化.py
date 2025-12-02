from ultralytics import YOLO


model = YOLO("C:/Users/hexne/Desktop/论文材料/消融实验结果和模型/finish_5/weights/best.pt")
results = model.predict(
    source="S:/Projects/repo/dataset/5/images/test",
    save=True
)
print("推理完成，结果已保存到 runs/predict/exp/")

