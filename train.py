from ultralytics import YOLO, SALWANet

model_name = ["SALWANetn"]

for name in model_name:
    model = SALWANet(f"{name}.yaml")  # build a new model from scratch
    # model = YOLO(f"{name}.yaml")  # build a new model from scratch  
    model.train(data="VisDrone.yaml",
                epochs=300,
                batch=8,
                imgsz=640,
                workers=8
                )
