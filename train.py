from ultralytics import YOLO, SALWANet

model_name = ["SALWANetn"]

for name in model_name:
    model = SALWANet(f"{name}.yaml")  # build a new model from scratch
    # model = YOLO(f"{name}.yaml")  # build a new model from scratch  
    model.train(data="VisDrone.yaml",
                epochs=300,
                batch=8,
                imgsz=640,
                workers=8,
                # save_period=1,
                # optimizer="AdamW",
                # lr0=0.001,
                # resume=True,
                project="runs/0830VisDrone",
                name=f"VisDrone_DET_{name}_e300bs8sz640_k37"
                )
