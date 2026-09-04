from ultralytics import YOLO, SALWANet


model = SALWANet("best.pt")
res = model.val(data="VisDrone.yaml", batch=8, imgsz=640, save_json=True)
