from ultralytics import YOLO, SALWANet

# Load a model
# model = YOLO("runs/NEU-DET/NEU_DET_TGFINet_mod_e300sz224bs32/weights/epoch150.pt")  # build a new model from scratch
# model = YOLO("runs/GC-DET/GC_DET_TGFINet_mod_SGD_e300sz512bs322/weights/epoch251.pt")  # build a new model from scratch

# model = SSTN("yolov8_test.yaml")  # build a new model from scratch
# model = HCSYOLO("yolov8_test1.yaml")  # build a new model from scratch

# model = YOLO("MSRDet.yaml")  # build a new model from scratch
# model = YOLO("/home/bbc1335/Documents/Detection/ultralytics/runs/detect/YOLOv8改3/weights/best.pt")  # Load a pretrained model (recommended for training)

# model_name = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolo11m", "yolo11l"]

# for name in model_name:
#     model = YOLO(f"{name}.yaml")  # build a new model from scratch
#     # Use the model
#     model.train(data="VisDrone.yaml",
#                 epochs=300,
#                 batch=8,
#                 imgsz=640,
#                 workers=8,
#                 # save_period=1,
#                 # optimizer="AdamW",
#                 # lr0=0.001,
#                 # resume=True,
#                 project="runs/Test_VisDrone",
#                 name=f"VisDrone_DET_{name}_e300bs8sz640_TopK5"
#                 )

# model_name = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolo11n","yolo11s", "yolo11m", "yolo11l"]

# for name in model_name:
#     model = HCSYOLO(f"{name}.yaml")  # build a new model from scratch
#     # Use the model
#     model.train(data="VisDrone.yaml",
#                 epochs=300,
#                 batch=8,
#                 imgsz=640,
#                 workers=8,
#                 # save_period=1,
#                 # optimizer="AdamW",
#                 # lr0=0.001,
#                 # resume=True,
#                 project="runs/Test_VisDrone",
#                 name=f"VisDrone_DET_{name}_e300bs8sz640_NewTAL"
#                 )
    
# model_name = ["yolov8_abl_1", "yolov8_abl_2", "yolov8_abl_3"]

# for name in model_name:
#     model = YOLO(f"{name}.yaml")  # build a new model from scratch
#     # Use the model
#     model.train(data="VisDrone.yaml",
#                 epochs=300,
#                 batch=8,
#                 imgsz=640,
#                 workers=8,
#                 # save_period=1,
#                 # optimizer="AdamW",
#                 # lr0=0.001,
#                 # resume=True,
#                 project="runs/Ablation_VisDrone",
#                 name=f"VisDrone_DET_{name}_e300bs8sz640"
#                 )

# model = HCSYOLO("yolov8_abl_11.yaml")  # build a new model from scratch
# # Use the model
# model.train(data="VisDrone.yaml",
#             epochs=300,
#             batch=8,
#             imgsz=640,
#             workers=8,
#             # save_period=1,
#             # resume=True,
#             project="runs/Ablation_VisDrone",
#             name="VisDrone_DET_yolov8_abl_11_SAL_e300bs8sz640")

# model = YOLO("ultralytics/cfg/models/v8/yolov8_abl_2_2.yaml")  # build a new model from scratch
# # Use the model
# model.train(data="VisDrone.yaml",
#             epochs=300,
#             batch=8,
#             imgsz=640,
#             workers=8,
#             # save_period=1,
#             # resume=True,
#             project="runs/Ablation_VisDrone",
#             name="VisDrone_DET_yolov8_abl_2_1_e300bs8sz640")

# model = HCSYOLO("ultralytics/cfg/models/v8/SALWANetn.yaml")  # build a new model from scratch
# # Use the model
# model.train(data="VisDrone.yaml",
#             epochs=300,
#             batch=8,
#             imgsz=640,
#             workers=8,
#             # save_period=1,
#             # resume=True,
#             project="runs/Ablation_VisDrone",
#             name="VisDrone_DET_SALWANetn_e300bs8sz640")

# model = YOLO("yolo11s.yaml")  # build a new model from scratch
# # Use the model
# model.train(data="aitod.yaml",
#             epochs=300,
#             batch=8,
#             imgsz=800,
#             workers=8,
#             # save_period=1,
#             # resume=True,
#             project="runs/Aitod",
#             name="AITod_yolo11s_e300bs8sz800")

# model = HCSYOLO("SALWANetn.yaml")  # build a new model from scratch
# # Use the model
# model.train(data="VisDrone.yaml",
#             epochs=300,
#             batch=8,
#             imgsz=640,
#             workers=8,
#             # save_period=1,
#             # resume=True,
#             project="runs/Test_VisDrone",
#             name="VisDrone_DET_SALWANetn_e300bs8sz640_NEWCSCAF")



model_name = ["SALWANetn"]

for name in model_name:
    model = SALWANet(f"{name}.yaml")  # build a new model from scratch
    # model = YOLO(f"{name}.yaml")  # build a new model from scratch  
    # Use the model
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
    
# model_name = ["SALWANet_abl4", "SALWANet_abl2", "SALWANet_abl3"]
# for name in model_name:
#     model = HCSYOLO(f"{name}.yaml")  # build a new model from scratch
#     # Use the model
#     model.train(data="VisDrone.yaml",
#                 epochs=300,
#                 batch=8,
#                 imgsz=640,
#                 workers=8,
#                 # save_period=1,
#                 # optimizer="AdamW",
#                 # lr0=0.001,
#                 # resume=True,
#                 project="runs/Ablation_VisDrone",
#                 name=f"VisDrone_DET_{name}_e300bs8sz640"
#                 )

# model_name = ["SALWANetn_11_woCSCAF", "SALWANets_11_woCSCAF", "SALWANetm_11_woCSCAF", "SALWANetl_11_woCSCAF"]
# for name in model_name:
#     model = HCSYOLO(f"{name}.yaml")  # build a new model from scratch
#     # Use the model
#     model.train(data="VisDrone.yaml",
#                 epochs=300,
#                 batch=8,
#                 imgsz=640,
#                 workers=8,
#                 # save_period=1,
#                 # optimizer="AdamW",
#                 # lr0=0.001,
#                 # resume=True,
#                 project="runs/VisDrone_11",
#                 name=f"VisDrone_DET_{name}_e300bs8sz640"
#                 )