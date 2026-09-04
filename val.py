import json
from pathlib import Path
from ultralytics import YOLO, SALWANet


# ==================== 配置 ====================
# predictions.json 由 Ultralytics 的 model.val(save_json=True) 生成。
# 如需重新生成，把下面两行取消注释并运行：
model = SALWANet("runs/0830VisDrone/VisDrone_DET_yolov8_abl_4_e300bs8sz640/weights/best.pt")
res = model.val(data="VisDrone.yaml", batch=8, imgsz=640, save_json=True)

# PREDICTIONS_PATH = Path("runs/detect/val2/predictions.json")
# ANNOTATIONS_PATH = Path("annotations_VisDrone_val_filtered.json")
# METRICS_PATH = Path("runs/detect/val2/metrics.json")


# # 优先使用 faster-coco-eval，没有安装时退回 pycocotools
# try:
#     from faster_coco_eval import COCO, COCOeval_faster
# except ImportError:
#     from pycocotools.coco import COCO
#     from pycocotools.cocoeval import COCOeval as COCOeval_faster


# def compute_ap(precision, iou_idx, area_idx):
#     """从 precision 中取某个 IoU、某个面积范围的 AP。"""
#     scores = precision[iou_idx, :, :, area_idx, -1]  # -1 表示 maxDets=100
#     scores = scores[scores > -1]
#     return float(scores.mean()) if scores.size else -1.0


# # 加载 COCO 标注和 Ultralytics 预测结果
# anno = COCO(str(ANNOTATIONS_PATH))
# with PREDICTIONS_PATH.open("r", encoding="utf-8") as f:
#     predictions = json.load(f)

# # Ultralytics 的 image_id 是文件名去掉后缀的字符串，
# # COCO 标注里的 image_id 是整数，这里做一次映射。
# image_id_map = {Path(img["file_name"]).stem: img["id"] for img in anno.dataset["images"]}
# for pred in predictions:
#     if isinstance(pred["image_id"], str):
#         pred["image_id"] = image_id_map.get(pred["image_id"], pred["image_id"])


# # 加载预测并计算 COCO bbox 指标
# pred = anno.loadRes(predictions)
# val = COCOeval_faster(anno, pred, iouType="bbox")
# if hasattr(val, "print_function"):
#     val.print_function = lambda *args, **kwargs: None  # 隐藏 faster 的逐行输出
# val.evaluate()
# val.accumulate()
# val.summarize()

# # 注意：不要手动设置 val.params.iouThrs = [0.5]，
# # faster-coco-eval 的 summarize() 会因此报 0d array 错误。
# # 保持默认 IoU 网格，再从 precision 里取 IoU=0.50 的 AP。
# precision = val.eval["precision"]
# stats = val.stats  # stats: 0=AP_all, 1=AP50, 2=AP75, 3=AP_small, 4=AP_medium, 5=AP_large

# summary = {
#     "AP50": round(compute_ap(precision, 0, 0) * 100, 2),
#     "AP50_small": round(compute_ap(precision, 0, 1) * 100, 2),
#     "AP50_medium": round(compute_ap(precision, 0, 2) * 100, 2),
#     "AP50_large": round(compute_ap(precision, 0, 3) * 100, 2),
#     "AP_small_50_95": round(stats[3] * 100, 2),
# }

# # 保存整理后的指标
# METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
# with METRICS_PATH.open("w", encoding="utf-8") as f:
#     json.dump(summary, f, indent=2, ensure_ascii=False)

# # 输出整齐的指标表格
# print("=" * 46)
# print("VisDrone val / val34 predictions")
# print("-" * 46)
# for name, value in summary.items():
#     print(f"{name:<22}{value:>8.2f}%")
# print("=" * 46)
# print(f"Saved: {METRICS_PATH}")
