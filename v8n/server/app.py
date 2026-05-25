"""
app.py — FastAPI server cho Rice Disease Detection
Chạy: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import base64
import time
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import uvicorn

# ── Config ────────────────────────────────────────────────────────────
# YOLO_PATH         = "./models/yolov8n_rice_leaf.pt"
# MOBILENET_PATH    = "./models/rice_disease_classifier.h5"

YOLO_PATH = "../models/yolo/rice_leaf_detect/weights/best.pt"
MOBILENET_PATH = "../models/rice_disease_classifier.h5"

CONF_THRESHOLD    = 0.4   # YOLO detection threshold
# MIN_YOLO_CONF     = 0.5   # Minimum YOLO confidence để accept
MIN_MOBILENET_CONF= 0.75  # Minimum MobileNet confidence để accept
PADDING           = 10

CLASS_NAMES = [
    "bacterial_blight",
    "barrow_brown_leaf_spot",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "leaf_smut",
    # "neck_blast",
    "rice_hispa",
    "sheath_blight",
    "tungro",
]

CLASS_INFO = {
    "bacterial_blight"      : {"vi": "Bạc lá vi khuẩn",     "severity": "Cao",       "color": "#e74c3c"},
    "barrow_brown_leaf_spot": {"vi": "Đốm nâu Barrow",      "severity": "Trung bình", "color": "#e67e22"},
    "brown_spot"            : {"vi": "Đốm nâu",              "severity": "Trung bình", "color": "#e67e22"},
    "healthy"               : {"vi": "Lá khỏe mạnh",         "severity": "Không",     "color": "#27ae60"},
    "leaf_blast"            : {"vi": "Đạo ôn lá",            "severity": "Cao",       "color": "#e74c3c"},
    "leaf_scald"            : {"vi": "Cháy bìa lá",          "severity": "Trung bình", "color": "#e67e22"},
    "leaf_smut"             : {"vi": "Nấm lá",               "severity": "Thấp",      "color": "#f1c40f"},
    # "neck_blast"            : {"vi": "Đạo ôn cổ bông",       "severity": "Rất cao",   "color": "#c0392b"},
    "rice_hispa"            : {"vi": "Bọ trĩ lúa (Hispa)",  "severity": "Trung bình", "color": "#e67e22"},
    "sheath_blight"         : {"vi": "Khô vằn",              "severity": "Cao",       "color": "#e74c3c"},
    "tungro"                : {"vi": "Vàng lùn Tungro",      "severity": "Rất cao",   "color": "#c0392b"},
}

# ── Load models ───────────────────────────────────────────────────────
print("Loading models...")
yolo      = YOLO(YOLO_PATH)
mobilenet = tf.keras.models.load_model(MOBILENET_PATH)
print("✅ Models loaded")

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="Rice Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def img_to_base64(img_bgr: np.ndarray) -> str:
    """Convert BGR numpy array → base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def draw_bbox_on_image(img_bgr: np.ndarray, box_xyxy: tuple, yolo_conf: float) -> np.ndarray:
    """Vẽ bounding box lên ảnh gốc."""
    img_out = img_bgr.copy()
    x1, y1, x2, y2 = box_xyxy
    cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 136), 3)
    label = f"rice_leaf {yolo_conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img_out, (x1, max(y1-th-10, 0)), (x1+tw+6, y1), (0, 170, 85), -1)
    cv2.putText(img_out, label, (x1+3, max(y1-5, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img_out


@app.get("/")
def root():
    return {"message": "Rice Disease Detection API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    print("filename:", file.filename)
    print("content_type:", file.content_type)

    # ── Validate file ─────────────────────────────────────────────────
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="File phải là ảnh")
    if not (
        file.content_type.startswith("image/")
        or file.filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ):
        raise HTTPException(status_code=400, detail="File phải là ảnh")

    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh")

    h_orig, w_orig = img.shape[:2]
    t_start = time.perf_counter()

    # ── Step 1: YOLO detect ───────────────────────────────────────────
    t0       = time.perf_counter()
    yolo_res = yolo.predict(img, conf=CONF_THRESHOLD, verbose=False)
    t_yolo   = (time.perf_counter() - t0) * 1000

    boxes         = yolo_res[0].boxes
    yolo_conf     = 0.0
    yolo_detected = False
    box_xyxy      = None
    crop          = None

    # if boxes is not None and len(boxes) > 0:
    #     xyxy  = boxes.xyxy.cpu().numpy()
    #     confs = boxes.conf.cpu().numpy()
    #     areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    #     best  = np.argmax(areas)
    #     yolo_conf = float(confs[best])

    #     if yolo_conf >= MIN_YOLO_CONF:
    #         x1, y1, x2, y2 = xyxy[best].astype(int)
    #         x1 = max(0, x1 - PADDING);      y1 = max(0, y1 - PADDING)
    #         x2 = min(w_orig, x2 + PADDING); y2 = min(h_orig, y2 + PADDING)
    #         box_xyxy      = (x1, y1, x2, y2)
    #         yolo_detected = True
    #         crop          = img[y1:y2, x1:x2]

    if boxes is not None and len(boxes) > 0:
        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best  = np.argmax(areas)

        x1, y1, x2, y2 = xyxy[best].astype(int)

        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w_orig, x2 + PADDING)
        y2 = min(h_orig, y2 + PADDING)

        box_xyxy = (x1, y1, x2, y2)

        yolo_conf = float(confs[best])

        yolo_detected = True

        crop = img[y1:y2, x1:x2]

    else:
        crop = img.copy()

    # if boxes is not None and len(boxes) > 0:
    
    #     xyxy  = boxes.xyxy.cpu().numpy()
    #     confs = boxes.conf.cpu().numpy()

    #     areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    #     best  = np.argmax(areas)

    #     yolo_conf = float(confs[best])

    #     # ❌ Detect yếu -> UNKNOWN
    #     if yolo_conf < MIN_YOLO_CONF:

    #         return JSONResponse(content={
    #             "success": False,
    #             "reason": f"YOLO confidence quá thấp ({yolo_conf*100:.1f}%)",
    #             "yolo_detected": False,
    #             "yolo_conf": float(round(yolo_conf * 100, 1)),
    #             "original_image": img_to_base64(img),
    #         })

    #     # ✅ Detect đủ mạnh
    #     x1, y1, x2, y2 = xyxy[best].astype(int)

    #     x1 = max(0, x1 - PADDING)
    #     y1 = max(0, y1 - PADDING)
    #     x2 = min(w_orig, x2 + PADDING)
    #     y2 = min(h_orig, y2 + PADDING)

    #     box_xyxy = (int(x1), int(y1), int(x2), int(y2))

    #     yolo_detected = True

    #     crop = img[y1:y2, x1:x2]

    # else:

    #     # ❌ Không detect gì
    #     return JSONResponse(content={
    #         "success": False,
    #         "reason": "Không tìm thấy lá lúa trong ảnh",
    #         "yolo_detected": False,
    #         "yolo_conf": 0.0,
    #         "original_image": img_to_base64(img),
    #     })

    # ── Từ chối nếu YOLO không detect ────────────────────────────────
    # if not yolo_detected:
    #     reason = (
    #         f"YOLO detect không rõ (conf={yolo_conf*100:.0f}% < {MIN_YOLO_CONF*100:.0f}%)"
    #         if yolo_conf > 0
    #         else "Không tìm thấy lá lúa trong ảnh"
    #     )
    #     return JSONResponse(content={
    #         "success"      : False,
    #         "reason"       : reason,
    #         "original_image": img_to_base64(img),
    #         "yolo_detected": False,
    #         "yolo_conf"    : round(yolo_conf * 100, 1),
    #     })

    if not yolo_detected:
        crop = img.copy()

    # ── Step 2: Resize crop → 224×224 ─────────────────────────────────
    crop_224 = cv2.resize(crop, (224, 224))

    # ── Step 3: MobileNet classify ────────────────────────────────────
    t1      = time.perf_counter()
    img_rgb = cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    probs   = mobilenet.predict(np.expand_dims(img_rgb, 0), verbose=0)[0]
    t_cls   = (time.perf_counter() - t1) * 1000
    t_total = (time.perf_counter() - t_start) * 1000

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf  = float(probs[pred_idx])

    # ── Từ chối nếu MobileNet không chắc ─────────────────────────────
    if pred_conf < MIN_MOBILENET_CONF:
        return JSONResponse(content={
            "success"        : False,
            "reason"         : f"Độ tin cậy thấp ({pred_conf*100:.1f}% < {MIN_MOBILENET_CONF*100:.0f}%) — không thể xác định bệnh",
            "original_image" : img_to_base64(img),
            # "yolo_detected"  : True,
            "yolo_detected": yolo_detected,
            "yolo_conf"      : round(yolo_conf * 100, 1),
            "crop_image"     : img_to_base64(crop_224),
        })

    # ── Build response ────────────────────────────────────────────────
    info        = CLASS_INFO[pred_class]
    # img_bbox    = draw_bbox_on_image(img, box_xyxy, yolo_conf)
    if box_xyxy is not None:
        img_bbox = draw_bbox_on_image(img, box_xyxy, yolo_conf)
    else:
        img_bbox = img.copy()

    # Top-5 probabilities
    top5_idx   = np.argsort(probs)[::-1][:5]
    # top5       = [
    #     {"class": CLASS_NAMES[i], "probability": round(float(probs[i]) * 100, 2)}
    #     for i in top5_idx
    # ]
    top5 = [
    {
        "class": str(CLASS_NAMES[i]),
        "probability": float(round(float(probs[i]) * 100, 2))
    }
    for i in top5_idx
]

    return JSONResponse(content={
        "success"        : True,
        "class_en"       : pred_class,
        "class_vi"       : info["vi"],
        "confidence"     : round(pred_conf * 100, 1),
        "severity"       : info["severity"],
        "color"          : info["color"],
        "yolo_detected"  : True,
        "yolo_conf"      : round(yolo_conf * 100, 1),
        # "box_xyxy"       : list(box_xyxy),
        "box_xyxy": [int(v) for v in box_xyxy] if box_xyxy else None,
        "top5"           : top5,
        "original_image" : img_to_base64(img),       # ảnh gốc không có bbox
        "bbox_image"     : img_to_base64(img_bbox),  # ảnh gốc có bbox
        "crop_image"     : img_to_base64(crop_224),  # crop 224×224
        # "latency": {
        #     "yolo_ms"  : round(t_yolo, 1),
        #     "cls_ms"   : round(t_cls, 1),
        #     "total_ms" : round(t_total, 1),
        # }
        "latency": {
            "yolo_ms"  : float(round(t_yolo, 1)),
            "cls_ms"   : float(round(t_cls, 1)),
            "total_ms" : float(round(t_total, 1)),
        }
    })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)