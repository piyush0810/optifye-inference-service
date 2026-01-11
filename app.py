from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import cv2
import numpy as np
import base64
from typing import List, Tuple

app = FastAPI(title="YOLOv8n ONNX Batch Inference")

# ────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────
MODEL_PATH = "/model/yolov8n.onnx"  # ← change if needed

try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    INPUT_NAME = session.get_inputs()[0].name
    print(f"Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise RuntimeError(f"Cannot load ONNX model: {e}")

INPUT_SIZE = (640, 640)


class FrameItem(BaseModel):
    frame_id: int
    timestamp: float
    image_base64: str


class BatchRequest(BaseModel):
    frames: List[FrameItem]


class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class FrameResult(BaseModel):
    frame_id: int
    detections: List[Detection]


class InferenceResponse(BaseModel):
    results: List[FrameResult]


def preprocess(
    base64_str: str
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float]]:
    """Letterbox + normalize → ready for YOLOv8 ONNX"""
    try:
        img_bytes = base64.b64decode(base64_str)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode base64 image")

        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox
        r = min(INPUT_SIZE[0] / orig_h, INPUT_SIZE[1] / orig_w)
        new_h, new_w = int(orig_h * r), int(orig_w * r)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = INPUT_SIZE[0] - new_h
        pad_w = INPUT_SIZE[1] - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        canvas = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

        # Normalize & prepare
        canvas = canvas.astype(np.float32) / 255.0
        canvas = canvas.transpose(2, 0, 1)      # HWC → CHW
        canvas = np.expand_dims(canvas, 0)      # add batch → (1,3,640,640)

        meta = (r, pad_left, pad_top, float(orig_h), float(orig_w))

        return canvas, meta

    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")


def postprocess(
    raw_output: np.ndarray,
    meta: Tuple[float, float, float, float, float],
    conf_thres: float = 0.001,   # ← start VERY low to see if anything appears
    iou_thres: float = 0.45,
    use_sigmoid: bool = False    # ← most important: False for official yolov8*.onnx
) -> List[Detection]:
    scale, pad_left, pad_top, orig_h, orig_w = meta

    # Standard official YOLOv8 ONNX layout
    preds = raw_output[0].transpose(1, 0)  # (8400, 84)

    print(f"Output shape after transpose: {preds.shape}")
    print(f"Max raw objectness (col 4): {preds[:,4].max():.4f}")

    # NO sigmoid – official export already has it
    obj_scores = preds[:, 4]

    boxes = []
    for i in range(len(preds)):
        obj_conf = obj_scores[i]
        if obj_conf < conf_thres:
            continue

        cx, cy, w, h = preds[i, :4]
        cls_scores = preds[i, 5:]                 # (80,)
        cls_id = int(np.argmax(cls_scores))
        cls_conf = float(cls_scores[cls_id])

        score = obj_conf * cls_conf               # ← combined confidence

        if score < conf_thres:
            continue

        # inverse letterbox (this part was already correct)
        x1 = max(0.0, (cx - w/2 - pad_left) / scale)
        y1 = max(0.0, (cy - h/2 - pad_top) / scale)
        x2 = min(orig_w, (cx + w/2 - pad_left) / scale)
        y2 = min(orig_h, (cy + h/2 - pad_top) / scale)

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2, score, cls_id])

    print(f"Raw candidates (conf_thres={conf_thres}): {len(boxes)}")

    if not boxes:
        return []

    boxes = np.array(boxes, dtype=np.float32)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes[:, :4].tolist(),
        boxes[:, 4].tolist(),
        conf_thres,
        iou_thres
    )

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2, conf, cls_id = boxes[i]
            detections.append(Detection(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                confidence=float(conf),
                class_id=int(cls_id)
            ))

    print(f"Final detections: {len(detections)}")
    return detections

@app.post("/infer", response_model=InferenceResponse)
async def infer(batch: BatchRequest):
    try:
        results = []

        for item in batch.frames:
            input_tensor, meta = preprocess(item.image_base64)

            outputs = session.run(None, {INPUT_NAME: input_tensor})

            # Try both sigmoid modes if needed
            detections = postprocess(
                raw_output=outputs[0],
                meta=meta,
                conf_thres=0.001,
                iou_thres=0.45,
                use_sigmoid=False   # ← change to False if max objectness stays very low
            )

            results.append(FrameResult(
                frame_id=item.frame_id,
                detections=detections
            ))

        return InferenceResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)