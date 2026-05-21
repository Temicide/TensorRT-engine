import logging
import cv2

# ──────────────────────────── COCO class names ────────────────────────────────
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multicam")

# ─────────────────────────── Draw helpers ─────────────────────────────────────
COLORS = [(0,255,0),(255,100,0),(0,100,255),(255,255,0),(0,255,255),
          (255,0,255),(128,255,0),(0,128,255),(255,128,0),(128,0,255)]

TRACK_COLORS = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),
    (255,0,255),(128,255,0),(255,128,0),(0,128,255),(128,0,255),
    (255,128,128),(128,255,128),(128,128,255),(255,255,128),(128,255,255),
    (255,128,255),(200,100,50),(50,200,100),(100,50,200),(200,50,100),
]

def _track_color(track_id: int):
    value = int(track_id) * 2654435761 % 255
    return (
        int((value * 3 + 80) % 255),
        int((value * 7 + 120) % 255),
        int((value * 11 + 160) % 255),
    )

def draw_frame(frame, detections, fps, cam_id):
    img = frame.copy()
    for d in detections:
        x1,y1,x2,y2 = [int(v) for v in d["bbox_xyxy"]]
        track_id = d.get("track_id", -1)
        if track_id is not None and track_id >= 0:
            color = _track_color(track_id)
            label = f"ID:{track_id} {d['class_name']} {d['confidence']:.2f}"
        else:
            cid = d["class_id"] % len(COLORS)
            color = COLORS[cid]
            label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    # FPS + cam id overlay
    overlay = f"[{cam_id}] {fps:.1f} FPS"
    cv2.putText(img, overlay, (11,31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, overlay, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)
    return img