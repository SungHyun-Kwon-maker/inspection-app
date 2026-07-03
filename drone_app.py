"""드론 기종·촬영조건 분류 데모 (순수 OpenVINO + Streamlit, torch 불필요).

파이프라인: 이미지 업로드 → YOLOv8 드론 검출(OpenVINO) → 드론 크롭 → 분류기(OpenVINO) → 기종 + 조건

배포/실행에 필요한 것(이 파일과 같은 폴더):
    drone_weights/
      ├── mobilenet_v2.xml (+ .bin)
      ├── efficientnet_v2_s.xml (+ .bin)
      └── detector.xml (+ .bin)          # YOLOv8n OpenVINO IR
requirements: streamlit, openvino, numpy, pillow, pandas  (torch / ultralytics 불필요)

로컬 실행:  streamlit run drone_app.py
Streamlit Cloud:  배포 시 Main file 을 drone_app.py 로 지정
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

APP_DIR = Path(__file__).resolve().parent
W = APP_DIR / "drone_weights"
DETECTOR_XML = W / "detector.xml"
CLASSIFIER_XML = {
    "MobileNetV2 (권장, 0.894)": W / "mobilenet_v2.xml",
    "EfficientNetV2-S (0.862)": W / "efficientnet_v2_s.xml",
}

# 학습 시 class_to_idx 순서와 동일 (기종 3 × 조건 3)
CLASS_NAMES = [
    "L1_LC_VTOL_sunny", "L1_LC_VTOL_snow", "L1_LC_VTOL_low-light",
    "L3_QUAD_MC_sunny", "L3_QUAD_MC_snow", "L3_QUAD_MC_low-light",
    "L1_HEXA_MC_sunny", "L1_HEXA_MC_snow", "L1_HEXA_MC_low-light",
]
DRONE_KR = {"L1_LC_VTOL": "고정익 VTOL", "L3_QUAD_MC": "쿼드콥터", "L1_HEXA_MC": "헥사콥터"}
COND_KR = {"sunny": "맑음", "snow": "설경", "low-light": "저조도"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224
DET_SIZE = 1280


@st.cache_resource(show_spinner="OpenVINO 모델 로드 중...")
def load_model(xml_path: str):
    core = ov.Core()
    return core.compile_model(core.read_model(xml_path), "CPU")


# ---------------- YOLOv8 (OpenVINO) 검출 ----------------
def letterbox(im: Image.Image, new: int = DET_SIZE, color: int = 114):
    w, h = im.size
    r = min(new / w, new / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (new, new), (color, color, color))
    left, top = (new - nw) // 2, (new - nh) // 2
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (left, top))
    return canvas, r, left, top


def detect_drone(detector, im: Image.Image, conf: float):
    """가장 신뢰도 높은 드론 박스 하나를 원본 좌표(x1,y1,x2,y2,conf)로 반환. 없으면 None."""
    canvas, r, left, top = letterbox(im)
    x = (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = detector(x)[detector.output(0)][0]  # (5, N): cx,cy,w,h,score
    preds = out.T
    scores = preds[:, 4]
    i = int(scores.argmax())
    if float(scores[i]) < conf:
        return None
    cx, cy, bw, bh = preds[i, :4]
    x1 = ((cx - bw / 2) - left) / r
    y1 = ((cy - bh / 2) - top) / r
    x2 = ((cx + bw / 2) - left) / r
    y2 = ((cy + bh / 2) - top) / r
    return [float(x1), float(y1), float(x2), float(y2), float(scores[i])]


def square_crop_box(x1, y1, x2, y2, W_, H_, margin=0.30, min_side_frac=0.06):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = max(side, min_side_frac * min(W_, H_))
    side = min(side, min(W_, H_))
    half = side / 2
    cx = min(max(cx, half), W_ - half)
    cy = min(max(cy, half), H_ - half)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def center_crop_box(W_, H_, frac=0.65):
    side = min(W_, H_) * frac
    half = side / 2
    return int(W_ / 2 - half), int(H_ / 2 - half), int(W_ / 2 + half), int(H_ / 2 + half)


# ---------------- 분류기 전처리 ----------------
def preprocess_for_classifier(crop: Image.Image) -> np.ndarray:
    img = crop.convert("RGB")
    w, h = img.size
    scale = (IMG_SIZE + 32) / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------- UI ----------------
st.set_page_config(page_title="드론 기종·조건 분류", page_icon="🚁", layout="wide")
st.title("🚁 드론 기종 · 촬영조건 분류 (OpenVINO)")
st.caption("이미지 업로드 → YOLO 드론 검출 → 크롭 → 분류기(OpenVINO)로 기종 + 조건 예측")

if not DETECTOR_XML.exists() or not any(p.exists() for p in CLASSIFIER_XML.values()):
    st.error("모델 파일이 없습니다. 이 파일 옆에 `drone_weights/`(mobilenet_v2 / efficientnet_v2_s / detector 의 .xml·.bin)를 넣어주세요.")
    st.stop()

with st.sidebar:
    st.header("설정")
    model_label = st.selectbox("분류 모델", list(CLASSIFIER_XML.keys()))
    conf_thr = st.slider("드론 검출 신뢰도 임계값", 0.05, 0.9, 0.15, 0.05)
    st.caption("검출·분류 모두 OpenVINO CPU 추론 (torch 불필요)")

detector = load_model(str(DETECTOR_XML))
classifier = load_model(str(CLASSIFIER_XML[model_label]))

uploaded = st.file_uploader("드론 이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is None:
    st.info("드론 이미지를 업로드하면 검출 → 크롭 → 분류 결과가 표시됩니다.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
W_, H_ = image.size

box = detect_drone(detector, image, conf_thr)
if box is not None:
    x1, y1, x2, y2, det_conf = box
    bx = square_crop_box(x1, y1, x2, y2, W_, H_)
    fallback = False
else:
    bx = center_crop_box(W_, H_)
    det_conf, fallback = 0.0, True

crop = image.crop(bx)
logits = classifier(preprocess_for_classifier(crop))[classifier.output(0)][0]
probs = softmax(logits)
top = int(probs.argmax())
drone, cond = CLASS_NAMES[top].rsplit("_", 1)

boxed = image.copy()
ImageDraw.Draw(boxed).rectangle(bx, outline=(255, 0, 0), width=max(3, W_ // 300))

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("① 검출 결과")
    st.image(boxed, caption="빨간 박스 = 검출된 드론" + (" (미검출→중앙 크롭)" if fallback else f" (conf={det_conf:.2f})"),
             use_container_width=True)
with col2:
    st.subheader("② 크롭 (분류기 입력)")
    st.image(crop, use_container_width=True)

st.subheader("③ 분류 결과")
c1, c2, c3 = st.columns(3)
c1.metric("기종", DRONE_KR.get(drone, drone))
c2.metric("촬영조건", COND_KR.get(cond, cond))
c3.metric("신뢰도", f"{probs[top]*100:.1f}%")
st.caption(f"예측 클래스: `{CLASS_NAMES[top]}`" + ("  ·  ⚠️ 드론 미검출로 중앙 크롭 사용" if fallback else ""))

st.markdown("**클래스별 확률 (상위 5)**")
st.bar_chart(
    pd.DataFrame({"class": CLASS_NAMES, "probability": probs})
    .sort_values("probability", ascending=False).head(5).set_index("class")
)
