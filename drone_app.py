"""드론 기종·촬영조건 분류 데모 (OpenVINO + Streamlit).

파이프라인: 이미지 업로드 → YOLO 드론 검출(OpenVINO) → 드론 크롭 → 분류기(OpenVINO) → 기종 + 조건

실행하려면 이 파일과 같은 폴더에 `drone_weights/` 가 있어야 한다:
    drone_weights/
      ├── mobilenet_v2.xml (+ .bin)
      ├── efficientnet_v2_s.xml (+ .bin)
      └── detector_openvino/   (YOLOv8n OpenVINO export 폴더: best.xml/.bin/metadata.yaml)
그리고 requirements 에 `ultralytics` 가 필요하다.

로컬 실행:  streamlit run drone_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
W = APP_DIR / "drone_weights"
DETECTOR_DIR = W / "detector_openvino"
CLASSIFIER_XML = {
    "MobileNetV2 (권장, 0.894)": W / "mobilenet_v2.xml",
    "EfficientNetV2-S (0.862)": W / "efficientnet_v2_s.xml",
}

# 학습 시 class_to_idx 순서와 동일해야 함 (기종 3 × 조건 3)
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


@st.cache_resource(show_spinner="OpenVINO 검출기 로드 중...")
def load_detector() -> YOLO:
    return YOLO(str(DETECTOR_DIR))


@st.cache_resource(show_spinner="OpenVINO 분류기 로드 중...")
def load_classifier(xml_path: str):
    core = ov.Core()
    return core.compile_model(core.read_model(xml_path), "CPU")


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


st.set_page_config(page_title="드론 기종·조건 분류", page_icon="🚁", layout="wide")
st.title("🚁 드론 기종 · 촬영조건 분류 (OpenVINO)")
st.caption("이미지 업로드 → YOLO 드론 검출 → 크롭 → 분류기(OpenVINO)로 기종 + 조건 예측")

if not DETECTOR_DIR.exists() or not any(p.exists() for p in CLASSIFIER_XML.values()):
    st.error("모델 파일이 없습니다. 이 파일 옆에 `drone_weights/`(분류기 xml/bin + detector_openvino/)를 넣어주세요.")
    st.stop()

with st.sidebar:
    st.header("설정")
    model_label = st.selectbox("분류 모델", list(CLASSIFIER_XML.keys()))
    conf_thr = st.slider("드론 검출 신뢰도 임계값", 0.05, 0.9, 0.15, 0.05)

detector = load_detector()
classifier = load_classifier(str(CLASSIFIER_XML[model_label]))

uploaded = st.file_uploader("드론 이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is None:
    st.info("드론 이미지를 업로드하면 검출 → 크롭 → 분류 결과가 표시됩니다.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
W_, H_ = image.size

det_res = detector.predict(image, imgsz=1280, conf=conf_thr, verbose=False)[0]
boxes = det_res.boxes
if boxes is not None and len(boxes) > 0:
    i = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
    det_conf = float(boxes.conf[i])
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
