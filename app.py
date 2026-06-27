import math
import json
import os
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

st.set_page_config(page_title="ระบบคัดกรองภาวะกระดูกสันหลังคดเบื้องต้น", layout="centered")

BASELINE_FILE = "baseline.json"
MODEL_FILE = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

def download_model():
    if not os.path.exists(MODEL_FILE):
        st.info("กำลังดาวน์โหลดโมเดล กรุณารอสักครู่...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

download_model()

@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.PoseLandmarker.create_from_options(options)

detector = load_detector()


def analyze_standing(image_bgr):
    h, w, _ = image_bgr.shape
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = detector.detect(mp_image)

    if not result.pose_landmarks:
        return None, None

    landmarks = result.pose_landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    dx_s = (right_shoulder.x - left_shoulder.x) * w
    dy_s = (right_shoulder.y - left_shoulder.y) * h
    shoulder_slope = abs(dy_s / dx_s) if dx_s != 0 else 0

    dx_h = (right_hip.x - left_hip.x) * w
    dy_h = (right_hip.y - left_hip.y) * h
    hip_slope = abs(dy_h / dx_h) if dx_h != 0 else 0

    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2 * w
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * h
    mid_hip_x = (left_hip.x + right_hip.x) / 2 * w
    mid_hip_y = (left_hip.y + right_hip.y) / 2 * h

    dx_trunk = mid_shoulder_x - mid_hip_x
    dy_trunk = mid_shoulder_y - mid_hip_y
    trunk_tilt_angle = math.degrees(math.atan2(abs(dx_trunk), abs(dy_trunk))) if dy_trunk != 0 else 90.0

    shoulder_tilt_dir = "right_up" if dy_s < 0 else "left_up" if dy_s > 0 else "level"
    hip_tilt_dir = "right_up" if dy_h < 0 else "left_up" if dy_h > 0 else "level"
    same_direction = (shoulder_tilt_dir == hip_tilt_dir) and shoulder_tilt_dir != "level"

    annotated = image_bgr.copy()
    for lm in [left_shoulder, right_shoulder, left_hip, right_hip]:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (x, y), 8, (0, 0, 255), -1)

    cv2.line(annotated,
             (int(left_shoulder.x * w), int(left_shoulder.y * h)),
             (int(right_shoulder.x * w), int(right_shoulder.y * h)),
             (255, 200, 0), 3)
    cv2.line(annotated,
             (int(left_hip.x * w), int(left_hip.y * h)),
             (int(right_hip.x * w), int(right_hip.y * h)),
             (255, 200, 0), 3)
    cv2.line(annotated,
             (int(mid_shoulder_x), int(mid_shoulder_y)),
             (int(mid_hip_x), int(mid_hip_y)),
             (0, 255, 255), 3)

    return {
        "shoulder_slope": shoulder_slope,
        "hip_slope": hip_slope,
        "trunk_tilt_angle": trunk_tilt_angle,
        "shoulder_tilt_dir": shoulder_tilt_dir,
        "hip_tilt_dir": hip_tilt_dir,
        "same_direction": same_direction,
    }, annotated


def get_risk_level_default(value, low_th, high_th):
    if value < low_th:
        return "ต่ำ (ปกติ)", "green"
    elif value < high_th:
        return "ปานกลาง", "orange"
    else:
        return "สูง", "red"


def load_baseline():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_risk_level_baseline(value, mean, sd):
    if sd == 0:
        sd = 1e-6
    diff = abs(value - mean)
    if diff <= sd:
        return "ต่ำ (ปกติ)", "green"
    elif diff <= 2 * sd:
        return "ปานกลาง", "orange"
    else:
        return "สูง", "red"


RECOMMENDATIONS = {
    "ต่ำ (ปกติ)": "ท่าทางของคุณอยู่ในเกณฑ์ปกติ ควรรักษาท่าทางที่ดีต่อไป และออกกำลังกายยืดเหยียดเป็นประจำ",
    "ปานกลาง": "พบความเอียงเล็กน้อย แนะนำให้สังเกตท่าทางตนเองในชีวิตประจำวัน และฝึกบริหารกล้ามเนื้อหลังและไหล่ หากมีอาการผิดปกติควรพบแพทย์",
    "สูง": "พบความเอียงค่อนข้างมาก แนะนำให้พบแพทย์หรือนักกายภาพบำบัดเพื่อตรวจวินิจฉัยเพิ่มเติม ระบบนี้เป็นเพียงการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์"
}

SLOPE_LOW, SLOPE_HIGH = 0.05, 0.15
baseline = load_baseline()

st.sidebar.title("เมนู")
mode = st.sidebar.radio("เลือกโหมด", [
    "ประเมินผล",
    "Calibration (สร้างเกณฑ์จากกลุ่มปกติ)"
])

if mode == "Calibration (สร้างเกณฑ์จากกลุ่มปกติ)":
    st.title("📊 Calibration: สร้างเกณฑ์จากกลุ่มคนหลังตรง")
    st.write("อัปโหลดภาพถ่ายด้านหลัง (ท่ายืนตรง) ของกลุ่มคนที่หลังตรง/ปกติ หลายๆ รูป")

    uploaded_files = st.file_uploader(
        "เลือกภาพถ่าย (หลายรูปได้)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        rows = []
        for f in uploaded_files:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result, _ = analyze_standing(image_bgr)

            if result is None:
                st.warning(f"ไม่พบร่างกายในภาพ: {f.name} (ข้ามรูปนี้)")
                continue

            rows.append({
                "filename": f.name,
                "shoulder_slope": round(result["shoulder_slope"], 4),
                "hip_slope": round(result["hip_slope"], 4),
                "trunk_tilt_angle": round(result["trunk_tilt_angle"], 2),
            })

        if rows:
            st.subheader("ผลลัพธ์รายบุคคล")
            st.table(rows)

            shoulder_vals = [r["shoulder_slope"] for r in rows]
            hip_vals = [r["hip_slope"] for r in rows]
            trunk_vals = [r["trunk_tilt_angle"] for r in rows]

            stats = {
                "n": len(rows),
                "shoulder_slope": {"mean": float(np.mean(shoulder_vals)), "sd": float(np.std(shoulder_vals))},
                "hip_slope": {"mean": float(np.mean(hip_vals)), "sd": float(np.std(hip_vals))},
                "trunk_tilt_angle": {"mean": float(np.mean(trunk_vals)), "sd": float(np.std(trunk_vals))},
            }

            st.subheader("สถิติสรุป (Mean ± SD)")
            st.write(f"จำนวนตัวอย่าง: {stats['n']} คน")
            st.write(f"Shoulder Slope: {stats['shoulder_slope']['mean']:.4f} ± {stats['shoulder_slope']['sd']:.4f}")
            st.write(f"Hip Slope: {stats['hip_slope']['mean']:.4f} ± {stats['hip_slope']['sd']:.4f}")
            st.write(f"Trunk Tilt Angle: {stats['trunk_tilt_angle']['mean']:.2f}° ± {stats['trunk_tilt_angle']['sd']:.2f}°")

            if st.button("💾 บันทึกเป็นเกณฑ์ (baseline.json)"):
                with open(BASELINE_FILE, "w", encoding="utf-8") as fp:
                    json.dump(stats, fp, ensure_ascii=False, indent=2)
                st.success(f"บันทึกเกณฑ์เรียบร้อย ({stats['n']} ตัวอย่าง)")

else:
    st.title("🦴 ระบบคัดกรองภาวะกระดูกสันหลังคดเบื้องต้น")
    st.write("อัปโหลดภาพถ่ายด้านหลัง (ท่ายืนตรง) เพื่อประเมินความเสี่ยงเบื้องต้น")

    if baseline:
        st.caption(f"✅ ใช้เกณฑ์จากกลุ่มตัวอย่างปกติ ({baseline['n']} คน)")
    else:
        st.caption("⚠️ ยังไม่มีเกณฑ์จากกลุ่มตัวอย่าง — ใช้ค่าเริ่มต้น")

    uploaded_file = st.file_uploader("เลือกภาพถ่าย (ท่ายืนตรง)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result, annotated = analyze_standing(image_bgr)

        if result is None:
            st.error("ไม่พบร่างกายในภาพ กรุณาลองใหม่ด้วยภาพที่เห็นไหล่และสะโพกชัดเจน")
        else:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="เส้นฟ้า = ไหล่/สะโพก | เส้นเหลือง = แนวกลางลำตัว", use_column_width=True)

            if baseline:
                shoulder_risk, shoulder_color = get_risk_level_baseline(
                    result["shoulder_slope"], baseline["shoulder_slope"]["mean"], baseline["shoulder_slope"]["sd"])
                hip_risk, hip_color = get_risk_level_baseline(
                    result["hip_slope"], baseline["hip_slope"]["mean"], baseline["hip_slope"]["sd"])
            else:
                shoulder_risk, shoulder_color = get_risk_level_default(result["shoulder_slope"], SLOPE_LOW, SLOPE_HIGH)
                hip_risk, hip_color = get_risk_level_default(result["hip_slope"], SLOPE_LOW, SLOPE_HIGH)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Shoulder Slope", f"{result['shoulder_slope']:.4f}")
                st.markdown(f"ความเสี่ยง: :{shoulder_color}[{shoulder_risk}]")
            with col2:
                st.metric("Hip Slope", f"{result['hip_slope']:.4f}")
                st.markdown(f"ความเสี่ยง: :{hip_color}[{hip_risk}]")
            with col3:
                st.metric("Trunk Tilt Angle", f"{result['trunk_tilt_angle']:.2f}°")

            if result["same_direction"]:
                curve_note = "ไหล่และสะโพกเอียงไปทาง**เดียวกัน** → ลำตัวเอียงไปทั้งแท่ง (C-curve)"
            elif result["shoulder_tilt_dir"] == "level" or result["hip_tilt_dir"] == "level":
                curve_note = "ไม่พบการเอียงที่ชัดเจน"
            else:
                curve_note = "ไหล่และสะโพกเอียง**คนละทาง** → อาจมีการบิดของลำตัวสองช่วง (เข้าข่าย S-curve เบื้องต้น)"

            st.write(f"**ลักษณะแนวลำตัว:** {curve_note}")

            overall_risk = max(shoulder_risk, hip_risk, key=lambda r: ["ต่ำ (ปกติ)", "ปานกลาง", "สูง"].index(r))

            if not result["same_direction"] and result["shoulder_tilt_dir"] != "level" and result["hip_tilt_dir"] != "level":
                if overall_risk == "ต่ำ (ปกติ)":
                    overall_risk = "ปานกลาง"

            st.subheader(f"ผลการประเมิน: {overall_risk}")
            st.info(RECOMMENDATIONS[overall_risk])
            st.caption("⚠️ ระบบนี้เป็นเครื่องมือคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้")