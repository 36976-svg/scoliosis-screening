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

HIP_VISIBILITY_THRESHOLD = 0.5  # ต่ำกว่านี้ถือว่าสะโพกไม่ได้อยู่ในเฟรม/โมเดลไม่มั่นใจ ไม่นำมาใช้คำนวณ


def draw_dashed_line(img, y, x_left, x_right, color=(200, 200, 200), thickness=1):
    for x in range(x_left, x_right, 20):
        cv2.line(img, (x, y), (min(x + 10, x_right), y), color, thickness)


def draw_zone_label(img, label, y_mid, color_bg, x_start):
    cv2.rectangle(img, (x_start, y_mid - 14), (x_start + 130, y_mid + 14), color_bg, -1)
    cv2.rectangle(img, (x_start, y_mid - 14), (x_start + 130, y_mid + 14), (255, 255, 255), 1)
    cv2.putText(img, label, (x_start + 5, y_mid + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def estimate_neck_point(nose, left_ear, right_ear, mid_shoulder_x, mid_shoulder_y, w, h):
    """ประมาณตำแหน่ง 'โคนคอ / โหนกกระดูกคอ (C7 vertebra prominens)'
    จากแนวหู (ประมาณแนวศีรษะ) ไล่ลงมาหาแนวไหล่ ~28% ของระยะห่าง
    ซึ่งใกล้เคียงตำแหน่งโหนกที่นูนที่สุดของกระดูกสันหลังส่วนคอ"""
    mid_ear_x = (left_ear.x + right_ear.x) / 2 * w
    mid_ear_y = (left_ear.y + right_ear.y) / 2 * h
    t = 0.28
    neck_x = mid_shoulder_x + t * (mid_ear_x - mid_shoulder_x)
    neck_y = mid_shoulder_y + t * (mid_ear_y - mid_shoulder_y)
    return neck_x, neck_y


def build_spine_chain(neck_pt, mid_shoulder, mid_hip, n_intermediate=3):
    """สร้างจุดข้อต่อไล่ตามแนวกระดูกสันหลัง: คอ -> ไหล่ -> (ทรวงอก/เอวโดยประมาณ) -> สะโพก
    จุดตรงกลางไล่เรียงเป็นสัดส่วนระหว่างไหล่กับสะโพก แทนระดับกระดูกสันหลังช่วงอก/เอวโดยประมาณ"""
    points = [neck_pt, mid_shoulder]
    for i in range(1, n_intermediate + 1):
        t = i / (n_intermediate + 1)
        x = mid_shoulder[0] + t * (mid_hip[0] - mid_shoulder[0])
        y = mid_shoulder[1] + t * (mid_hip[1] - mid_shoulder[1])
        points.append((x, y))
    points.append(mid_hip)
    return points


def draw_spine_chain(img, points, dev_low_ratio=0.02, dev_high_ratio=0.05):
    """วาดกระดูกสันหลังเป็นเส้นตรงต่อกันทีละข้อ (polyline) พร้อมจุดข้อต่อ
    แทนที่จะเป็นเส้นโค้งเบซิเยร์เรียบ ๆ เพื่อให้เห็นตำแหน่งแต่ละระดับชัดเจน
    สีของจุดข้อต่อบ่งบอกระดับการเบี่ยงเบนจากแนวดิ่งอ้างอิง (คอ-สะโพก) ของข้อนั้น ๆ"""
    ref_top = points[0]
    ref_bottom = points[-1]
    ref_dx = ref_bottom[0] - ref_top[0]
    ref_dy = ref_bottom[1] - ref_top[1]
    ref_len = math.hypot(ref_dx, ref_dy) or 1.0

    # เส้นอ้างอิงแนวดิ่ง (plumb line) แบบเส้นประ เพื่อเทียบความเอียง
    p_top = (int(ref_top[0]), int(ref_top[1]))
    p_bottom = (int(ref_bottom[0]), int(ref_bottom[1]))
    n_dash = 24
    for i in range(n_dash):
        if i % 2 == 0:
            continue
        t0, t1 = i / n_dash, (i + 1) / n_dash
        x0 = int(ref_top[0] + t0 * ref_dx); y0 = int(ref_top[1] + t0 * ref_dy)
        x1 = int(ref_top[0] + t1 * ref_dx); y1 = int(ref_top[1] + t1 * ref_dy)
        cv2.line(img, (x0, y0), (x1, y1), (180, 180, 180), 1, cv2.LINE_AA)

    # เส้นกระดูกสันหลัง: ต่อจุดข้อต่อด้วยเส้นตรงทีละช่วง (ไม่ใช่เส้นโค้งเรียบ)
    for i in range(len(points) - 1):
        p1 = (int(points[i][0]), int(points[i][1]))
        p2 = (int(points[i + 1][0]), int(points[i + 1][1]))
        cv2.line(img, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)

    max_dev_px = 0.0
    max_dev_dir = "-"
    joint_devs = []
    for pt in points:
        px, py = pt
        vx, vy = px - ref_top[0], py - ref_top[1]
        proj = (vx * ref_dx + vy * ref_dy) / (ref_len ** 2)
        proj = min(max(proj, 0.0), 1.0)
        line_x = ref_top[0] + proj * ref_dx
        line_y = ref_top[1] + proj * ref_dy
        dev_px = px - line_x  # ค่าบวก = เบี่ยงขวา, ค่าลบ = เบี่ยงซ้าย (เทียบผู้ถูกถ่ายภาพ)
        dev_ratio = abs(dev_px) / ref_len
        joint_devs.append(dev_px)

        if abs(dev_px) > abs(max_dev_px):
            max_dev_px = dev_px
            max_dev_dir = "ขวา" if dev_px > 0 else "ซ้าย" if dev_px < 0 else "-"

        if dev_ratio < dev_low_ratio:
            color = (0, 200, 0)
        elif dev_ratio < dev_high_ratio:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.circle(img, (int(px), int(py)), 7, color, -1)
        cv2.circle(img, (int(px), int(py)), 7, (255, 255, 255), 1, cv2.LINE_AA)

    max_dev_ratio = abs(max_dev_px) / ref_len
    return {
        "max_dev_px": max_dev_px,
        "max_dev_ratio": max_dev_ratio,
        "max_dev_dir": max_dev_dir,
        "joint_devs": joint_devs,
        "ref_len": ref_len,
    }


def analyze_standing(image_bgr):
    h, w, _ = image_bgr.shape
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result    = detector.detect(mp_image)

    if not result.pose_landmarks:
        return None, None

    landmarks      = result.pose_landmarks[0]
    nose           = landmarks[0]
    left_ear       = landmarks[7]
    right_ear      = landmarks[8]
    left_shoulder  = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip       = landmarks[23]
    right_hip      = landmarks[24]
    left_knee      = landmarks[25]
    right_knee     = landmarks[26]

    dx_s = (right_shoulder.x - left_shoulder.x) * w
    dy_s = (right_shoulder.y - left_shoulder.y) * h
    shoulder_slope = abs(dy_s / dx_s) if dx_s != 0 else 0

    # เช็คว่าสะโพกอยู่ในเฟรมจริงหรือไม่ (ถ้า visibility ต่ำ = โมเดลกำลังเดา ไม่ใช่จุดจริง)
    hip_visibility = min(
        getattr(left_hip, "visibility", 1.0),
        getattr(right_hip, "visibility", 1.0),
    )
    hips_visible = hip_visibility >= HIP_VISIBILITY_THRESHOLD

    if hips_visible:
        dx_h = (right_hip.x - left_hip.x) * w
        dy_h = (right_hip.y - left_hip.y) * h
        hip_slope = abs(dy_h / dx_h) if dx_h != 0 else 0
    else:
        dy_h = None
        hip_slope = None

    mid_shoulder_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
    mid_shoulder_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
    y_knee         = int((left_knee.y + right_knee.y) / 2 * h)

    if hips_visible:
        mid_hip_x = int((left_hip.x + right_hip.x) / 2 * w)
        mid_hip_y = int((left_hip.y + right_hip.y) / 2 * h)
    else:
        # สะโพกไม่อยู่ในเฟรม -> ใช้ขอบล่างสุดของภาพที่มองเห็นจริงแทน (ตรงลงมาจากไหล่)
        mid_hip_x = mid_shoulder_x
        mid_hip_y = h - 10

    if hips_visible:
        dx_trunk = mid_shoulder_x - mid_hip_x
        dy_trunk = mid_shoulder_y - mid_hip_y
        trunk_tilt_angle = math.degrees(math.atan2(abs(dx_trunk), abs(dy_trunk))) if dy_trunk != 0 else 90.0
    else:
        trunk_tilt_angle = None

    shoulder_tilt_dir = "right_up" if dy_s < 0 else "left_up" if dy_s > 0 else "level"
    if hips_visible:
        hip_tilt_dir   = "right_up" if dy_h < 0 else "left_up" if dy_h > 0 else "level"
        same_direction = (shoulder_tilt_dir == hip_tilt_dir) and shoulder_tilt_dir != "level"
    else:
        hip_tilt_dir   = "unknown"
        same_direction = False

    annotated    = image_bgr.copy()
    x_left_edge  = 10
    x_right_edge = w - 10

    # เส้นประแบ่งโซน
    draw_dashed_line(annotated, mid_shoulder_y, x_left_edge, x_right_edge)
    if hips_visible:
        draw_dashed_line(annotated, mid_hip_y, x_left_edge, x_right_edge)

    # ป้ายโซน
    x_label = w - 140
    if hips_visible:
        draw_zone_label(annotated, 'Zone1: Shoulder', (mid_shoulder_y + mid_hip_y) // 2, (200, 100, 30), x_label)
        draw_zone_label(annotated, 'Zone2: Hip',      (mid_hip_y + y_knee) // 2,         (30, 80, 200),  x_label)
    else:
        draw_zone_label(annotated, 'Zone1: Shoulder', (mid_shoulder_y + h) // 2, (200, 100, 30), x_label)
        cv2.putText(annotated, "Hip: not in frame", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # เส้นไหล่และสะโพก
    cv2.line(annotated,
             (int(left_shoulder.x*w), int(left_shoulder.y*h)),
             (int(right_shoulder.x*w), int(right_shoulder.y*h)),
             (255, 200, 0), 3)
    if hips_visible:
        cv2.line(annotated,
                 (int(left_hip.x*w), int(left_hip.y*h)),
                 (int(right_hip.x*w), int(right_hip.y*h)),
                 (255, 200, 0), 3)

    # จุดพิกัด
    visible_points = [left_shoulder, right_shoulder]
    if hips_visible:
        visible_points += [left_hip, right_hip]
    for lm in visible_points:
        x, y = int(lm.x*w), int(lm.y*h)
        cv2.circle(annotated, (x, y), 8, (0, 0, 255), -1)

    # แนวกระดูกสันหลัง: เส้นตรงต่อกันทีละข้อ (polyline) ยึดจุดจาก
    # คอ (ประมาณจากหู+ไหล่) -> ไหล่ -> ระดับอก/เอวโดยประมาณ -> สะโพก
    neck_pt = estimate_neck_point(nose, left_ear, right_ear, mid_shoulder_x, mid_shoulder_y, w, h)
    spine_points = build_spine_chain(
        neck_pt,
        (mid_shoulder_x, mid_shoulder_y),
        (mid_hip_x, mid_hip_y),
        n_intermediate=3,
    )
    spine_info = draw_spine_chain(annotated, spine_points)

    # Rib Hump
    rib_hump_diff = 0.0
    rib_hump_side = "-"
    y_top    = mid_shoulder_y
    y_bottom = mid_hip_y
    x_center = mid_shoulder_x
    if y_bottom > y_top + 10 and 0 < x_center < w:
        region = image_bgr[y_top:y_bottom, :, :]
        h_r, w_r, _ = region.shape
        if h_r > 10 and w_r > 10:
            gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            cx     = min(x_center, w_r - 1)
            left_b  = float(np.mean(gray[:, :cx]))
            right_b = float(np.mean(gray[:, cx:]))
            rib_hump_diff = abs(left_b - right_b)
            rib_hump_side = "Left" if left_b > right_b else "Right"
            lc = (0, 200, 255) if left_b  > right_b else (80, 80, 80)
            rc = (0, 200, 255) if right_b > left_b  else (80, 80, 80)
            cv2.rectangle(annotated, (0, y_top),        (x_center, y_bottom), lc, 1)
            cv2.rectangle(annotated, (x_center, y_top), (w, y_bottom),        rc, 1)
            cv2.putText(annotated,
                        f"Rib Hump: {rib_hump_side} ({rib_hump_diff:.1f})",
                        (10, max(y_top - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    return {
        "shoulder_slope":       shoulder_slope,
        "hip_slope":            hip_slope,
        "trunk_tilt_angle":     trunk_tilt_angle,
        "shoulder_tilt_dir":    shoulder_tilt_dir,
        "hip_tilt_dir":         hip_tilt_dir,
        "same_direction":       same_direction,
        "rib_hump_diff":        rib_hump_diff,
        "rib_hump_side":        rib_hump_side,
        "spine_dev_ratio":      spine_info["max_dev_ratio"],
        "spine_dev_dir":        spine_info["max_dev_dir"],
        "spine_joint_devs":     spine_info["joint_devs"],
        "hips_visible":         hips_visible,
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
SPINE_DEV_LOW, SPINE_DEV_HIGH = 0.02, 0.05  # สัดส่วนการเบี่ยงเบนต่อความยาวลำตัว (คอ-สะโพก)
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
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result, _  = analyze_standing(image_bgr)

            if result is None:
                st.warning(f"ไม่พบร่างกายในภาพ: {f.name}")
                continue

            if not result["hips_visible"]:
                st.warning(f"ไม่เห็นสะโพกในภาพ (ข้ามรูปนี้เพื่อไม่ให้เกณฑ์เพี้ยน): {f.name}")
                continue

            rows.append({
                "filename":       f.name,
                "shoulder_slope": round(result["shoulder_slope"], 4),
                "hip_slope":      round(result["hip_slope"], 4),
                "trunk_tilt":     round(result["trunk_tilt_angle"], 2),
                "rib_hump_diff":  round(result["rib_hump_diff"], 2),
                "spine_dev_ratio": round(result["spine_dev_ratio"], 4),
            })

        if rows:
            st.subheader("ผลลัพธ์รายบุคคล")
            st.table(rows)

            stats = {
                "n": len(rows),
                "shoulder_slope":   {"mean": float(np.mean([r["shoulder_slope"] for r in rows])), "sd": float(np.std([r["shoulder_slope"] for r in rows]))},
                "hip_slope":        {"mean": float(np.mean([r["hip_slope"]      for r in rows])), "sd": float(np.std([r["hip_slope"]      for r in rows]))},
                "trunk_tilt_angle": {"mean": float(np.mean([r["trunk_tilt"]     for r in rows])), "sd": float(np.std([r["trunk_tilt"]     for r in rows]))},
                "rib_hump_diff":    {"mean": float(np.mean([r["rib_hump_diff"]  for r in rows])), "sd": float(np.std([r["rib_hump_diff"]  for r in rows]))},
                "spine_dev_ratio":  {"mean": float(np.mean([r["spine_dev_ratio"] for r in rows])), "sd": float(np.std([r["spine_dev_ratio"] for r in rows]))},
            }

            st.subheader("สถิติสรุป (Mean ± SD)")
            st.write(f"จำนวนตัวอย่าง: {stats['n']} คน")
            for key, label in [
                ("shoulder_slope",   "Shoulder Slope"),
                ("hip_slope",        "Hip Slope"),
                ("trunk_tilt_angle", "Trunk Tilt Angle"),
                ("rib_hump_diff",    "Rib Hump Diff"),
                ("spine_dev_ratio",  "Spine Deviation Ratio"),
            ]:
                st.write(f"{label}: {stats[key]['mean']:.4f} ± {stats[key]['sd']:.4f}")

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
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("กำลังวิเคราะห์..."):
            result, annotated = analyze_standing(image_bgr)

        if result is None:
            st.error("ไม่พบร่างกายในภาพ กรุณาลองใหม่")
        else:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb,
                     caption="จุดแดง = ไหล่ (และสะโพกถ้าเห็นในภาพ) | เส้นเหลือง = แนวกระดูกสันหลัง (คอ→ไหล่→อก/เอว→สะโพกหรือขอบภาพ) | "
                             "จุดข้อต่อเขียว/ส้ม/แดง = ระดับการเบี่ยงเบนจากแนวดิ่งของแต่ละข้อ | เส้นประเทา = แนวดิ่งอ้างอิง | กรอบฟ้า = ด้านที่นูนกว่า",
                     use_container_width=True)

            if not result["hips_visible"]:
                st.warning("⚠️ ไม่เห็นสะโพกในภาพ — ระบบจะประเมินเฉพาะแนวไหล่และหลังส่วนบนเท่านั้น "
                           "หากต้องการประเมินความเอียงของสะโพก (pelvic obliquity) ด้วย กรุณาถ่ายภาพให้เห็นตั้งแต่ไหล่ถึงสะโพก")

            if baseline and "spine_dev_ratio" in baseline:
                shoulder_risk, shoulder_color = get_risk_level_baseline(
                    result["shoulder_slope"], baseline["shoulder_slope"]["mean"], baseline["shoulder_slope"]["sd"])
                spine_risk, spine_color = get_risk_level_baseline(
                    result["spine_dev_ratio"], baseline["spine_dev_ratio"]["mean"], baseline["spine_dev_ratio"]["sd"])
                if result["hips_visible"]:
                    hip_risk, hip_color = get_risk_level_baseline(
                        result["hip_slope"], baseline["hip_slope"]["mean"], baseline["hip_slope"]["sd"])
            else:
                shoulder_risk, shoulder_color = get_risk_level_default(result["shoulder_slope"], SLOPE_LOW, SLOPE_HIGH)
                spine_risk, spine_color       = get_risk_level_default(result["spine_dev_ratio"], SPINE_DEV_LOW, SPINE_DEV_HIGH)
                if result["hips_visible"]:
                    hip_risk, hip_color = get_risk_level_default(result["hip_slope"], SLOPE_LOW, SLOPE_HIGH)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Shoulder Slope", f"{result['shoulder_slope']:.4f}")
                st.markdown(f"ความเสี่ยง: :{shoulder_color}[{shoulder_risk}]")
            with col2:
                if result["hips_visible"]:
                    st.metric("Hip Slope", f"{result['hip_slope']:.4f}")
                    st.markdown(f"ความเสี่ยง: :{hip_color}[{hip_risk}]")
                else:
                    st.metric("Hip Slope", "—")
                    st.markdown("ไม่พบสะโพกในภาพ")
            with col3:
                if result["trunk_tilt_angle"] is not None:
                    st.metric("Trunk Tilt Angle", f"{result['trunk_tilt_angle']:.2f}°")
                else:
                    st.metric("Trunk Tilt Angle", "—")

            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Rib Hump Diff", f"{result['rib_hump_diff']:.1f}")
            with col5:
                st.write(f"**ด้านที่นูนกว่า:** {result['rib_hump_side']}")
            with col6:
                st.metric("Spine Deviation", f"{result['spine_dev_ratio']:.4f}")
                st.markdown(f"ความเสี่ยง: :{spine_color}[{spine_risk}] (เบี่ยง{result['spine_dev_dir']})")

            if not result["hips_visible"]:
                curve_note = "ไม่เห็นสะโพกในภาพ จึงยังไม่สามารถบอกลักษณะการบิดของลำตัวทั้งท่อนได้"
            elif result["same_direction"]:
                curve_note = "ไหล่และสะโพกเอียงไปทาง**เดียวกัน** → ลำตัวเอียงไปทั้งแท่ง (C-curve)"
            elif result["shoulder_tilt_dir"] == "level" or result["hip_tilt_dir"] == "level":
                curve_note = "ไม่พบการเอียงที่ชัดเจน"
            else:
                curve_note = "ไหล่และสะโพกเอียง**คนละทาง** → อาจมีการบิดของลำตัวสองช่วง (เข้าข่าย S-curve เบื้องต้น)"

            st.write(f"**ลักษณะแนวลำตัว:** {curve_note}")

            if result["hips_visible"]:
                overall_risk = max(shoulder_risk, hip_risk, spine_risk, key=lambda r: ["ต่ำ (ปกติ)", "ปานกลาง", "สูง"].index(r))
            else:
                overall_risk = max(shoulder_risk, spine_risk, key=lambda r: ["ต่ำ (ปกติ)", "ปานกลาง", "สูง"].index(r))

            if result["hips_visible"] and not result["same_direction"] and result["shoulder_tilt_dir"] != "level" and result["hip_tilt_dir"] != "level":
                if overall_risk == "ต่ำ (ปกติ)":
                    overall_risk = "ปานกลาง"

            st.subheader(f"ผลการประเมิน: {overall_risk}")
            st.info(RECOMMENDATIONS[overall_risk])
            st.caption("⚠️ ระบบนี้เป็นเครื่องมือคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้")