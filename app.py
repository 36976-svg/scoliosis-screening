import math
import json
import os
import ssl

import streamlit as st
import cv2
import numpy as np

# บาง environment (เช่น เครื่อง Windows ที่ certificate store ไม่ครบ) จะ verify SSL
# ไม่ผ่านตอนดาวน์โหลดโมเดลจาก HuggingFace ครั้งแรก จึงปิดการเช็คไว้กันปัญหานี้
ssl._create_default_https_context = ssl._create_unverified_context

from spinepose import SpinePoseEstimator

st.set_page_config(page_title="ระบบคัดกรองภาวะกระดูกสันหลังคดเบื้องต้น", layout="centered")

BASELINE_FILE = "baseline.json"

# ---------- ดัชนี keypoint ของโมเดล SpinePose (โครงกระดูก SpineTrack, 37 จุด) ----------
# ต่างจาก MediaPipe ตรงที่โมเดลนี้ถูกเทรนมาให้จับจุด "บนแนวกระดูกสันหลังจริง"
# (โหนกกระดูกที่เห็นเวลาก้ม/ถ่ายภาพหลัง) ไม่ใช่แค่ไหล่กับสะโพกแล้วมาประมาณเอาเอง
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP           = 11, 12
NECK   = 18   # โคนคอ / ประมาณตำแหน่ง C7
PELVIS = 19   # กึ่งกลางสะโพก/กระเบนเหน็บ (sacrum)
SPINE_01, SPINE_02, SPINE_03, SPINE_04, SPINE_05 = 26, 27, 28, 29, 30  # โหนกกระดูกสันหลังช่วงอก-เอว
NECK_02, NECK_03 = 35, 36  # โหนกกระดูกคอช่วงล่าง (ใกล้ท้ายทอย)

# แนวกระดูกสันหลังจริงจากโมเดล เรียงจาก "บนสุด (ท้ายทอย/โคนคอ)" ไล่ลงมา "ล่างสุด (สะโพก)"
SPINE_CHAIN_TOP_TO_BOTTOM = [NECK_03, NECK_02, NECK, SPINE_05, SPINE_04, SPINE_03, SPINE_02, SPINE_01, PELVIS]

KEYPOINT_SCORE_THRESHOLD = 0.4  # ความมั่นใจขั้นต่ำที่ยอมรับว่าจุดนั้น "อยู่ในภาพจริง"
HIP_SCORE_THRESHOLD      = 0.5  # สะโพกต้องมั่นใจสูงกว่าหน่อยเพราะใช้คำนวณความเสี่ยงโดยตรง

MODEL_SIZE_OPTIONS = ["small", "medium", "large", "xlarge"]


@st.cache_resource(show_spinner=False)
def load_estimator(mode: str):
    with st.spinner(f"กำลังโหลดโมเดล SpinePose ({mode}) — ครั้งแรกจะดาวน์โหลดไฟล์โมเดล อาจใช้เวลาสักครู่..."):
        return SpinePoseEstimator(mode=mode, detector="rfdetr")


def draw_dashed_line(img, y, x_left, x_right, color=(200, 200, 200), thickness=1):
    for x in range(x_left, x_right, 20):
        cv2.line(img, (x, y), (min(x + 10, x_right), y), color, thickness)


def draw_zone_label(img, label, y_mid, color_bg, x_start):
    cv2.rectangle(img, (x_start, y_mid - 14), (x_start + 130, y_mid + 14), color_bg, -1)
    cv2.rectangle(img, (x_start, y_mid - 14), (x_start + 130, y_mid + 14), (255, 255, 255), 1)
    cv2.putText(img, label, (x_start + 5, y_mid + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def get_visible_spine_chain(kp, sc, threshold=KEYPOINT_SCORE_THRESHOLD):
    """คืนจุดข้อต่อบนแนวกระดูกสันหลัง (จุดจริงจากโมเดล ไม่ใช่จุดประมาณ)
    เฉพาะจุดที่โมเดลมั่นใจว่าเห็นจริงในภาพ เรียงจากบน (คอ) ลงล่าง (สะโพก)"""
    points = []
    for idx in SPINE_CHAIN_TOP_TO_BOTTOM:
        if sc[idx] >= threshold:
            points.append((float(kp[idx][0]), float(kp[idx][1])))
    return points


def draw_spine_chain(img, points, dev_low_ratio=0.02, dev_high_ratio=0.05):
    """วาดกระดูกสันหลังเป็นเส้นตรงต่อกันทีละข้อ (polyline) พร้อมจุดข้อต่อจริงจากโมเดล
    สีของจุดข้อต่อบ่งบอกระดับการเบี่ยงเบนจากแนวดิ่งอ้างอิง (จุดบนสุด-ล่างสุดที่เห็น) ของข้อนั้น ๆ"""
    ref_top = points[0]
    ref_bottom = points[-1]
    ref_dx = ref_bottom[0] - ref_top[0]
    ref_dy = ref_bottom[1] - ref_top[1]
    ref_len = math.hypot(ref_dx, ref_dy) or 1.0

    # เส้นอ้างอิงแนวดิ่ง (plumb line) แบบเส้นประ เพื่อเทียบความเอียง
    n_dash = 24
    for i in range(n_dash):
        if i % 2 == 0:
            continue
        t0, t1 = i / n_dash, (i + 1) / n_dash
        x0 = int(ref_top[0] + t0 * ref_dx); y0 = int(ref_top[1] + t0 * ref_dy)
        x1 = int(ref_top[0] + t1 * ref_dx); y1 = int(ref_top[1] + t1 * ref_dy)
        cv2.line(img, (x0, y0), (x1, y1), (180, 180, 180), 1, cv2.LINE_AA)

    # เส้นกระดูกสันหลัง: ต่อจุดข้อต่อจริงด้วยเส้นตรงทีละช่วง (ไม่ใช่เส้นโค้งเรียบ)
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


def analyze_standing(image_bgr, estimator):
    h, w = image_bgr.shape[:2]

    bboxes = estimator.detect(image_bgr)
    if bboxes is None or len(bboxes) == 0:
        return None, None

    keypoints, scores = estimator.estimate(image_bgr, bboxes)
    if keypoints.shape[0] == 0:
        return None, None

    # ใช้คนแรกที่ตรวจพบ (บวก confidence กล่องสูงสุด ถ้ามีหลายคนในภาพ)
    kp = keypoints[0]
    sc = scores[0]

    if sc[LEFT_SHOULDER] < KEYPOINT_SCORE_THRESHOLD or sc[RIGHT_SHOULDER] < KEYPOINT_SCORE_THRESHOLD:
        return None, None  # มองไม่เห็นไหล่ชัดพอ ประเมินไม่ได้

    left_shoulder, right_shoulder = kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]

    dx_s = right_shoulder[0] - left_shoulder[0]
    dy_s = right_shoulder[1] - left_shoulder[1]
    shoulder_slope = abs(dy_s / dx_s) if dx_s != 0 else 0

    hips_visible = sc[LEFT_HIP] >= HIP_SCORE_THRESHOLD and sc[RIGHT_HIP] >= HIP_SCORE_THRESHOLD
    if hips_visible:
        left_hip, right_hip = kp[LEFT_HIP], kp[RIGHT_HIP]
        dx_h = right_hip[0] - left_hip[0]
        dy_h = right_hip[1] - left_hip[1]
        hip_slope = abs(dy_h / dx_h) if dx_h != 0 else 0
        mid_hip_x, mid_hip_y = (left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2
    else:
        dy_h = None
        hip_slope = None
        mid_hip_x = mid_hip_y = None

    mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
    mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    # จุดข้อต่อจริงบนกระดูกสันหลัง (เฉพาะจุดที่โมเดลมั่นใจ)
    spine_points = get_visible_spine_chain(kp, sc)
    if len(spine_points) < 2:
        return None, None  # จุดกระดูกสันหลังไม่พอสำหรับวิเคราะห์ (ภาพไม่ชัด/บังมากไป)

    if not hips_visible:
        # ใช้จุดกระดูกสันหลังที่ต่ำสุดเท่าที่มั่นใจได้ แทนสะโพกที่มองไม่เห็นในเฟรม
        mid_hip_x, mid_hip_y = spine_points[-1]

    top_pt, bottom_pt = spine_points[0], spine_points[-1]
    dx_trunk = top_pt[0] - bottom_pt[0]
    dy_trunk = top_pt[1] - bottom_pt[1]
    trunk_tilt_angle = math.degrees(math.atan2(abs(dx_trunk), abs(dy_trunk))) if dy_trunk != 0 else 90.0

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

    # เส้นประแบ่งโซน + ป้ายโซน
    draw_dashed_line(annotated, int(mid_shoulder_y), x_left_edge, x_right_edge)
    x_label = w - 140
    if hips_visible:
        draw_dashed_line(annotated, int(mid_hip_y), x_left_edge, x_right_edge)
        draw_zone_label(annotated, 'Zone1: Shoulder', int((mid_shoulder_y + mid_hip_y) // 2), (200, 100, 30), x_label)
    else:
        draw_zone_label(annotated, 'Zone1: Shoulder', int((mid_shoulder_y + h) // 2), (200, 100, 30), x_label)
        cv2.putText(annotated, "Hip: not in frame", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # เส้นไหล่และสะโพก
    cv2.line(annotated,
             (int(left_shoulder[0]), int(left_shoulder[1])),
             (int(right_shoulder[0]), int(right_shoulder[1])),
             (255, 200, 0), 3)
    if hips_visible:
        cv2.line(annotated,
                 (int(left_hip[0]), int(left_hip[1])),
                 (int(right_hip[0]), int(right_hip[1])),
                 (255, 200, 0), 3)

    # จุดพิกัดไหล่/สะโพก
    coord_points = [left_shoulder, right_shoulder]
    if hips_visible:
        coord_points += [left_hip, right_hip]
    for lm in coord_points:
        cv2.circle(annotated, (int(lm[0]), int(lm[1])), 8, (0, 0, 255), -1)

    # แนวกระดูกสันหลังจริง: เส้นตรงต่อกันทีละข้อ (polyline) จากจุดที่โมเดลตรวจพบจริง
    spine_info = draw_spine_chain(annotated, spine_points)

    # Rib Hump: เทียบความสว่างซ้าย-ขวาของแนวลำตัวช่วงที่เห็น
    rib_hump_diff = 0.0
    rib_hump_side = "-"
    y_top    = int(mid_shoulder_y)
    y_bottom = int(mid_hip_y)
    x_center = int(mid_shoulder_x)
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
        "shoulder_slope":     shoulder_slope,
        "hip_slope":          hip_slope,
        "trunk_tilt_angle":   trunk_tilt_angle,
        "shoulder_tilt_dir":  shoulder_tilt_dir,
        "hip_tilt_dir":       hip_tilt_dir,
        "same_direction":     same_direction,
        "rib_hump_diff":      rib_hump_diff,
        "rib_hump_side":      rib_hump_side,
        "spine_dev_ratio":    spine_info["max_dev_ratio"],
        "spine_dev_dir":      spine_info["max_dev_dir"],
        "spine_joint_devs":   spine_info["joint_devs"],
        "hips_visible":       hips_visible,
        "spine_points_found": len(spine_points),
        "spine_points_total": len(SPINE_CHAIN_TOP_TO_BOTTOM),
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
SPINE_DEV_LOW, SPINE_DEV_HIGH = 0.02, 0.05  # สัดส่วนการเบี่ยงเบนต่อความยาวช่วงกระดูกสันหลังที่เห็น
baseline = load_baseline()

st.sidebar.title("เมนู")
mode = st.sidebar.radio("เลือกโหมด", [
    "ประเมินผล",
    "Calibration (สร้างเกณฑ์จากกลุ่มปกติ)"
])
model_size = st.sidebar.selectbox(
    "ขนาดโมเดล SpinePose",
    MODEL_SIZE_OPTIONS,
    index=MODEL_SIZE_OPTIONS.index("large"),
    help="โมเดลใหญ่ขึ้น = แม่นขึ้นแต่ช้าลง (large แนะนำสำหรับงานคัดกรอง, small ถ้าต้องการความเร็ว)"
)
estimator = load_estimator(model_size)

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
            result, _  = analyze_standing(image_bgr, estimator)

            if result is None:
                st.warning(f"ไม่พบร่างกาย/กระดูกสันหลังในภาพ: {f.name}")
                continue

            if not result["hips_visible"]:
                st.warning(f"ไม่เห็นสะโพกในภาพ (ข้ามรูปนี้เพื่อไม่ให้เกณฑ์เพี้ยน): {f.name}")
                continue

            rows.append({
                "filename":        f.name,
                "shoulder_slope":  round(result["shoulder_slope"], 4),
                "hip_slope":       round(result["hip_slope"], 4),
                "trunk_tilt":      round(result["trunk_tilt_angle"], 2),
                "rib_hump_diff":   round(result["rib_hump_diff"], 2),
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
    st.caption("ขับเคลื่อนด้วย SpinePose — โมเดลที่จับจุดบนแนวกระดูกสันหลังจริง ไม่ใช่จุดประมาณจากไหล่/สะโพก")

    if baseline:
        st.caption(f"✅ ใช้เกณฑ์จากกลุ่มตัวอย่างปกติ ({baseline['n']} คน)")
    else:
        st.caption("⚠️ ยังไม่มีเกณฑ์จากกลุ่มตัวอย่าง — ใช้ค่าเริ่มต้น")

    uploaded_file = st.file_uploader("เลือกภาพถ่าย (ท่ายืนตรง)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("กำลังวิเคราะห์..."):
            result, annotated = analyze_standing(image_bgr, estimator)

        if result is None:
            st.error("ไม่พบร่างกายหรือจุดกระดูกสันหลังในภาพชัดพอ กรุณาลองใหม่ด้วยภาพที่เห็นไหล่และหลังชัดเจน")
        else:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb,
                     caption="จุดแดง = ไหล่ (และสะโพกถ้าเห็นในภาพ) | เส้นเหลือง = แนวกระดูกสันหลังจริงจากโมเดล (คอ→อก→เอว→สะโพก) | "
                             "จุดข้อต่อเขียว/ส้ม/แดง = ระดับการเบี่ยงเบนจากแนวดิ่งของแต่ละข้อ | เส้นประเทา = แนวดิ่งอ้างอิง | กรอบฟ้า = ด้านที่นูนกว่า",
                     use_container_width=True)

            st.caption(f"🔎 ตรวจพบจุดกระดูกสันหลังชัดเจน {result['spine_points_found']}/{result['spine_points_total']} จุด")
            if result["spine_points_found"] < result["spine_points_total"]:
                st.caption("(บางจุดอาจถูกบังหรืออยู่นอกเฟรม ระบบจะประเมินจากจุดที่มั่นใจได้เท่านั้น)")

            if not result["hips_visible"]:
                st.warning("⚠️ ไม่เห็นสะโพกในภาพ — ระบบจะประเมินเฉพาะแนวไหล่และหลังส่วนที่เห็นเท่านั้น "
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
                st.metric("Trunk Tilt Angle", f"{result['trunk_tilt_angle']:.2f}°")

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
