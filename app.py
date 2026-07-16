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
HIP_SLOPE_MAX_PLAUSIBLE = 0.35  # ~19 องศา ถือว่าสุดขั้วมากแล้วสำหรับสะโพกคนยืนตรง เกินนี้ = จุดหลุดแน่ๆ
HIP_WIDTH_MIN_RATIO     = 0.5   # เส้นสะโพกต้องกว้างอย่างน้อยครึ่งหนึ่งของเส้นไหล่ ไม่งั้นถือว่าจุดหลุด
SCAPULA_ZONE_FRAC = 0.35  # สัดส่วนช่วงไหล่-เอวที่นับเป็นโซนสะบัก (นับจากใต้ไหล่ลงมา)

# ---------- การหา "เอว" แบบไม่พึ่งจุด landmark สะโพก ----------
# ใช้การเปรียบเทียบสีของแต่ละพิกเซลกับสีพื้นหลังในแถวเดียวกัน (รองรับพื้นหลังไล่เฉด)
# แล้วหาจุดที่ขอบลำตัวซ้าย/ขวาเว้าเข้ามากที่สุด (จุดคอดที่สุดของเอว) แยกกันแต่ละฝั่ง
WAIST_BG_DIST_THRESHOLD  = 35   # ค่าความต่างสี (BGR) ขั้นต่ำที่ถือว่าเป็น "ลำตัว" ไม่ใช่พื้นหลัง
WAIST_TOP_MARGIN_FRAC    = 0.15  # ตัดขอบบนออก (ใกล้รักแร้/ไหล่ ยังไม่ใช่เอว)
WAIST_BOTTOM_MARGIN_FRAC = 0.12  # ตัดขอบล่างออก (กันขอบกางเกง/ชุดชั้นในกวนผล)
WAIST_MIN_VALID_ROWS     = 15    # ต้องมีแถวที่หาลำตัวเจอมากพอ ไม่งั้นถือว่าตรวจไม่ได้


def estimate_bg_gradient(image_bgr):
    """ประมาณสีพื้นหลังจาก 4 มุมภาพ (ซึ่งเป็นพื้นหลังแน่นอนเสมอในภาพถ่ายลักษณะนี้)
    แล้วทำ interpolation แนวตั้งเพื่อรองรับพื้นหลังไล่เฉด โดยไม่เสี่ยงปนเปื้อนจาก
    ตัวแบบ (ต่างจากการ sample ขอบซ้าย-ขวาของแต่ละแถวที่อาจโดนแขนบังได้ในบางภาพ)"""
    h, w, _ = image_bgr.shape
    patch = max(4, int(min(h, w) * 0.04))
    tl = image_bgr[0:patch, 0:patch].reshape(-1, 3)
    tr = image_bgr[0:patch, w - patch:w].reshape(-1, 3)
    bl = image_bgr[h - patch:h, 0:patch].reshape(-1, 3)
    br = image_bgr[h - patch:h, w - patch:w].reshape(-1, 3)
    top_bg = np.median(np.vstack([tl, tr]), axis=0)
    bot_bg = np.median(np.vstack([bl, br]), axis=0)
    return top_bg, bot_bg


def _merge_close_runs(fg_row, max_bridge_gap):
    """รวมช่วง foreground ที่อยู่ใกล้กันมาก (ช่องว่างแคบกว่า max_bridge_gap)
    เพื่อไม่ให้เงาร่องกลางหลัง (เส้นกระดูกสันหลัง) มาตัดแบ่งลำตัวออกเป็นสองท่อนเท็จๆ
    แต่ยังคงแยกลำตัวออกจากแขนได้ ถ้าช่องว่างระหว่างกันกว้างกว่านี้จริง (เช่น ช่องรักแร้)"""
    runs = []
    x, n = 0, len(fg_row)
    while x < n:
        if fg_row[x]:
            start = x
            while x < n and fg_row[x]:
                x += 1
            runs.append([start, x - 1])
        else:
            x += 1
    if not runs:
        return []
    merged = [runs[0]]
    for r in runs[1:]:
        gap = r[0] - merged[-1][1] - 1
        if gap <= max_bridge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def find_waist_points(image_bgr, y_top, y_bottom,
                       top_margin_frac=WAIST_TOP_MARGIN_FRAC,
                       bottom_margin_frac=WAIST_BOTTOM_MARGIN_FRAC,
                       bg_dist_threshold=WAIST_BG_DIST_THRESHOLD,
                       min_valid_rows=WAIST_MIN_VALID_ROWS,
                       min_width_frac=0.10,
                       bridge_gap_frac=0.015):
    """หาตำแหน่งจุด 'เอว' ซ้าย/ขวา จากรูปทรงลำตัว (silhouette) แทนที่จะพึ่งจุดสะโพก
    เหมาะกับภาพที่ถูก crop สูงกว่าสะโพก เพราะเอวมักยังติดอยู่ในเฟรม
    คืนค่า (left_pt, right_pt) หรือ None ถ้าหาไม่ได้ชัดพอ"""
    h, w, _ = image_bgr.shape
    y_top = max(0, min(int(y_top), h - 1))
    y_bottom = max(0, min(int(y_bottom), h - 1))
    if y_bottom <= y_top:
        return None

    span = y_bottom - y_top
    y_start = y_top + int(span * top_margin_frac)
    y_end   = y_bottom - int(span * bottom_margin_frac)
    if y_end <= y_start:
        return None

    top_bg, bot_bg = estimate_bg_gradient(image_bgr)
    bridge_gap = max(4, int(w * bridge_gap_frac))
    min_width  = w * min_width_frac
    left_xs, right_xs = {}, {}

    for y in range(y_start, y_end):
        t = y / max(h - 1, 1)
        bg_ref = top_bg * (1 - t) + bot_bg * t  # รองรับพื้นหลังไล่เฉดแนวตั้ง
        row = image_bgr[y].astype(np.float32)
        diffs = np.linalg.norm(row - bg_ref, axis=1)
        fg = diffs > bg_dist_threshold

        runs = _merge_close_runs(fg, bridge_gap)
        if not runs:
            continue
        best = max(runs, key=lambda r: r[1] - r[0])  # ช่วงลำตัว = ช่วงที่กว้างสุดหลังเชื่อมรอยเงา
        if (best[1] - best[0]) < min_width:
            continue  # แถวนี้หาลำตัวไม่เจอชัดพอ ข้ามไป

        left_xs[y]  = best[0]
        right_xs[y] = best[1]

    if len(left_xs) < min_valid_rows or len(right_xs) < min_valid_rows:
        return None

    # จุดเอวฝั่งซ้าย = แถวที่ขอบซ้ายเว้าเข้ามาทางขวาสุด (x มากสุด)
    left_waist_y  = max(left_xs, key=lambda y: left_xs[y])
    # จุดเอวฝั่งขวา = แถวที่ขอบขวาเว้าเข้ามาทางซ้ายสุด (x น้อยสุด)
    right_waist_y = min(right_xs, key=lambda y: right_xs[y])

    left_pt  = (float(left_xs[left_waist_y]),   float(left_waist_y))
    right_pt = (float(right_xs[right_waist_y]), float(right_waist_y))
    return left_pt, right_pt


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


def measure_zone_brightness(image_bgr, annotated, y_top, y_bottom, x_center, w,
                             label, label_y_offset=-10, box_color_hint=(0, 200, 255)):
    """เทียบความสว่างเฉลี่ยซ้าย-ขวาของโซนที่กำหนด (ใช้ตรวจความไม่สมมาตรของพื้นผิวหลัง
    เช่น Rib Hump หรือสะบักนูน) แล้ววาดกรอบ+ป้ายกำกับลงบนภาพ
    คืนค่า (diff, side) หรือ (0.0, '-') ถ้าโซนเล็กเกินไปจนวัดไม่ได้"""
    diff, side = 0.0, "-"
    if y_bottom > y_top + 10 and 0 < x_center < w:
        region = image_bgr[y_top:y_bottom, :, :]
        h_r, w_r, _ = region.shape
        if h_r > 10 and w_r > 10:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            cx = min(x_center, w_r - 1)
            left_b  = float(np.mean(gray[:, :cx]))
            right_b = float(np.mean(gray[:, cx:]))
            diff = abs(left_b - right_b)
            side = "Left" if left_b > right_b else "Right"
            lc = box_color_hint if left_b  > right_b else (80, 80, 80)
            rc = box_color_hint if right_b > left_b  else (80, 80, 80)
            cv2.rectangle(annotated, (0, y_top),        (x_center, y_bottom), lc, 1)
            cv2.rectangle(annotated, (x_center, y_top), (w, y_bottom),        rc, 1)
            label_y = y_top + label_y_offset if label_y_offset < 0 else y_bottom + label_y_offset
            cv2.putText(annotated,
                        f"{label}: {side} ({diff:.1f})",
                        (10, max(label_y, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_hint, 1, cv2.LINE_AA)
    return diff, side


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

        # เช็คความสมเหตุสมผลเพิ่มเติม: บาง case MediaPipe มั่นใจสูงแต่จุดหลุดจริง
        # (เช่น โดนขอบกางเกง/เงาบัง) ทำให้ hip_slope พุ่งเกินจริงทางกายวิภาค
        # หรือเส้นสะโพกแคบผิดปกติเมื่อเทียบไหล่ -> ถือว่าจุดไม่น่าเชื่อถือ ตัดกลับไปใช้ Waist แทน
        shoulder_width_px = abs(dx_s)
        hip_width_px = abs(dx_h)
        implausible_slope = hip_slope > HIP_SLOPE_MAX_PLAUSIBLE
        implausible_width = (
            shoulder_width_px > 0 and
            hip_width_px < HIP_WIDTH_MIN_RATIO * shoulder_width_px
        )
        if implausible_slope or implausible_width:
            hips_visible = False
            dy_h = None
            hip_slope = None
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

    # Rib Hump: เทียบความสว่างซ้าย-ขวาทั่วทั้งแผ่นหลัง (ไหล่ถึงเอว/สะโพก)
    rib_hump_diff, rib_hump_side = measure_zone_brightness(
        image_bgr, annotated, mid_shoulder_y, mid_hip_y, mid_shoulder_x, w,
        label="Rib Hump", label_y_offset=-10)

    # Scapula Prominence: เทียบความสว่างซ้าย-ขวาเฉพาะโซนสะบัก (ใต้ไหล่ลงมา ~35%
    # ของช่วงไหล่-เอว) เพราะสะบักนูนไม่เท่ากันเป็นสัญญาณคัดกรองแยกจาก Rib Hump ทั้งแผ่นหลัง
    scapula_y_top = mid_shoulder_y
    scapula_y_bottom = mid_shoulder_y + int(SCAPULA_ZONE_FRAC * (mid_hip_y - mid_shoulder_y))
    scapula_diff, scapula_side = measure_zone_brightness(
        image_bgr, annotated, scapula_y_top, scapula_y_bottom, mid_shoulder_x, w,
        label="Scapula", label_y_offset=14, box_color_hint=(0, 255, 120))

    y_top, y_bottom = mid_shoulder_y, mid_hip_y  # ใช้ต่อสำหรับหา Waist

    # เอว: หาจากรูปทรงลำตัว ไม่พึ่งจุด landmark สะโพก จึงใช้ได้แม้ภาพตัดสูงกว่าสะโพก
    waist_result = find_waist_points(image_bgr, y_top, y_bottom)
    if waist_result:
        left_waist, right_waist = waist_result
        dx_wst = right_waist[0] - left_waist[0]
        dy_wst = right_waist[1] - left_waist[1]
        waist_slope    = abs(dy_wst / dx_wst) if dx_wst != 0 else 0
        waist_tilt_dir = "right_up" if dy_wst < 0 else "left_up" if dy_wst > 0 else "level"
        waist_detected = True

        cv2.circle(annotated, (int(left_waist[0]), int(left_waist[1])), 6, (255, 0, 255), -1)
        cv2.circle(annotated, (int(right_waist[0]), int(right_waist[1])), 6, (255, 0, 255), -1)
        cv2.line(annotated,
                 (int(left_waist[0]), int(left_waist[1])),
                 (int(right_waist[0]), int(right_waist[1])),
                 (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "Waist",
                    (int(left_waist[0]) - 10, int(min(left_waist[1], right_waist[1])) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    else:
        waist_slope    = None
        waist_tilt_dir = "unknown"
        waist_detected = False

    return {
        "shoulder_slope":       shoulder_slope,
        "hip_slope":            hip_slope,
        "trunk_tilt_angle":     trunk_tilt_angle,
        "shoulder_tilt_dir":    shoulder_tilt_dir,
        "hip_tilt_dir":         hip_tilt_dir,
        "same_direction":       same_direction,
        "rib_hump_diff":        rib_hump_diff,
        "rib_hump_side":        rib_hump_side,
        "scapula_diff":         scapula_diff,
        "scapula_side":         scapula_side,
        "spine_dev_ratio":      spine_info["max_dev_ratio"],
        "spine_dev_dir":        spine_info["max_dev_dir"],
        "spine_joint_devs":     spine_info["joint_devs"],
        "hips_visible":         hips_visible,
        "waist_detected":       waist_detected,
        "waist_slope":          waist_slope,
        "waist_tilt_dir":       waist_tilt_dir,
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
        hip_vals = []
        waist_vals = []
        for f in uploaded_files:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result, _  = analyze_standing(image_bgr)

            if result is None:
                st.warning(f"ไม่พบร่างกายในภาพ: {f.name}")
                continue

            row = {
                "filename":        f.name,
                "shoulder_slope":  round(result["shoulder_slope"], 4),
                "trunk_tilt":      round(result["trunk_tilt_angle"], 2) if result["trunk_tilt_angle"] is not None else None,
                "rib_hump_diff":   round(result["rib_hump_diff"], 2),
                "scapula_diff":    round(result["scapula_diff"], 2),
                "spine_dev_ratio": round(result["spine_dev_ratio"], 4),
                "hip_slope":       round(result["hip_slope"], 4) if result["hips_visible"] else None,
                "waist_slope":     round(result["waist_slope"], 4) if result["waist_detected"] else None,
            }
            rows.append(row)
            if result["hips_visible"]:
                hip_vals.append(row["hip_slope"])
            if result["waist_detected"]:
                waist_vals.append(row["waist_slope"])
            if not result["hips_visible"] and not result["waist_detected"]:
                st.warning(f"ไม่เห็นทั้งสะโพกและเอวในภาพ (ข้ามการนับ Hip/Waist สำหรับรูปนี้): {f.name}")

        if rows:
            st.subheader("ผลลัพธ์รายบุคคล")
            st.table(rows)

            def _stat(vals):
                if not vals:
                    return None
                return {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}

            stats = {
                "n":                len(rows),
                "shoulder_slope":   _stat([r["shoulder_slope"] for r in rows]),
                "trunk_tilt_angle": _stat([r["trunk_tilt"] for r in rows if r["trunk_tilt"] is not None]),
                "rib_hump_diff":    _stat([r["rib_hump_diff"] for r in rows]),
                "scapula_diff":     _stat([r["scapula_diff"] for r in rows]),
                "spine_dev_ratio":  _stat([r["spine_dev_ratio"] for r in rows]),
                "hip_slope":        _stat(hip_vals),
                "n_hip":            len(hip_vals),
                "waist_slope":      _stat(waist_vals),
                "n_waist":          len(waist_vals),
            }

            st.subheader("สถิติสรุป (Mean ± SD)")
            st.write(f"จำนวนตัวอย่างทั้งหมด: {stats['n']} คน "
                     f"(มีข้อมูลสะโพก {stats['n_hip']} คน, มีข้อมูลเอว {stats['n_waist']} คน)")
            for key, label in [
                ("shoulder_slope",   "Shoulder Slope"),
                ("hip_slope",        "Hip Slope"),
                ("waist_slope",      "Waist Slope"),
                ("trunk_tilt_angle", "Trunk Tilt Angle"),
                ("rib_hump_diff",    "Rib Hump Diff"),
                ("scapula_diff",     "Scapula Prominence"),
                ("spine_dev_ratio",  "Spine Deviation Ratio"),
            ]:
                if stats[key] is None:
                    st.write(f"{label}: ไม่มีข้อมูลเพียงพอ")
                else:
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
                     caption="จุดแดง = ไหล่ (และสะโพกถ้าเห็นในภาพ) | จุดม่วง = เอว (หาจากรูปทรงลำตัว) | เส้นเหลือง = แนวกระดูกสันหลัง (คอ→ไหล่→อก/เอว→สะโพกหรือขอบภาพ) | "
                             "จุดข้อต่อเขียว/ส้ม/แดง = ระดับการเบี่ยงเบนจากแนวดิ่งของแต่ละข้อ | เส้นประเทา = แนวดิ่งอ้างอิง | กรอบฟ้า = Rib Hump ด้านที่นูนกว่า | กรอบเขียว = Scapula (สะบัก) ด้านที่นูนกว่า",
                     use_container_width=True)

            if not result["hips_visible"] and not result["waist_detected"]:
                st.warning("⚠️ ไม่พบทั้งสะโพกและเอวในภาพ — ระบบจะประเมินเฉพาะแนวไหล่และหลังส่วนบนเท่านั้น "
                           "แนะนำให้ถ่ายภาพให้เห็นตั้งแต่ไหล่ถึงเอวอย่างน้อย เพื่อความแม่นยำที่ดีขึ้น")
            elif not result["hips_visible"]:
                st.info("ℹ️ ไม่เห็นสะโพกในภาพ แต่ตรวจพบแนวเอวแทนได้ ระบบจะใช้ Waist Slope ช่วยประเมินความเอียงของลำตัวส่วนล่าง")

            if baseline and "spine_dev_ratio" in baseline:
                shoulder_risk, shoulder_color = get_risk_level_baseline(
                    result["shoulder_slope"], baseline["shoulder_slope"]["mean"], baseline["shoulder_slope"]["sd"])
                spine_risk, spine_color = get_risk_level_baseline(
                    result["spine_dev_ratio"], baseline["spine_dev_ratio"]["mean"], baseline["spine_dev_ratio"]["sd"])
                if result["hips_visible"] and baseline.get("hip_slope"):
                    hip_risk, hip_color = get_risk_level_baseline(
                        result["hip_slope"], baseline["hip_slope"]["mean"], baseline["hip_slope"]["sd"])
                elif result["hips_visible"]:
                    hip_risk, hip_color = get_risk_level_default(result["hip_slope"], SLOPE_LOW, SLOPE_HIGH)
                if result["waist_detected"] and baseline.get("waist_slope"):
                    waist_risk, waist_color = get_risk_level_baseline(
                        result["waist_slope"], baseline["waist_slope"]["mean"], baseline["waist_slope"]["sd"])
                elif result["waist_detected"]:
                    waist_risk, waist_color = get_risk_level_default(result["waist_slope"], SLOPE_LOW, SLOPE_HIGH)
            else:
                shoulder_risk, shoulder_color = get_risk_level_default(result["shoulder_slope"], SLOPE_LOW, SLOPE_HIGH)
                spine_risk, spine_color       = get_risk_level_default(result["spine_dev_ratio"], SPINE_DEV_LOW, SPINE_DEV_HIGH)
                if result["hips_visible"]:
                    hip_risk, hip_color = get_risk_level_default(result["hip_slope"], SLOPE_LOW, SLOPE_HIGH)
                if result["waist_detected"]:
                    waist_risk, waist_color = get_risk_level_default(result["waist_slope"], SLOPE_LOW, SLOPE_HIGH)

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

            col7, col8 = st.columns(2)
            with col7:
                if result["waist_detected"]:
                    st.metric("Waist Slope", f"{result['waist_slope']:.4f}")
                    st.markdown(f"ความเสี่ยง: :{waist_color}[{waist_risk}]")
                else:
                    st.metric("Waist Slope", "—")
                    st.markdown("ไม่พบแนวเอวในภาพ")
            with col8:
                st.metric("Scapula Prominence", f"{result['scapula_diff']:.1f}")
                st.write(f"**สะบักด้านที่นูนกว่า:** {result['scapula_side']}")

            if not result["hips_visible"]:
                curve_note = "ไม่เห็นสะโพกในภาพ จึงยังไม่สามารถบอกลักษณะการบิดของลำตัวทั้งท่อนได้"
            elif result["same_direction"]:
                curve_note = "ไหล่และสะโพกเอียงไปทาง**เดียวกัน** → ลำตัวเอียงไปทั้งแท่ง (C-curve)"
            elif result["shoulder_tilt_dir"] == "level" or result["hip_tilt_dir"] == "level":
                curve_note = "ไม่พบการเอียงที่ชัดเจน"
            else:
                curve_note = "ไหล่และสะโพกเอียง**คนละทาง** → อาจมีการบิดของลำตัวสองช่วง (เข้าข่าย S-curve เบื้องต้น)"

            st.write(f"**ลักษณะแนวลำตัว:** {curve_note}")

            if result["waist_detected"]:
                waist_dir_label = {"right_up": "เอียงขึ้นทางขวา", "left_up": "เอียงขึ้นทางซ้าย", "level": "ไม่เอียงชัดเจน"}.get(result["waist_tilt_dir"], "-")
                st.caption(f"แนวเอว: {waist_dir_label} (Waist Slope {result['waist_slope']:.4f})")

            risk_pool = [shoulder_risk, spine_risk]
            if result["hips_visible"]:
                risk_pool.append(hip_risk)
            if result["waist_detected"]:
                risk_pool.append(waist_risk)
            overall_risk = max(risk_pool, key=lambda r: ["ต่ำ (ปกติ)", "ปานกลาง", "สูง"].index(r))

            if result["hips_visible"] and not result["same_direction"] and result["shoulder_tilt_dir"] != "level" and result["hip_tilt_dir"] != "level":
                if overall_risk == "ต่ำ (ปกติ)":
                    overall_risk = "ปานกลาง"

            st.subheader(f"ผลการประเมิน: {overall_risk}")
            st.info(RECOMMENDATIONS[overall_risk])
            st.caption("⚠️ ระบบนี้เป็นเครื่องมือคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้")