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

BASELINE_FILE  = "baseline.json"
MODEL_FILE     = "pose_landmarker_heavy.task"
MODEL_URL      = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

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
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)

detector = load_detector()

# ─── ค่าคงที่ ───────────────────────────────────────────────────────────────
HIP_VISIBILITY_THRESHOLD  = 0.4
HIP_SLOPE_MAX_PLAUSIBLE   = 0.35
HIP_WIDTH_MIN_RATIO       = 0.45
SLOPE_LOW, SLOPE_HIGH     = 0.05, 0.15
SPINE_DEV_LOW, SPINE_DEV_HIGH         = 0.02, 0.05
SCAPULA_DIFF_LOW, SCAPULA_DIFF_HIGH   = 6.0, 15.0
SCAPULA_HEIGHT_LOW, SCAPULA_HEIGHT_HIGH = 0.20, 0.40

# ─── helpers ──────────────────────────────────────────────────────────────────

def get_clean_mask(result, h, w):
    """สร้าง person mask จาก MediaPipe segmentation mask
    — ใช้เฉพาะเพื่อแสดงใน Processing Steps เท่านั้น ไม่ใช้วิเคราะห์ต่อ"""
    if not result.segmentation_masks:
        return np.ones((h, w), dtype=bool)
    try:
        seg = np.asarray(result.segmentation_masks[0].numpy_view())
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (seg > 0.5)
        # ขยาย + เติมรูโหว่ เพื่อให้ภาพใน step 2-4 สวยงาม
        m8 = mask.astype(np.uint8) * 255
        m8 = cv2.dilate(m8, np.ones((20, 20), np.uint8), iterations=2)
        ff = m8.copy()
        cv2.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0, 0), 128)
        m8[ff == 0] = 255
        return m8 > 0
    except Exception:
        return np.ones((h, w), dtype=bool)


def draw_dashed_line(img, y, x0, x1, color=(200, 200, 200), thickness=1):
    for x in range(x0, x1, 20):
        cv2.line(img, (x, y), (min(x+10, x1), y), color, thickness)


def draw_zone_label(img, label, y_mid, color_bg, x_start, w_label=130):
    x_start = max(0, min(x_start, img.shape[1] - w_label - 2))
    cv2.rectangle(img, (x_start, y_mid-14), (x_start+w_label, y_mid+14), color_bg, -1)
    cv2.rectangle(img, (x_start, y_mid-14), (x_start+w_label, y_mid+14), (255,255,255), 1)
    cv2.putText(img, label, (x_start+5, y_mid+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)


def apply_high_contrast(image_bgr, mask):
    """ขาวดำ contrast สูงเฉพาะส่วนคน"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[mask]
    if vals.size == 0:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo: hi = lo + 1
    s = np.clip((gray.astype(np.float32)-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)
    s[~mask] = 0
    return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)


def estimate_neck(nose, left_ear, right_ear, mid_sx, mid_sy, w, h):
    mx = (left_ear.x + right_ear.x)/2 * w
    my = (left_ear.y + right_ear.y)/2 * h
    return mx + 0.28*(mid_sx-mx), my + 0.28*(mid_sy-my)


def build_spine_chain(neck, mid_sh, mid_hip, n=3):
    pts = [neck, mid_sh]
    for i in range(1, n+1):
        t = i/(n+1)
        pts.append((mid_sh[0]+t*(mid_hip[0]-mid_sh[0]),
                     mid_sh[1]+t*(mid_hip[1]-mid_sh[1])))
    pts.append(mid_hip)
    return pts


def draw_spine_chain(img, pts, dev_lo=0.02, dev_hi=0.05):
    ref_dx = pts[-1][0]-pts[0][0]
    ref_dy = pts[-1][1]-pts[0][1]
    ref_len = math.hypot(ref_dx, ref_dy) or 1.0

    # เส้นประแนวอ้างอิง
    n_dash = 24
    for i in range(n_dash):
        if i % 2 == 0: continue
        t0, t1 = i/n_dash, (i+1)/n_dash
        cv2.line(img,
                 (int(pts[0][0]+t0*ref_dx), int(pts[0][1]+t0*ref_dy)),
                 (int(pts[0][0]+t1*ref_dx), int(pts[0][1]+t1*ref_dy)),
                 (180,180,180), 1, cv2.LINE_AA)

    for i in range(len(pts)-1):
        cv2.line(img,
                 (int(pts[i][0]), int(pts[i][1])),
                 (int(pts[i+1][0]), int(pts[i+1][1])),
                 (0,255,255), 3, cv2.LINE_AA)

    max_dev_px, max_dev_dir, devs = 0.0, "-", []
    for pt in pts:
        vx, vy = pt[0]-pts[0][0], pt[1]-pts[0][1]
        proj = min(max((vx*ref_dx+vy*ref_dy)/ref_len**2, 0), 1)
        lx = pts[0][0]+proj*ref_dx; ly = pts[0][1]+proj*ref_dy
        dev = pt[0]-lx
        devs.append(dev)
        dr = abs(dev)/ref_len
        c = (0,200,0) if dr<dev_lo else (0,165,255) if dr<dev_hi else (0,0,255)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 7, c, -1)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 7, (255,255,255), 1, cv2.LINE_AA)
        if abs(dev) > abs(max_dev_px):
            max_dev_px = dev
            max_dev_dir = "ขวา" if dev>0 else "ซ้าย" if dev<0 else "-"

    return {"max_dev_ratio": abs(max_dev_px)/ref_len,
            "max_dev_dir": max_dev_dir,
            "joint_devs": devs}


def find_scapula_landmark_based(image_bgr, left_shoulder, right_shoulder, w, h):
    """วิเคราะห์สะบักโดยใช้ตำแหน่ง landmark ไหล่เป็น reference
    — แบ่งโซนซ้าย/ขวาด้วย x กึ่งกลางไหล่จริง ไม่พึ่ง mask"""
    sx_l = int(left_shoulder.x * w)
    sx_r = int(right_shoulder.x * w)
    sy   = int((left_shoulder.y + right_shoulder.y)/2 * h)
    cx   = (sx_l + sx_r) // 2

    # โซนสะบัก: ใต้ไหล่ 10%-55% ของความสูงภาพ
    h_img = image_bgr.shape[0]
    y_top    = sy + int(0.05 * h_img)
    y_bottom = sy + int(0.35 * h_img)
    y_top    = max(0, min(y_top, h_img-1))
    y_bottom = max(0, min(y_bottom, h_img-1))
    if y_bottom - y_top < 20:
        return None

    region = image_bgr[y_top:y_bottom, :, :]
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h_r, w_r = gray.shape
    cx_r = min(cx, w_r-1)

    def detrended_peak(col_data):
        n = len(col_data)
        if n < 6: return 0, 0.0, 0.0
        x = np.arange(n, dtype=np.float32)
        coeffs = np.polyfit(x, col_data, 1)
        detail = col_data - np.polyval(coeffs, x)
        margin = max(1, n//8)
        seg = detail[margin:n-margin]
        if len(seg) == 0: return 0, 0.0, 0.0
        peak_idx = int(np.argmax(seg)) + margin
        sig = float(detail[peak_idx]) / (float(np.std(detail)) or 1e-6)
        return peak_idx, float(col_data[peak_idx]), sig

    left_profile  = gray[:, :cx_r].mean(axis=1)
    right_profile = gray[:, cx_r:].mean(axis=1)

    l_row, l_val, l_sig = detrended_peak(left_profile)
    r_row, r_val, r_sig = detrended_peak(right_profile)

    min_sig = 1.2
    if l_sig < min_sig and r_sig < min_sig:
        return None

    diff  = abs(l_val - r_val)
    side  = "Left" if l_val > r_val else "Right"
    h_diff = abs(l_row - r_row) / max(h_r, 1)
    h_side = "Left" if l_row < r_row else "Right" if r_row < l_row else "-"

    return {"detected": True, "prominence_diff": diff, "prominent_side": side,
            "height_diff_ratio": h_diff, "higher_side": h_side,
            "left_pt": (cx_r//2, y_top+l_row),
            "right_pt": (cx_r + (w_r-cx_r)//2, y_top+r_row)}


def find_waist_landmark_based(image_bgr, left_shoulder, right_shoulder,
                               left_hip, right_hip, hips_visible, w, h):
    """หาเอวจากขอบลำตัวในโซนระหว่างไหล่-สะโพก (ใช้ landmark เป็น reference)
    — ไม่ต้องใช้ mask เลย แค่สแกนแถวหา x ที่ pixel สว่างกว่า threshold"""
    sx_l = int(left_shoulder.x * w)
    sx_r = int(right_shoulder.x * w)
    sy   = int((left_shoulder.y + right_shoulder.y)/2 * h)
    cx   = (sx_l + sx_r) // 2
    sh_w = abs(sx_r - sx_l)

    if hips_visible:
        hy = int((left_hip.y + right_hip.y)/2 * h)
    else:
        hy = min(sy + int(0.5 * h), h - 10)

    span = hy - sy
    if span < 20:
        return None

    y_start = sy + int(0.15 * span)
    y_end   = hy - int(0.12 * span)
    if y_end - y_start < 15:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    left_xs, right_xs = {}, {}
    for y in range(y_start, y_end):
        row = gray[y, :].astype(np.float32)
        # threshold แบบ adaptive: ใช้ค่าเฉลี่ยของ 2 มุม * 0.85
        bg_l = float(row[:max(1, sx_l-sh_w//3)].mean()) if sx_l > sh_w//3 else 30.0
        bg_r = float(row[min(w-1, sx_r+sh_w//3):].mean()) if sx_r+sh_w//3 < w else 30.0
        th = (bg_l + bg_r) / 2 * 0.85 + 10

        # หาขอบซ้าย (จาก cx ออกไปซ้ายจนเจอ background)
        for x in range(cx, max(0, cx - sh_w), -1):
            if row[x] < th:
                left_xs[y] = x
                break

        # หาขอบขวา
        for x in range(cx, min(w, cx + sh_w)):
            if row[x] < th:
                right_xs[y] = x
                break

    if len(left_xs) < 10 or len(right_xs) < 10:
        return None

    # เอวซ้าย = จุดที่ขอบซ้ายเว้าเข้ามาขวาสุด (x มากสุด)
    ly = max(left_xs, key=lambda y: left_xs[y])
    # เอวขวา = จุดที่ขอบขวาเว้าเข้ามาซ้ายสุด (x น้อยสุด)
    ry = min(right_xs, key=lambda y: right_xs[y])

    if abs(ly - ry) / span > 0.2:
        return None  # จุดเอวสองข้างอยู่คนละระดับเกินไป

    return (float(left_xs[ly]), float(ly)), (float(right_xs[ry]), float(ry))


# ─── วิเคราะห์หลัก ────────────────────────────────────────────────────────────

def analyze_standing(image_bgr):
    h, w, _ = image_bgr.shape
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result  = detector.detect(mp_img)

    if not result.pose_landmarks:
        return None, None

    # ── Step A: สร้าง mask สำหรับแสดงผลเท่านั้น ──────────────────────────────
    person_mask = get_clean_mask(result, h, w)
    bg_removed  = image_bgr.copy()
    bg_removed[~person_mask] = 0
    contrast_img = apply_high_contrast(bg_removed, person_mask)
    mask_vis     = np.zeros((h, w), dtype=np.uint8)
    mask_vis[person_mask] = 255

    # ── Step B: ดึง landmark ──────────────────────────────────────────────────
    lm          = result.pose_landmarks[0]
    nose        = lm[0]
    left_ear    = lm[7];   right_ear    = lm[8]
    left_sh     = lm[11];  right_sh     = lm[12]
    left_el     = lm[13];  right_el     = lm[14]
    left_hip    = lm[23];  right_hip    = lm[24]
    left_knee   = lm[25];  right_knee   = lm[26]

    # ── Step C: Shoulder slope ────────────────────────────────────────────────
    dx_s = (right_sh.x - left_sh.x) * w
    dy_s = (right_sh.y - left_sh.y) * h
    shoulder_slope  = abs(dy_s/dx_s) if dx_s != 0 else 0
    shoulder_dir    = "right_up" if dy_s<0 else "left_up" if dy_s>0 else "level"

    mid_sx = int((left_sh.x+right_sh.x)/2*w)
    mid_sy = int((left_sh.y+right_sh.y)/2*h)
    sh_w   = abs(int(right_sh.x*w) - int(left_sh.x*w))

    # ── Step D: Hip slope ─────────────────────────────────────────────────────
    hip_vis_score = min(getattr(left_hip,"visibility",1.0), getattr(right_hip,"visibility",1.0))
    hips_visible  = hip_vis_score >= HIP_VISIBILITY_THRESHOLD

    if hips_visible:
        dx_h = (right_hip.x-left_hip.x)*w
        dy_h = (right_hip.y-left_hip.y)*h
        hip_slope  = abs(dy_h/dx_h) if dx_h!=0 else 0
        hip_dir    = "right_up" if dy_h<0 else "left_up" if dy_h>0 else "level"
        if hip_slope > HIP_SLOPE_MAX_PLAUSIBLE or abs(dx_h) < HIP_WIDTH_MIN_RATIO*abs(dx_s):
            hips_visible = False
            hip_slope = hip_dir = None
        else:
            mid_hx = int((left_hip.x+right_hip.x)/2*w)
            mid_hy = int((left_hip.y+right_hip.y)/2*h)
    else:
        hip_slope = hip_dir = None

    if not hips_visible:
        mid_hx, mid_hy = mid_sx, h-10

    # Trunk tilt
    if hips_visible:
        dx_t = mid_sx - mid_hx; dy_t = mid_sy - mid_hy
        trunk_tilt = math.degrees(math.atan2(abs(dx_t),abs(dy_t))) if dy_t!=0 else 90.0
        same_dir   = shoulder_dir == hip_dir and shoulder_dir != "level"
    else:
        trunk_tilt = None
        same_dir   = False

    # ── Step E: Spine chain ───────────────────────────────────────────────────
    neck_pt    = estimate_neck(nose, left_ear, right_ear, mid_sx, mid_sy, w, h)
    spine_pts  = build_spine_chain(neck_pt, (mid_sx,mid_sy), (mid_hx,mid_hy), n=3)

    # ── Step F: Waist (landmark-based, no mask) ───────────────────────────────
    waist_res = find_waist_landmark_based(image_bgr, left_sh, right_sh,
                                           left_hip, right_hip, hips_visible, w, h)

    if waist_res:
        lw, rw      = waist_res
        dx_w        = rw[0]-lw[0]; dy_w = rw[1]-lw[1]
        waist_slope = abs(dy_w/dx_w) if dx_w!=0 else 0
        waist_dir   = "right_up" if dy_w<0 else "left_up" if dy_w>0 else "level"
        waist_ok    = True
    else:
        waist_slope = waist_dir = None
        waist_ok    = False

    # ── Step G: Scapula (landmark-based, no mask) ─────────────────────────────
    scap_res = find_scapula_landmark_based(image_bgr, left_sh, right_sh, w, h)

    # ── Step H: วาดภาพผลลัพธ์ ─────────────────────────────────────────────────
    annotated = image_bgr.copy()
    y_knee    = int((left_knee.y+right_knee.y)/2*h)
    xlbl      = w - 140

    draw_dashed_line(annotated, mid_sy, 10, w-10)
    if hips_visible:
        draw_dashed_line(annotated, mid_hy, 10, w-10)

    if hips_visible:
        draw_zone_label(annotated,"Zone1: Shoulder",(mid_sy+mid_hy)//2,(200,100,30),xlbl)
        draw_zone_label(annotated,"Zone2: Hip",     (mid_hy+y_knee)//2,(30,80,200), xlbl)
    else:
        draw_zone_label(annotated,"Zone1: Shoulder",(mid_sy+h)//2,     (200,100,30),xlbl)

    # เส้นไหล่
    cv2.line(annotated,
             (int(left_sh.x*w), int(left_sh.y*h)),
             (int(right_sh.x*w),int(right_sh.y*h)),
             (0,200,255), 3, cv2.LINE_AA)

    # เส้นสะโพก
    if hips_visible:
        cv2.line(annotated,
                 (int(left_hip.x*w), int(left_hip.y*h)),
                 (int(right_hip.x*w),int(right_hip.y*h)),
                 (0,200,255), 3, cv2.LINE_AA)

    # จุดไหล่/สะโพก
    for lmark in ([left_sh, right_sh] + ([left_hip, right_hip] if hips_visible else [])):
        cv2.circle(annotated,(int(lmark.x*w),int(lmark.y*h)),8,(0,0,255),-1)

    # spine
    spine_info = draw_spine_chain(annotated, spine_pts)

    # waist
    if waist_ok:
        lw, rw = waist_res
        cv2.circle(annotated,(int(lw[0]),int(lw[1])),6,(255,0,255),-1)
        cv2.circle(annotated,(int(rw[0]),int(rw[1])),6,(255,0,255),-1)
        cv2.line(annotated,(int(lw[0]),int(lw[1])),(int(rw[0]),int(rw[1])),(255,0,255),2,cv2.LINE_AA)
        cv2.putText(annotated,"Waist",
                    (int(lw[0])-10,int(min(lw[1],rw[1]))-12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,255),1,cv2.LINE_AA)

    # scapula
    if scap_res and scap_res.get("detected"):
        lp = scap_res["left_pt"]; rp = scap_res["right_pt"]
        cv2.circle(annotated,lp,6,(0,255,120),-1)
        cv2.circle(annotated,rp,6,(0,255,120),-1)
        cv2.line(annotated,lp,rp,(0,255,120),2,cv2.LINE_AA)
        cv2.putText(annotated,
                    f"Scapula:{scap_res['prominent_side']}({scap_res['prominence_diff']:.1f})",
                    (10, min(lp[1]+20, h-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,120),1,cv2.LINE_AA)

    if not hips_visible:
        cv2.putText(annotated,"Hip: not in frame",(10,h-15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

    return {
        "shoulder_slope":       shoulder_slope,
        "shoulder_tilt_dir":    shoulder_dir,
        "hip_slope":            hip_slope,
        "hip_tilt_dir":         hip_dir,
        "trunk_tilt_angle":     trunk_tilt,
        "same_direction":       same_dir,
        "hips_visible":         hips_visible,
        "waist_detected":       waist_ok,
        "waist_slope":          waist_slope,
        "waist_tilt_dir":       waist_dir,
        "scapula_detected":     bool(scap_res and scap_res.get("detected")),
        "scapula_diff":         scap_res["prominence_diff"] if scap_res and scap_res.get("detected") else 0.0,
        "scapula_side":         scap_res["prominent_side"]  if scap_res and scap_res.get("detected") else "-",
        "scapula_height_ratio": scap_res["height_diff_ratio"] if scap_res and scap_res.get("detected") else 0.0,
        "scapula_higher_side":  scap_res["higher_side"]      if scap_res and scap_res.get("detected") else "-",
        "spine_dev_ratio":      spine_info["max_dev_ratio"],
        "spine_dev_dir":        spine_info["max_dev_dir"],
        "spine_joint_devs":     spine_info["joint_devs"],
        "person_mask":          person_mask,
        "bg_removed":           bg_removed,
        "contrast_img":         contrast_img,
        "mask_vis":             mask_vis,
        "arms_natural":         True,
        "arm_warn_sides":       [],
    }, annotated


# ─── Risk helpers ──────────────────────────────────────────────────────────────

def get_risk_level_default(value, lo, hi):
    if value is None: return "—", "gray"
    return ("ต่ำ (ปกติ)","green") if value<lo else ("ปานกลาง","orange") if value<hi else ("สูง","red")

def load_baseline():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    return None

def get_risk_level_baseline(value, mean, sd):
    if value is None or sd==0: return "—","gray"
    d = abs(value-mean); sd = max(sd,1e-6)
    return ("ต่ำ (ปกติ)","green") if d<=sd else ("ปานกลาง","orange") if d<=2*sd else ("สูง","red")

RECOMMENDATIONS = {
    "ต่ำ (ปกติ)": "ท่าทางของคุณอยู่ในเกณฑ์ปกติ ควรรักษาท่าทางที่ดีต่อไป และออกกำลังกายยืดเหยียดเป็นประจำ",
    "ปานกลาง":    "พบความเอียงเล็กน้อย แนะนำให้สังเกตท่าทางตนเองในชีวิตประจำวัน และฝึกบริหารกล้ามเนื้อหลังและไหล่ หากมีอาการผิดปกติควรพบแพทย์",
    "สูง":         "พบความเอียงค่อนข้างมาก แนะนำให้พบแพทย์หรือนักกายภาพบำบัดเพื่อตรวจวินิจฉัยเพิ่มเติม ระบบนี้เป็นเพียงการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์",
}

SLOPE_LOW_UI, SLOPE_HIGH_UI = 0.05, 0.15

baseline = load_baseline()

# ─── UI ────────────────────────────────────────────────────────────────────────

st.sidebar.title("เมนู")
mode = st.sidebar.radio("เลือกโหมด",[
    "ประเมินผล",
    "Calibration (สร้างเกณฑ์จากกลุ่มปกติ)",
])

# ── Calibration mode ────────────────────────────────────────────────────────────
if mode == "Calibration (สร้างเกณฑ์จากกลุ่มปกติ)":
    st.title("📊 Calibration: สร้างเกณฑ์จากกลุ่มคนหลังตรง")
    files = st.file_uploader("เลือกภาพถ่าย (หลายรูปได้)",
                              type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files:
        rows=[]; hip_v=[]; waist_v=[]; scap_v=[]
        for f in files:
            fb   = np.asarray(bytearray(f.read()),dtype=np.uint8)
            img  = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            res, _ = analyze_standing(img)
            if res is None:
                st.warning(f"ไม่พบร่างกายในภาพ: {f.name}"); continue
            row = {
                "filename":        f.name,
                "shoulder_slope":  round(res["shoulder_slope"],4),
                "spine_dev_ratio": round(res["spine_dev_ratio"],4),
                "hip_slope":       round(res["hip_slope"],4) if res["hips_visible"] else None,
                "waist_slope":     round(res["waist_slope"],4) if res["waist_detected"] else None,
                "scapula_diff":    round(res["scapula_diff"],2) if res["scapula_detected"] else None,
            }
            rows.append(row)
            if res["hips_visible"]: hip_v.append(res["hip_slope"])
            if res["waist_detected"]: waist_v.append(res["waist_slope"])
            if res["scapula_detected"]: scap_v.append(res["scapula_diff"])

        if rows:
            st.table(rows)
            def _s(v): return {"mean":float(np.mean(v)),"sd":float(np.std(v))} if v else None
            stats = {
                "n": len(rows),
                "shoulder_slope":   _s([r["shoulder_slope"]  for r in rows]),
                "spine_dev_ratio":  _s([r["spine_dev_ratio"] for r in rows]),
                "hip_slope":        _s(hip_v),   "n_hip":   len(hip_v),
                "waist_slope":      _s(waist_v), "n_waist": len(waist_v),
                "scapula_diff":     _s(scap_v),
            }
            st.subheader("สถิติสรุป (Mean ± SD)")
            st.write(f"จำนวน: {stats['n']} คน (สะโพก {stats['n_hip']}, เอว {stats['n_waist']})")
            for k,lbl in [("shoulder_slope","Shoulder Slope"),("spine_dev_ratio","Spine Deviation"),
                          ("hip_slope","Hip Slope"),("waist_slope","Waist Slope"),("scapula_diff","Scapula Prominence")]:
                s = stats[k]
                st.write(f"{lbl}: {s['mean']:.4f} ± {s['sd']:.4f}" if s else f"{lbl}: ไม่มีข้อมูลเพียงพอ")
            if st.button("💾 บันทึกเป็นเกณฑ์ (baseline.json)"):
                with open(BASELINE_FILE,"w",encoding="utf-8") as fp:
                    json.dump(stats,fp,ensure_ascii=False,indent=2)
                st.success(f"บันทึกเกณฑ์เรียบร้อย ({stats['n']} ตัวอย่าง)")

# ── ประเมินผล mode ─────────────────────────────────────────────────────────────
else:
    st.title("🦴 ระบบคัดกรองภาวะกระดูกสันหลังคดเบื้องต้น")
    st.write("อัปโหลดภาพถ่ายด้านหลัง (ท่ายืนตรง) เพื่อประเมินความเสี่ยงเบื้องต้น")
    if baseline:
        st.caption(f"✅ ใช้เกณฑ์จากกลุ่มตัวอย่างปกติ ({baseline['n']} คน)")
    else:
        st.caption("⚠️ ยังไม่มีเกณฑ์จากกลุ่มตัวอย่าง — ใช้ค่าเริ่มต้น")

    uf = st.file_uploader("เลือกภาพถ่าย (ท่ายืนตรง)", type=["jpg","jpeg","png"])
    if uf:
        fb  = np.asarray(bytearray(uf.read()),dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)

        with st.spinner("กำลังวิเคราะห์..."):
            res, ann = analyze_standing(img)

        if res is None:
            st.error("ไม่พบร่างกายในภาพ กรุณาลองใหม่")
        else:
            # Processing steps
            with st.expander("🔍 ดูขั้นตอนการประมวลผล (Processing Steps)", expanded=False):
                c1,c2,c3,c4,c5 = st.columns(5)
                with c1:
                    st.caption("1) ภาพต้นฉบับ")
                    st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.caption("2) ตัดพื้นหลังออกจริง")
                    st.image(cv2.cvtColor(res["bg_removed"],cv2.COLOR_BGR2RGB), use_container_width=True)
                with c3:
                    st.caption("3) ขาวดำ Contrast สูงสุด")
                    st.image(cv2.cvtColor(res["contrast_img"],cv2.COLOR_BGR2RGB), use_container_width=True)
                with c4:
                    st.caption("4) Person Mask (ขาว=คน, ดำ=พื้นหลัง)")
                    st.image(res["mask_vis"], use_container_width=True)
                with c5:
                    st.caption("5) จุด Landmark + วิเคราะห์ (ผลลัพธ์สุดท้าย)")
                    st.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption("ขั้นตอน: (1) รับภาพ → (2) ตัดพื้นหลังด้วย MediaPipe Segmentation Mask → "
                           "(3) ขาวดำ Contrast สูง → (4) Person Mask ขาว-ดำ → "
                           "(5) วิเคราะห์จุด Landmark + Waist + Scapula โดยใช้ Landmark เป็น reference ทั้งหมด "
                           "(ไม่พึ่ง Mask สำหรับการวิเคราะห์ เพื่อความแม่นยำสม่ำเสมอ)")

            st.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),
                     caption="จุดแดง=ไหล่/สะโพก | เส้นฟ้า=ไหล่/สะโพก | เส้นเหลือง=แนวกระดูกสันหลัง | จุดม่วง=เอว | จุดเขียว=สะบัก",
                     use_container_width=True)

            # Risk
            bl = baseline
            def risk(key, lo, hi):
                v = res.get(key)
                if v is None: return "—","gray"
                if bl and bl.get(key) and bl[key]:
                    return get_risk_level_baseline(v,bl[key]["mean"],bl[key]["sd"])
                return get_risk_level_default(v,lo,hi)

            sh_r,sh_c = risk("shoulder_slope",SLOPE_LOW_UI,SLOPE_HIGH_UI)
            sp_r,sp_c = risk("spine_dev_ratio",SPINE_DEV_LOW,SPINE_DEV_HIGH)

            c1,c2,c3 = st.columns(3)
            with c1:
                st.metric("Shoulder Slope",f"{res['shoulder_slope']:.4f}")
                st.markdown(f"ความเสี่ยง: :{sh_c}[{sh_r}]")
            with c2:
                if res["hips_visible"]:
                    hr,hc = risk("hip_slope",SLOPE_LOW_UI,SLOPE_HIGH_UI)
                    st.metric("Hip Slope",f"{res['hip_slope']:.4f}")
                    st.markdown(f"ความเสี่ยง: :{hc}[{hr}]")
                else:
                    st.metric("Hip Slope","—")
                    st.markdown("ไม่พบสะโพกในภาพ")
            with c3:
                tt = res["trunk_tilt_angle"]
                st.metric("Trunk Tilt Angle", f"{tt:.2f}°" if tt else "—")

            st.metric("Spine Deviation",f"{res['spine_dev_ratio']:.4f}")
            st.markdown(f"ความเสี่ยง: :{sp_c}[{sp_r}] (เบี่ยง{res['spine_dev_dir']})")

            cw,cs = st.columns(2)
            with cw:
                if res["waist_detected"]:
                    wr,wc = risk("waist_slope",SLOPE_LOW_UI,SLOPE_HIGH_UI)
                    st.metric("Waist Slope",f"{res['waist_slope']:.4f}")
                    st.markdown(f"ความเสี่ยง: :{wc}[{wr}]")
                else:
                    st.metric("Waist Slope","—")
                    st.markdown("ไม่พบแนวเอวในภาพ")
            with cs:
                if res["scapula_detected"]:
                    scr,scc = risk("scapula_diff",SCAPULA_DIFF_LOW,SCAPULA_DIFF_HIGH)
                    st.metric("Scapula Prominence",f"{res['scapula_diff']:.1f}")
                    st.markdown(f"ความเสี่ยง: :{scc}[{scr}]")
                    st.write(f"สะบักด้านที่นูนกว่า: **{res['scapula_side']}**")
                else:
                    st.metric("Scapula Prominence","—")
                    st.markdown("ไม่พบสะบักที่ชัดเจนพอ")

            # curve type
            if not res["hips_visible"]:
                cnote = "ไม่เห็นสะโพกในภาพ ยังไม่สามารถบอกลักษณะการบิดของลำตัวทั้งท่อนได้"
            elif res["same_direction"]:
                cnote = "ไหล่และสะโพกเอียงไปทาง**เดียวกัน** → ลำตัวเอียงทั้งแท่ง (C-curve)"
            elif res["shoulder_tilt_dir"]=="level" or res["hip_tilt_dir"]=="level":
                cnote = "ไม่พบการเอียงที่ชัดเจน"
            else:
                cnote = "ไหล่และสะโพกเอียง**คนละทาง** → อาจมีการบิดของลำตัวสองช่วง (เข้าข่าย S-curve เบื้องต้น)"
            st.write(f"**ลักษณะแนวลำตัว:** {cnote}")

            # Overall risk
            risk_pool = [sh_r, sp_r]
            if res["hips_visible"]: risk_pool.append(hr)
            if res["waist_detected"]: risk_pool.append(wr)
            if res["scapula_detected"]: risk_pool.append(scr)
            valid = [r for r in risk_pool if r != "—"]
            overall = max(valid, key=lambda r: ["ต่ำ (ปกติ)","ปานกลาง","สูง"].index(r)) if valid else "ต่ำ (ปกติ)"
            if res["hips_visible"] and not res["same_direction"] and res["shoulder_tilt_dir"]!="level" and res["hip_tilt_dir"]!="level":
                if overall=="ต่ำ (ปกติ)": overall="ปานกลาง"

            st.subheader(f"ผลการประเมิน: {overall}")
            st.info(RECOMMENDATIONS[overall])
            st.caption("⚠️ ระบบนี้เป็นเครื่องมือคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้")