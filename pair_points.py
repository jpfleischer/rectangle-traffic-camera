# pair_points_zoom.py
# CAMERA (top) is static. ORTHO (bottom) is zoomable (+/-) and pannable (arrows/WASD).
# Click CAM then the matching point in ORTHO. 'u' undo, 's' solve/save, 'q' quit.

import cv2
import numpy as np
import rasterio as rio

CAM_PATH   = "camera_frame.png"
ORTHO_PATH = "ortho_zoom.tif"
H_OUT      = "H_cam_to_map.npy"
WARP_OUT   = "camera_warped_to_ortho.png"

# ---------------- load camera ----------------
cam_bgr = cv2.imread(CAM_PATH)
if cam_bgr is None:
    raise SystemExit(f"Couldn't read {CAM_PATH}")
cam_h, cam_w = cam_bgr.shape[:2]

# ---------------- load ortho (GeoTIFF -> 8-bit RGB) ----------------
with rio.open(ORTHO_PATH) as src:
    nodata = src.nodata
    bands = [1,2,3] if src.count >= 3 else [1]
    arr = src.read(bands)            # (C,H,W)
    arr = np.moveaxis(arr, 0, 2)     # (H,W,C)
    if arr.ndim == 2:
        arr = np.stack([arr,arr,arr], 2)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.dtype != np.uint8:
        o = arr.astype(np.float32)
        mask = np.any(arr == nodata, axis=2) if nodata is not None else None
        for c in range(o.shape[2]):
            ch = o[..., c]
            vals = ch[~mask] if mask is not None else ch
            if vals.size == 0:
                lo, hi = 0, 255
            else:
                lo, hi = np.percentile(vals, (1, 99))
                if hi <= lo: hi = lo + 1.0
            o[..., c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
        ortho_rgb = o.astype(np.uint8)
    else:
        ortho_rgb = arr

ortho_h, ortho_w = ortho_rgb.shape[:2]

# ---------------- layout + state ----------------
MAX_W, MAX_H = 1500, 900  # window cap; tweak to your screen

def fit_size(w, h, max_w, max_h):
    s = min(max_w / w, max_h / h, 1.0)
    return int(w * s), int(h * s), s

# camera shown at fixed reasonable size
cam_wd, cam_hd, cam_scale = fit_size(cam_w, cam_h, MAX_W, MAX_H // 2)
cam_disp = cv2.resize(cam_bgr, (cam_wd, cam_hd), interpolation=cv2.INTER_AREA)

# ORTHO interactive view state
base_wd, base_hd, base_scale = fit_size(ortho_w, ortho_h, MAX_W, MAX_H // 2)
zoom = 2.0      # start with 2x zoom so features are bigger
pan_x = 0       # pan offsets in *resized* pixels (after base_scale*zoom)
pan_y = 0

# interaction state
cam_pts = []   # original camera pixels (x,y)
map_pts = []   # original ortho pixels  (x,y)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def make_ortho_view():
    """Return (view_bgr, view_w, view_h, eff_scale) and clamp pan."""
    global pan_x, pan_y
    eff_scale = base_scale * zoom
    big_w = max(1, int(round(ortho_w * eff_scale)))
    big_h = max(1, int(round(ortho_h * eff_scale)))
    big = cv2.resize(ortho_rgb, (big_w, big_h), interpolation=cv2.INTER_LINEAR)
    big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)

    view_w = min(MAX_W, big_w)
    view_h = min(MAX_H // 2, big_h)

    pan_x = max(0, min(pan_x, big_w - view_w))
    pan_y = max(0, min(pan_y, big_h - view_h))

    view = big[pan_y:pan_y+view_h, pan_x:pan_x+view_w].copy()
    return view, view_w, view_h, eff_scale

def disp_to_ortho_px(x_disp, y_disp, eff_scale):
    x_big = pan_x + x_disp
    y_big = pan_y + y_disp
    x_px = int(round(x_big / eff_scale))
    y_px = int(round(y_big / eff_scale))
    x_px = max(0, min(ortho_w - 1, x_px))
    y_px = max(0, min(ortho_h - 1, y_px))
    return x_px, y_px

# canvas and ROIs (recomputed each redraw)
sep = 16
ortho_view, ortho_vw, ortho_vh, eff_scale = make_ortho_view()
canvas_h = cam_hd + sep + ortho_vh
canvas_w = max(cam_wd, ortho_vw)
canvas = np.full((canvas_h, canvas_w, 3), 20, np.uint8)

def rois():
    cam_roi   = (0, 0, cam_wd, cam_hd)
    ortho_roi = (0, cam_hd + sep, ortho_vw, ortho_vh)
    return cam_roi, ortho_roi

def draw_canvas():
    global ortho_view, ortho_vw, ortho_vh, eff_scale
    ortho_view, ortho_vw, ortho_vh, eff_scale = make_ortho_view()
    ch = cam_hd + sep + ortho_vh
    cw = max(cam_wd, ortho_vw)
    canv = np.full((ch, cw, 3), 20, np.uint8)
    canv[0:cam_hd, 0:cam_wd] = cam_disp
    canv[cam_hd+sep:cam_hd+sep+ortho_vh, 0:ortho_vw] = ortho_view
    return canv

def add_mark(img, pt, idx, color):
    cv2.circle(img, pt, 5, color, -1)
    cv2.putText(img, str(idx), (pt[0]+6, pt[1]-6), FONT, 0.6, color, 1, cv2.LINE_AA)

def redraw():
    canv = draw_canvas()
    cam_roi, ortho_roi = rois()
    # re-draw markers in display coordinates
    for i, (x,y) in enumerate(cam_pts, 1):
        add_mark(canv, (int(x * cam_scale), int(y * cam_scale)), i, (0,255,0))
    for i, (x,y) in enumerate(map_pts, 1):
        x_big = int(round(x * eff_scale)); y_big = int(round(y * eff_scale))
        x_disp = x_big - pan_x
        y_disp = y_big - pan_y
        if 0 <= x_disp < ortho_vw and 0 <= y_disp < ortho_vh:
            add_mark(canv, (x_disp + ortho_roi[0], y_disp + ortho_roi[1]), i, (0,200,255))

    tmp = canv.copy()
    msg = f"Pairs: {min(len(cam_pts), len(map_pts))} | Zoom:{zoom:.2f} | Pan:({pan_x},{pan_y}) | Keys: +/- zoom, arrows/WASD pan, u undo, s solve, q quit"
    cv2.putText(tmp, msg, (20, 30), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow(WIN, tmp)

# mouse callback (set once)
def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    cam_roi, ortho_roi = rois()
    cx, cy, cw, ch = cam_roi
    ox, oy, ow, oh = ortho_roi
    # camera click?
    if cx <= x < cx+cw and cy <= y < cy+ch:
        x_px = int((x - cx) / cam_scale)
        y_px = int((y - cy) / cam_scale)
        cam_pts.append([x_px, y_px])
        return
    # ortho click?
    if ox <= x < ox+ow and oy <= y < oy+oh:
        x_px, y_px = disp_to_ortho_px(x - ox, y - oy, eff_scale)
        map_pts.append([x_px, y_px])
        return

WIN = "Pair points (top=CAM, bottom=ORTHO)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, min(MAX_W, cam_wd), min(MAX_H, cam_hd + sep + ortho_vh))
cv2.setMouseCallback(WIN, on_mouse)

# ---------------- main loop ----------------
while True:
    redraw()
    k = cv2.waitKey(30) & 0xFF

    if k == ord('q'):
        break
    # Zoom
    if k in (ord('+'), ord('=')):  # zoom in
        zoom = min(16.0, zoom * 1.25)
    if k == ord('-'):              # zoom out
        zoom = max(0.25, zoom / 1.25)
    # Pan
    step = 50
    if k in (81, ord('a')): pan_x -= step   # left
    if k in (83, ord('d')): pan_x += step   # right
    if k in (82, ord('w')): pan_y -= step   # up
    if k in (84, ord('s')): pan_y += step   # down

    # Undo
    if k == ord('u'):
        if len(cam_pts) > len(map_pts):
            cam_pts.pop()
        elif len(map_pts) > len(cam_pts):
            map_pts.pop()
        elif cam_pts:
            cam_pts.pop(); map_pts.pop()

    # Solve
    if k == ord('s'):
        n = min(len(cam_pts), len(map_pts))
        if n < 4:
            print("Need at least 4 matched pairs.")
            continue
        cam_arr = np.array(cam_pts[:n], dtype=np.float32)
        map_arr = np.array(map_pts[:n], dtype=np.float32)
        H, inliers = cv2.findHomography(cam_arr, map_arr, cv2.RANSAC, 3.0)
        if H is None:
            print("Homography failed; try more / better-spread points.")
            continue
        np.save(H_OUT, H)
        print(f"Saved H to {H_OUT}. Inliers {int(inliers.sum())}/{n}")

        bev = cv2.warpPerspective(cam_bgr, H, (ortho_w, ortho_h), flags=cv2.INTER_LINEAR)
        cv2.imwrite(WARP_OUT, bev)
        print(f"Wrote {WARP_OUT}")
        overlay = cv2.addWeighted(cv2.cvtColor(ortho_rgb, cv2.COLOR_RGB2BGR), 0.6, bev, 0.4, 0)
        cv2.imshow("Warp preview (overlay)", overlay)

cv2.destroyAllWindows()
