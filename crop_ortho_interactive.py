import cv2, numpy as np, rasterio as rio
from rasterio.windows import Window

ORTHO = "ortho_monroe_2024.tif"
OUT   = "ortho_zoom.tif"

# --- display limits (fit on screen) ---
MAX_W, MAX_H = 1500, 800   # tweak if needed

def to_8bit(img):
    if img.dtype == np.uint8:
        return img
    o = img.astype(np.float32)
    for c in range(o.shape[2]):
        lo, hi = np.percentile(o[..., c], (1, 99))
        if hi <= lo: hi = lo + 1.0
        o[..., c] = np.clip((o[..., c] - lo) / (hi - lo) * 255.0, 0, 255)
    return o.astype(np.uint8)

with rio.open(ORTHO) as src:
    rgb = src.read([1,2,3]) if src.count >= 3 else np.repeat(src.read(1)[None, ...], 3, 0)
    img = np.moveaxis(rgb, 0, 2)  # (H,W,C)
    disp = to_8bit(img)

H, W = disp.shape[:2]

# scale preview to fit on screen
scale = min(1.0, min(MAX_W / W, MAX_H / H))
disp_sz = (int(W * scale), int(H * scale))
disp_img = cv2.resize(disp, disp_sz, interpolation=cv2.INTER_AREA)

ix = iy = ex = ey = -1
drag = False

def on_mouse(e, x, y, flags, param):
    global ix, iy, ex, ey, drag
    if e == cv2.EVENT_LBUTTONDOWN:
        ix, iy, ex, ey, drag = x, y, x, y, True
    elif e == cv2.EVENT_MOUSEMOVE and drag:
        ex, ey = x, y
    elif e == cv2.EVENT_LBUTTONUP:
        ex, ey, drag = x, y, False

cv2.namedWindow("crop: drag rectangle, 's' save, 'q' quit")
cv2.setMouseCallback("crop: drag rectangle, 's' save, 'q' quit", on_mouse)

while True:
    frame = disp_img.copy()
    if ix >= 0 and ex >= 0:
        cv2.rectangle(frame, (ix, iy), (ex, ey), (0, 255, 255), 2)
    cv2.imshow("crop: drag rectangle, 's' save, 'q' quit", frame)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('q'):
        break
    if k == ord('s') and ix >= 0 and ex >= 0:
        # map display coords -> original pixel coords
        x0d, x1d = sorted([ix, ex]); y0d, y1d = sorted([iy, ey])
        x0 = int(round(x0d / scale))
        x1 = int(round(x1d / scale))
        y0 = int(round(y0d / scale))
        y1 = int(round(y1d / scale))
        wpx = max(1, x1 - x0)
        hpx = max(1, y1 - y0)

        with rio.open(ORTHO) as src:
            win = Window(x0, y0, wpx, hpx)
            data = src.read(window=win)
            transform = src.window_transform(win)
            profile = src.profile.copy()
            profile.update(width=wpx, height=hpx, transform=transform)
            with rio.open(OUT, "w", **profile) as dst:
                dst.write(data)

        print(f"Wrote {OUT}  ({wpx} x {hpx} px)  bounds={transform*(0,0)}..{transform*(wpx,hpx)}")
        break

cv2.destroyAllWindows()
