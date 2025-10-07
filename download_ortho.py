# download_ortho.py
# Downloads a GeoTIFF from Monroe County Orthos 2024 ImageServer for your bbox.
# Handles both direct TIFF and JSON-with-href responses.

import os, sys, json, requests, urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

IMAGESERVER = "https://mcgis4.monroecounty-fl.gov/public/rest/services/Images/Orthos2024/ImageServer"

# Your bbox (minLon, minLat, maxLon, maxLat)
BBOX   = (-81.753611, 24.568889, -81.749722, 24.570556)

# Output spatial reference:
#  - 4326 = WGS84 lon/lat
#  - 32617 = UTM 17N (meters)  ← good for distance/speed
OUT_SR = 32617

SIZE   = (3500, 3500)           # server may clamp; 3500 is usually safe
OUT    = "ortho_monroe_2024.tif"

def fetch_tiff(url, verify, out_path):
    """Download a URL that should be a TIFF, with magic-byte check."""
    r = requests.get(url, stream=True, timeout=120, verify=verify)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        for ch in r.iter_content(1 << 20):
            if ch:
                f.write(ch)
    with open(tmp, "rb") as f:
        magic = f.read(4)
    if magic not in (b"II*\x00", b"MM\x00*"):
        os.remove(tmp)
        raise RuntimeError(f"Downloaded file is not a TIFF (magic={magic!r})")
    os.replace(tmp, out_path)
    return out_path

def export_once(verify=True):
    params = {
        "bbox": f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}",
        "bboxSR": 4326,
        "outSR": OUT_SR,
        "size": f"{SIZE[0]},{SIZE[1]}",
        "format": "tiff",
        "pixelType": "U8",
        "noData": 0,
        "f": "image",  # ask for image; some servers reply with JSON containing an href
    }
    url = f"{IMAGESERVER}/exportImage"

    print("Requesting:", url)
    print("Params:", params, "\nverify:", verify)

    r = requests.get(url, params=params, stream=True, timeout=120, verify=verify)
    ctype = r.headers.get("Content-Type", "")

    # Case A: server streamed the TIFF directly
    if "image/tiff" in ctype or "application/octet-stream" in ctype:
        print("Server returned TIFF directly.")
        return fetch_tiff(r.url, verify, OUT)

    # Case B: server responded with JSON (even though f=image)
    # Read text and try to parse JSON for an 'href' to the actual TIFF
    text = r.text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try again explicitly with f=json to see the message
        rj = requests.get(url, params={**params, "f": "json"}, timeout=60, verify=verify)
        try:
            data = rj.json()
        except Exception:
            print("Unexpected response, first 1200 chars:\n", text[:1200])
            raise SystemExit(1)

    href = data.get("href")
    if not href:
        print("No 'href' in server JSON, response snippet:\n", str(data)[:1200])
        raise SystemExit(1)

    print("Server returned JSON with href to TIFF:\n", href)
    return fetch_tiff(href, verify, OUT)

def main():
    try:
        path = export_once(verify=True)
    except requests.exceptions.SSLError:
        print("\nTLS verify failed — retrying INSECURELY...\n")
        path = export_once(verify=False)

    print(f"Saved {path}  ({os.path.getsize(path)/1e6:.2f} MB)")
    print("Done. Tip: if OUT_SR=32617, the raster is in meters (nice for distances).")

if __name__ == "__main__":
    main()
