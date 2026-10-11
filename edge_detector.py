import json
import math
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "yolov8n.onnx"
IMG_SIZE = 640
CPU_THREADS = 2
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100

# --- LiDAR settings ---------------------------------------------------------
LIDAR_PATH_DEFAULT = "image.npy"          # override: python edge_detector.py img.jpg other.npy
LIDAR_CANDIDATES = ["image.npy", "compressed_data.npy"]   # inspected at start-up (diagnostics only)
DEPTH_SCALE = 1.0                         # metres maps only: raw value * DEPTH_SCALE = metres (0.001 if mm)

# How the depth map encodes distance. "auto" picks from the array itself:
#   uint8 (0..255)        -> treated as an 8-bit depth image: metres = value / 255 * LIDAR_MAX_RANGE_M
#   float with max <= 1.0 -> treated as normalised depth:      metres = value * LIDAR_MAX_RANGE_M
#   anything else         -> already metres:                   metres = value * DEPTH_SCALE
# Force a mode with "metres", "uint8" or "normalized" if auto guesses wrong.
LIDAR_ENCODING = "auto"
LIDAR_MAX_RANGE_M = 20.0                  # !! SET THIS to the real max range your depth map was generated with
LIDAR_INVERTED = False                    # True if BRIGHT/LARGE values mean NEAR (inverse-depth maps)
SEARCH_RADII = (1, 3, 6, 12)              # LiDAR cells: 3x3 first, then widen if every cell is empty

options = ort.SessionOptions()
options.intra_op_num_threads = CPU_THREADS
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=options,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
print("Provider:", session.get_providers())
print("Input:", session.get_inputs()[0].shape)
print("Output:", session.get_outputs()[0].shape)


def preprocess(image):
    h, w = image.shape[:2]
    scale = min(IMG_SIZE / w, IMG_SIZE / h)

    nw = round(w * scale)
    nh = round(h * scale)

    resized = cv2.resize(image, (nw, nh))

    left = round((IMG_SIZE - nw) / 2 - 0.1)
    top = round((IMG_SIZE - nh) / 2 - 0.1)
    right = IMG_SIZE - nw - left
    bottom = IMG_SIZE - nh - top

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    tensor = np.ascontiguousarray(tensor)

    transform = {
        "scale": scale,
        "pad_x": left,
        "pad_y": top,
        "width": w,
        "height": h,
    }
    return tensor, transform


def postprocess(output, transform):
    pred = np.asarray(output[0])

    if pred.ndim != 3 or pred.shape[0] != 1:
        raise ValueError(f"Unexpected output shape: {pred.shape}")

    pred = pred[0]

    # YOLOv8 commonly outputs (84, 8400).
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.shape[1] != 84:
        raise ValueError(
            f"Expected 84 columns for COCO YOLOv8n; got {pred.shape}"
        )

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    valid = (
        np.isfinite(scores)
        & np.isfinite(boxes_xywh).all(axis=1)
        & (scores >= CONF_THRESHOLD)
    )

    boxes_xywh = boxes_xywh[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if len(boxes_xywh) == 0:
        return []

    scale = transform["scale"]
    px = transform["pad_x"]
    py = transform["pad_y"]
    ow = transform["width"]
    oh = transform["height"]

    boxes = []

    for cx, cy, bw, bh in boxes_xywh:
        x1 = (cx - bw / 2 - px) / scale
        y1 = (cy - bh / 2 - py) / scale
        x2 = (cx + bw / 2 - px) / scale
        y2 = (cy + bh / 2 - py) / scale

        x1 = max(0, min(int(x1), ow - 1))
        y1 = max(0, min(int(y1), oh - 1))
        x2 = max(x1 + 1, min(int(x2), ow))
        y2 = max(y1 + 1, min(int(y2), oh))

        boxes.append([x1, y1, x2 - x1, y2 - y1])

    kept = []

    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]

        selected = cv2.dnn.NMSBoxes(
            [boxes[i] for i in indices],
            scores[indices].tolist(),
            CONF_THRESHOLD,
            IOU_THRESHOLD,
        )

        if len(selected) > 0:
            for j in np.asarray(selected).reshape(-1):
                kept.append(indices[int(j)])

    kept.sort(key=lambda i: float(scores[i]), reverse=True)
    kept = kept[:MAX_DETECTIONS]

    return [
        {
            "class_id": int(class_ids[i]),
            "confidence": float(scores[i]),
            "bbox_xywh": boxes[i],
        }
        for i in kept
    ]


# =============================================================================
# LiDAR loading + diagnostics
# =============================================================================
def describe_npy(path):
    """Print a one-line health report for a .npy file (diagnostics only, never raises)."""
    if not os.path.exists(path):
        print(f"  {path}: not found")
        return
    try:
        a = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False))
        finite = np.isfinite(a)
        n_valid = int(np.count_nonzero(finite & (a > 0)))
        max_val = a[finite].max() if finite.any() else float("nan")
        print(f"  {path}: shape={a.shape} dtype={a.dtype} max={max_val} "
              f"valid(>0)={n_valid}/{a.size} ({100.0 * n_valid / max(a.size, 1):.1f}%)")
    except Exception as e:
        print(f"  {path}: could not be read as a plain array ({type(e).__name__}: {e})")


def load_lidar(path):
    """np.load() with a clear error if the file is missing. Returns the RAW array."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LiDAR file not found: {path} (cwd = {os.getcwd()}). "
            f"Pass it explicitly: python edge_detector.py image.jpg path\\to\\depth.npy"
        )
    data = np.load(path, allow_pickle=False)
    if hasattr(data, "files"):                       # it was really an .npz archive
        print(f"{path} is an archive with keys {data.files}; using '{data.files[0]}'")
        data = data[data.files[0]]
    return data


def prepare_depth_map(raw):
    """Raw array -> clean 2-D float32 depth map in metres. Raises with a helpful message if impossible."""
    depth = np.squeeze(np.asarray(raw))              # (1,H,W) / (H,W,1) -> (H,W)
    if depth.ndim != 2:
        hint = ""
        if depth.ndim == 3:
            hint = " (H,W,C) looks like a colour IMAGE, not a depth map - wrong file?"
        elif depth.ndim == 1:
            hint = " 1-D data looks like a compressed byte stream - it must be decompressed/reshaped first."
        raise ValueError(f"LiDAR array must be 2-D (H x W) but has shape {depth.shape}.{hint}")
    # --- convert the stored values to metres --------------------------------
    mode = LIDAR_ENCODING
    if mode == "auto":
        finite = depth[np.isfinite(depth)]
        top = float(finite.max()) if finite.size else 0.0
        if depth.dtype == np.uint8:
            mode = "uint8"
        elif depth.dtype.kind == "f" and 0.0 < top <= 1.0:
            mode = "normalized"
        else:
            mode = "metres"

    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)                 # 0 / NaN / inf = no laser return
    if mode == "uint8":
        norm = d / 255.0
    elif mode == "normalized":
        norm = d
    elif mode == "metres":
        out = d * DEPTH_SCALE
        out[~valid] = 0.0
        print(f"LiDAR encoding: metres (x{DEPTH_SCALE})")
        return out
    else:
        raise ValueError(f"Unknown LIDAR_ENCODING '{LIDAR_ENCODING}'")

    if LIDAR_INVERTED:
        norm = np.clip(1.0 - norm, 0.01, 1.0)        # bright = near; keep result strictly > 0
    out = norm * LIDAR_MAX_RANGE_M
    out[~valid] = 0.0                                # keep "no return" cells empty
    print(f"LiDAR encoding: {mode}{' (inverted)' if LIDAR_INVERTED else ''}, "
          f"0..1 scaled to 0..{LIDAR_MAX_RANGE_M} m  <- check LIDAR_MAX_RANGE_M is right")
    return out


# =============================================================================
# Pixel -> LiDAR lookup (resolution-independent)
# =============================================================================
def camera_to_lidar_px(cx, cy, img_w, img_h, lid_w, lid_h):
    """
    Map a CAMERA pixel to the matching LiDAR cell by proportional scaling, e.g.
    camera 1920x1080 -> LiDAR 640x480: pixel (960, 540) -> cell (320, 240).
    Assumes the LiDAR map covers the same field of view as the camera.
    """
    lx = int(cx * lid_w / img_w)
    ly = int(cy * lid_h / img_h)
    return min(max(lx, 0), lid_w - 1), min(max(ly, 0), lid_h - 1)


def get_hazard_distance(depth_map, center_x, center_y, img_w, img_h):
    """
    Distance (m) to the hazard at camera pixel (center_x, center_y), or None if the LiDAR
    has no valid return nearby.

    1. Scale the camera pixel into LiDAR coordinates (handles any resolution mismatch).
    2. Median of the valid cells in a 3x3 window; if every cell there is empty, widen the
       window step by step (SEARCH_RADII). Valid = finite and > 0 (NaN/inf/0 = no return).
    """
    lid_h, lid_w = depth_map.shape
    lx, ly = camera_to_lidar_px(center_x, center_y, img_w, img_h, lid_w, lid_h)

    for r in SEARCH_RADII:
        patch = depth_map[max(0, ly - r): min(lid_h, ly + r + 1),
                          max(0, lx - r): min(lid_w, lx + r + 1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size:
            return float(np.median(valid))
    return None


def calculate_hazard_coordinates(distance, pixel_x, img_width, rover_x, rover_y, rover_yaw):
    """
    Converts a LiDAR distance and camera pixel into absolute 2D map coordinates.
    Assumes a standard 60-degree horizontal camera Field of View (FOV).
    """
    # 1. Find how far off-center the rock is in the image (value between -1 and 1)
    # Negative means left of center, positive means right.
    normalized_x = (pixel_x - (img_width / 2)) / (img_width / 2)

    # 2. Convert that pixel offset to a real-world angle (60 deg FOV = ~1.047 radians)
    fov_radians = 1.047
    angle_offset = normalized_x * (fov_radians / 2)

    # 3. Combine the robot's current heading with the camera's offset angle
    total_angle = rover_yaw + angle_offset

    # 4. Use trigonometry to project the distance onto the global map
    hazard_x = rover_x + (distance * math.cos(total_angle))
    hazard_y = rover_y + (distance * math.sin(total_angle))

    return round(hazard_x, 3), round(hazard_y, 3)


def save_hazard_to_json(haz_x, haz_y, class_id, conf, distance=0.0):
    filename = "hazards.json"

    new_hazard = {
        "type": f"class_{class_id}",
        "x": haz_x,
        "y": haz_y,
        "confidence": round(conf, 2),
        "t": round(time.time(), 3),                       # detection time, used for data aging on the dashboard
        "sigma": round(0.10 + 0.02 * distance, 2),        # 1-sigma position error (m): placeholder model, grows with range
    }

    # Read existing hazards if the file exists so we don't overwrite them
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new hazard and save it back
    data.append(new_hazard)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def main(image_path, lidar_path=LIDAR_PATH_DEFAULT):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # ------------------------------------------------------------------
    # Load the LiDAR depth map + DEFENSIVE DEBUG PRINTS
    # ------------------------------------------------------------------
    lidar_array = load_lidar(lidar_path)
    print("\n=== LiDAR DEBUG ===")
    print(f"file                : {lidar_path}")
    print(f"lidar_array.shape   : {lidar_array.shape}   (dtype {lidar_array.dtype})")
    print(f"image.shape         : {image.shape}   (H, W, channels)")
    print(f"np.max(lidar_array) : {np.max(lidar_array)}")
    print("candidate files in this folder:")
    for cand in LIDAR_CANDIDATES:
        describe_npy(cand)
    print("===================\n")

    lidar_array = prepare_depth_map(lidar_array)      # -> clean 2-D float32 metres
    img_h, img_w = image.shape[:2]
    lid_h, lid_w = lidar_array.shape

    valid_mask = np.isfinite(lidar_array) & (lidar_array > 0)
    if not valid_mask.any():
        print("!! LiDAR map has NO valid values (all zero / NaN / negative). "
              "This is the wrong file or an empty scan - every hazard will be 'blind'. "
              "Try: python edge_detector.py <image> compressed_data.npy")
    else:
        print(f"LiDAR valid cells: {int(valid_mask.sum())}/{valid_mask.size} ({100.0 * valid_mask.mean():.2f}%), "
              f"range {lidar_array[valid_mask].min():.2f} .. {lidar_array[valid_mask].max():.2f} m")
        if lidar_array[valid_mask].max() > 200:
            print("!! Depths above 200 - the map may be in millimetres. Set DEPTH_SCALE = 0.001.")
    if (img_w, img_h) != (lid_w, lid_h):
        print(f"Resolution mismatch: camera {img_w}x{img_h} vs LiDAR {lid_w}x{lid_h} -> "
              f"pixels will be scaled automatically.")
        if abs(img_w / img_h - lid_w / lid_h) > 0.05:
            print("!! Aspect ratios differ: the LiDAR may be cropped or have a different FOV than the "
                  "camera, so a plain proportional scale can be off. Check how the depth map is aligned.")

    tensor, transform = preprocess(image)

    # Warm up ONNX Runtime.
    for _ in range(5):
        session.run(None, {input_name: tensor})

    start = time.perf_counter()
    output = session.run(None, {input_name: tensor})
    inference_ms = (time.perf_counter() - start) * 1000

    detections = postprocess(output, transform)

    print(f"Inference latency: {inference_ms:.2f} ms")
    print(f"Detections: {len(detections)}")

    rover_x, rover_y, rover_yaw = 0.0, 0.0, 0.0     # TODO: real pose from odometry

    for det in detections:
        print(det)
        x, y, w, h = det["bbox_xywh"]

        # Center pixel of the bounding box (CAMERA coordinates)
        center_x = x + w / 2
        center_y = y + h / 2
        lx, ly = camera_to_lidar_px(center_x, center_y, img_w, img_h, lid_w, lid_h)

        print(f"Pixel X: {center_x:.1f}")
        print(f"Pixel Y: {center_y:.1f}")
        print(f"-> LiDAR cell ({lx}, {ly}) of {lid_w}x{lid_h}")

        distance = get_hazard_distance(lidar_array, center_x, center_y, img_w, img_h)
        if distance is not None:
            print(f"Hazard {det['class_id']} is {distance:.2f} meters away.")

            haz_x, haz_y = calculate_hazard_coordinates(
                distance, center_x, img_w, rover_x, rover_y, rover_yaw
            )
            print(f"Mapped Hazard to Global Coordinates: X={haz_x}, Y={haz_y}")

            save_hazard_to_json(haz_x, haz_y, det['class_id'], det['confidence'], distance)
        else:
            print(f"Hazard {det['class_id']} detected, but LiDAR is blind here "
                  f"(raw value at cell = {lidar_array[ly, lx]}; no valid return within "
                  f"{SEARCH_RADII[-1]} cells).")

        label = f"class={det['class_id']} {det['confidence']:.2f}"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # These run exactly once, after all hazards are processed
    cv2.imwrite("detections.jpg", image)
    print("Saved: detections.jpg")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python edge_detector.py path\\to\\image.jpg [path\\to\\lidar_depth.npy]")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else LIDAR_PATH_DEFAULT)
