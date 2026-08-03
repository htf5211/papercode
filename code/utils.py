from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# -----------------------------
# Image listing
# -----------------------------
# 读入的图像类型
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 可单独使用每次返回一个图像名称，也可与下面的list_images一起使用，返回整个路径中的图像名称列表
def iter_images(images_dir):
    # 将传入的路径转换为Path对象，以使用对应的功能
    p = Path(images_dir)

    # 对图像排序，并筛选出文件（排除文件夹）和读入的图像类型
    files = sorted(
        fp for fp in p.glob("*")
        if fp.is_file() and fp.suffix.lower() in exts
    )
    yield from files

# 返回路径中所有文件列表（排序过的）
def list_images(images_dir):
    return list(iter_images(images_dir))
# -----------------------------
# Image reading
# -----------------------------
# 返回灰度图
def read_gray_u8(img_path):
    """Read grayscale image as uint8 [0,255]."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return img

# 返回归一化的灰度图，用于一些图像处理算法
def read_gray01(img_path):
    """Read grayscale image as float32 in [0,1]."""
    g = read_gray_u8(img_path)
    return g.astype(np.float32) / 255.0

# 读取三通道彩色图
def read_bgr_u8(img_path) :
    """Read BGR image as uint8 [0,255]."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    return img

# 确保BGR图像是[0,255]，不是的话就转化
def to_uint8_bgr(bgr):
    """
    Ensure uint8 BGR in [0,255].
    Accepts float images in [0,1] or [0,255].
    """
    # 已经是的话，就不用转，直接返回
    if bgr.dtype == np.uint8:
        return bgr
    # 否则先转化为浮点数，再转换为0-255
    x = bgr.astype(np.float32)
    if x.size and x.max() <= 1.0:
        x *= 255.0
    x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)

# 若图像最长边超过long_side，则按比例缩放
# 根据是否需要加速，判断是否启用
def maybe_resize_long_side_u8(img_u8, long_side):
    """
    Resize keeping aspect ratio so that max(H,W) == long_side (if larger).
    If long_side is None or <=0 or image already small enough, return original.
    """
    if long_side is None or long_side <= 0:
        return img_u8
    h, w = img_u8.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img_u8
    scale = long_side / float(m)
    nh, nw = max(2, int(round(h * scale))), max(2, int(round(w * scale)))
    return cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_AREA)
# -----------------------------
# Basic ops
# -----------------------------
# 计算图像Laplacian绝对值均值，用于衡量高频强度，输入灰度图，采用经典Lap核大小3
def mean_abs_lap(img01):
    """Mean absolute Laplacian on grayscale [0,1] float32."""
    lap = cv2.Laplacian(img01.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap)))

# 先缩小再放大图像，用于模拟重采样过程，scale是缩放比例，采用双线性插值法
def resample(img01, scale, interp):
    """Downsample then upsample back to original size."""
    h, w = img01.shape[:2]
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(img01, (sw, sh), interpolation=interp)
    recon = cv2.resize(small, (w, h), interpolation=interp)
    return recon

# 将一个浮点数四舍五入为整数，并限制在指定区间内
def clamp_int(x, lo, hi):
    """Clamp and round float to int within [lo, hi]."""
    return int(max(lo, min(hi, round(x))))
# -----------------------------
# YOLO labels & boxes
# -----------------------------
# 读取YOLO标签（原始标签是归一化值的）
def read_yolo_labels(txt_path):
    """
    Read YOLO txt label file. Return list of (cls, cx, cy, w, h) normalized.
    """
    p = Path(txt_path)
    if not p.exists():
        return []
    # 得到列表组成的标签内容
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    out: List[Tuple[int, float, float, float, float]] = []
    for line in lines:
        # 去掉行内空白
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        out.append((cls, cx, cy, w, h))
    return out

# 将YOLO的（cx,cy,w,h）转化为（x1,y1,x2,y2），YOLO读取的数值是归一化的，所以需要一点处理；浮点数形式
def yolo_to_xyxy_float(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    """
    YOLO normalized (cx,cy,w,h) -> pixel xyxy (float), clamped to [0,W]/[0,H].
    Uses half-open box convention: x in [0,W], y in [0,H].
    """
    bw = w * W
    bh = h * H
    x1 = cx * W - bw / 2.0
    y1 = cy * H - bh / 2.0
    x2 = cx * W + bw / 2.0
    y2 = cy * H + bh / 2.0
    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    return x1, y1, x2, y2

# 将YOLO的（cx,cy,w,h）转化为（x1,y1,x2,y2），YOLO读取的数值是归一化的，所以需要一点处理；整数形式
def yolo_to_xyxy_int(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    """
    YOLO normalized (cx,cy,w,h) -> pixel xyxy (int), clamped.
    x1,y1 in [0,W-1]/[0,H-1], x2,y2 in [1,W]/[1,H] to avoid negative width/height.
    """
    x1f, y1f, x2f, y2f = yolo_to_xyxy_float(cx, cy, w, h, W, H)
    x1 = clamp_int(x1f, 0, max(0, W - 1))
    y1 = clamp_int(y1f, 0, max(0, H - 1))
    x2 = clamp_int(x2f, 1, max(1, W))
    y2 = clamp_int(y2f, 1, max(1, H))
    # ensure valid box
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2

# 计算交并比
def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU for float xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-12))