from __future__ import annotations
import math
from typing import Dict
import cv2

from utils import *
"""
空间结构特性的具体指标
    几何稳定性
        尺度波动
        空间位置离散性
    空间信息不足
        高频信息不足
        重采样损失
        最小像素尺度
    空间复杂度
        边缘密度
        高频能量占比
        自相似度缺失
"""
# ============================================================
# 1) 几何稳定性
# 对整个数据集中的所有标注框，计算其尺度变异系数
# CV越大，尺度波动越大，几何稳定性越差
# 分数越接近1，几何稳定性越强，越接近0，几何稳定性越差
# ============================================================
def scale_cv_score(images_dir: str | Path, labels_dir: str | Path, eps: float = 1e-12) -> Dict[str, float]:
    """
    数据集级尺度波动（几何稳定性）。

    对数据集中所有 bbox：
        w_px = x2 - x1
        h_px = y2 - y1
        s_i  = sqrt(w*h)

    CV = std(s) / mean(s)
    score = 1 / (1 + CV)

    返回：
      - scale_mu
      - scale_sd
      - scale_cv
      - scale_score
      - n_bboxes
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    s_list = []

    for img_path in iter_images(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_path)

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        H, W = img.shape[:2]

        for (_, cx, cy, bw, bh) in labels:

            x1, y1, x2, y2 = yolo_to_xyxy_float(cx, cy, bw, bh, W, H)

            w_px = float(x2 - x1)
            h_px = float(y2 - y1)

            s = np.sqrt(max(w_px * h_px, 0.0))
            s_list.append(s)

    s_arr = np.asarray(s_list, dtype=np.float64)

    mu = float(np.mean(s_arr))
    sd = float(np.std(s_arr))
    cv = float(sd / (mu + eps))
    score = float(1.0 / (1.0 + cv))

    return {
        "scale_mu": mu,
        "scale_sd": sd,
        "scale_cv": cv,
        "scale_score": score,
        "n_bboxes": float(s_arr.size),
    }
# ============================================================
# 2) 空间位置离散性
# H越大，位置越分散，越不固定
# 分数越接近1，越固定，越接近0，越随机
# grid_m是信息熵计算中分为几块，可以调整
# ============================================================
def position_entropy_score(images_dir: str | Path, labels_dir: str | Path, grid_m: int = 8, eps: float = 1e-12) -> Dict[str, float]:
    """
    数据集级空间位置离散性（基于信息熵）。

    对数据集中所有 bbox：
      1) 计算中心点 (cx_px, cy_px)
      2) 归一化为 (cx/W, cy/H)
      3) 划分为 MxM 网格
      4) 计算信息熵 H
      5) H_norm = H / log(M*M)
      6) pos_score = 1 - H_norm

    含义：
      H 越大 → 位置越分散
      pos_score 越接近 1 → 越集中（越稳定）
      pos_score 越接近 0 → 越随机（越不稳定）

    返回：
      - pos_entropy
      - pos_entropy_norm
      - pos_score
      - n_bboxes
    """

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    M = int(grid_m)

    centers_norm = []

    for img_path in iter_images(images_dir):

        label_path = labels_dir / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_path)

        for (_, cx, cy, bw, bh) in labels:
            cxn = np.clip(cx, 0.0, 1.0)
            cyn = np.clip(cy, 0.0, 1.0)

            centers_norm.append([cxn, cyn])

    c = np.asarray(centers_norm, dtype=np.float64)

    # -------- 网格统计 --------
    gx = np.minimum((c[:, 0] * M).astype(int), M - 1)
    gy = np.minimum((c[:, 1] * M).astype(int), M - 1)

    hist = np.zeros((M, M), dtype=np.float64)
    np.add.at(hist, (gy, gx), 1.0)

    # -------- 信息熵 --------
    p = hist.reshape(-1)
    p = p / (np.sum(p) + eps)

    H_ent = float(-np.sum(p * np.log(p + eps)))
    H_norm = float(H_ent / (math.log(M * M) + eps))
    score = float(1.0 - np.clip(H_norm, 0.0, 1.0))

    return {
        "pos_entropy": H_ent,
        "pos_entropy_norm": H_norm,
        "pos_score": score,
        "n_bboxes": float(c.shape[0]),
    }
# ============================================================
# 3) 高频信息不足
# 计算每个缺陷的高频信息
# 然后兼顾均值与L10值，计算高频信息质量
# 然后直接用常数来归一化，得到评分
# ============================================================
def hf_insufficiency_score(
    images_dir: str | Path,
    labels_dir: str | Path,
    C: float = 1,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    高频信息不足评分（bbox级）

    步骤：
        1. 遍历数据集中所有 bbox
        2. 对每个 bbox 区域计算 L_i = mean(|Lap|)
        3. 计算：
            - L_mean = mean(L_i)
            - L10    = percentile_10(L_i)
        4. 融合得到：
            L_hf = (L_mean + L10) / 2
        5. 使用单调饱和函数转为评分：
            score = L_hf / (L_hf + C)

    参数：
        - C: 控制评分曲线形状的常数，当 L_hf = C 时，score = 0.5

    返回：
        - hf_L_mean
        - hf_L10
        - hf_L_hf
        - hf_score
        - n_bboxes_valid
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    L_list = []

    for img_path in iter_images(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_path)

        img01 = read_gray01(img_path)
        H, W = img01.shape[:2]

        for (_, cx, cy, bw, bh) in labels:
            x1, y1, x2, y2 = yolo_to_xyxy_float(cx, cy, bw, bh, W, H)

            x1 = max(0, int(np.floor(x1)))
            y1 = max(0, int(np.floor(y1)))
            x2 = min(W, int(np.ceil(x2)))
            y2 = min(H, int(np.ceil(y2)))

            # 跳过非法框
            if x2 <= x1 or y2 <= y1:
                continue

            patch = img01[y1:y2, x1:x2]

            # 跳过空区域
            if patch.size == 0:
                continue

            L_i = mean_abs_lap(patch)
            L_list.append(L_i)

    arr = np.asarray(L_list, dtype=np.float64)

    if arr.size == 0:
        return {
            "hf_L_mean": 0.0,
            "hf_L10": 0.0,
            "hf_L_hf": 0.0,
            "hf_score": 0.0,
            "n_bboxes_valid": 0.0,
        }

    L_mean = float(np.mean(arr))
    L10 = float(np.percentile(arr, 10))
    L_hf = float((L_mean + L10) / 2.0)

    score = float(L_hf / (L_hf + C + eps))

    return {
        "hf_L_mean": L_mean,
        "hf_L10": L10,
        "hf_L_hf": L_hf,
        "hf_score": score,
        "n_bboxes_valid": float(arr.size),
    }
# ============================================================
# 4) 重采样信息丢失
# 用于判断那种插值模糊图像
# D-L0尾部越接近于0，插值模糊混入越严重
# ============================================================
def resample_interpolation_blur_L10(
    images_dir: str | Path,
    scale: float = 0.5,
    tau_L0_min: float = 0.005,
) -> Dict[str, float]:
    """
    识别“插值/重采样导致的已模糊图像”的数据集级指标（整图版，加入 L0 过滤）。

    过滤逻辑：
      若 L0 < tau_L0_min，则认为该图高频几乎为零（可能异常/极端退化），跳过，
      避免 Q = L1/L0 数值不稳定污染 D 的尾部统计。

    输出：
      rs_interp_D_L10
      n_images_valid
      n_images_filtered
    """
    D_list = []
    n_filtered = 0

    for img_path in iter_images(images_dir):
        img01 = read_gray01(img_path)
        if img01 is None:
            continue

        L0 = mean_abs_lap(img01)
        if L0 < float(tau_L0_min):
            n_filtered += 1
            continue

        img2 = resample(img01, scale=scale, interp=cv2.INTER_LINEAR)
        L1 = mean_abs_lap(img2)

        Q = float(L1 / (L0 + 1e-12))
        D = float(1.0 - Q)  # 插值模糊图通常 D≈0
        D_list.append(D)

    arr = np.asarray(D_list, dtype=np.float64)
    return {
        "rs_interp_D_mean": float(np.mean(arr)),
        "n_images_valid": float(arr.size),
        "n_images_filtered": float(n_filtered),
    }
# ============================================================
# 5) 最小像素尺度不足
# 这个实现起来很简单，其实核心代码就那么几行，大部分都是为了保持鲁棒性才写了那么多
# 也是越接近1越好了，大部分应该都能到1
# 阈值按经验设置为8像素
# ============================================================
def min_pixel_sufficiency_prob(
    images_dir: str | Path,
    labels_dir: str | Path,
    tau_r: float = 8.0,
    shrink: float = 1.0,
) -> Dict[str, float]:
    """
    对数据集中每个 bbox 计算最小像素尺度：
      r = min(w_px, h_px)

    数据集级指标：
      P(r < tau_r)

    返回：
      - px_prob_r_lt_tau
      - n_bboxes
      - tau_r
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    r_list = []

    for img_path in iter_images(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_path)
        if not labels:
            continue

        # 为了拿到 W,H，这里读一次图像尺寸即可
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape[:2]

        for (_, cx, cy, bw, bh) in labels:
            if shrink < 1.0:
                bw *= shrink
                bh *= shrink

            x1, y1, x2, y2 = yolo_to_xyxy_float(cx, cy, bw, bh, W, H)
            if x2 <= x1 or y2 <= y1:
                continue
            r = float(min(x2 - x1, y2 - y1))
            r_list.append(r)

    if not r_list:
        return {"tau_r": float(tau_r), "px_prob_r_lt_tau": float("nan"), "n_bboxes": 0.0}

    r = np.asarray(r_list, dtype=np.float64)
    p = float(np.mean((r > float(tau_r)).astype(np.float32)))
    return {"tau_r": float(tau_r), "px_prob_r_lt_tau": p, "n_bboxes": float(r.size)}
# ============================================================
# 6) 边缘密度
# 若边缘强度大于high，则为强边缘，保留
# 若边缘强度小于high且大于low，则为弱边缘，与强边缘连接的保留
# 若边缘强度小于low，则为噪声，舍弃
# 使用的阈值100，200是经典阈值
# ============================================================
### 边缘密度
def edge_density_dataset(
    images_dir: str | Path,
    low: int = 100,
    high: int = 200,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    数据集级边缘密度指标

    每张图：
      ED = (#edge pixels) / (H*W)

    higher => more structural clutter

    返回：
      edge_density_mean
      edge_density_L10
      edge_density_L90
      n_images_valid
    """
    ed_list = []

    for img_path in iter_images(images_dir):
        gray = read_gray_u8(img_path)
        if gray is None:
            continue

        edges = cv2.Canny(gray, threshold1=low, threshold2=high)
        ed = float(np.count_nonzero(edges)) / float(
            gray.shape[0] * gray.shape[1] + eps
        )
        ed_list.append(ed)

    arr = np.asarray(ed_list, dtype=np.float64)
    score = 1 - arr.mean()

    return {
        "edge_density_score": float(score),
        "edge_density_mean": float(arr.mean()),
        "edge_density_L10": float(np.percentile(arr, 10)),
        "edge_density_L90": float(np.percentile(arr, 90)),
        "n_images_valid": float(arr.size),
    }
# ============================================================
# 7) 高频能量占比
# 将图片视为H*W个二维正弦基函数的叠加
# 只要尺寸相同的图片，其最大频谱是相同的，这样就具有了跨图像和跨数据集相比的特性
# 噪声一般多在高频，越细碎的结构，频谱越靠外
# rho是最大频谱的半径，0.55是一种经验，但不是约定俗成的，所以要说明一下
# ============================================================
# def high_freq_energy_ratio_dataset(
#     images_dir: str | Path,
#     rho: float = 0.55,
#     use_power: bool = True,
#     target_size: int = 640,
#     eps: float = 1e-12,
# ) -> Dict[str, float]:
#     """
#     数据集级高频能量占比指标（统一 resize 到 target_size × target_size）。
#
#     每张图：
#       1) resize 到固定尺寸
#       2) FFT
#       3) 计算外环高频能量比例
#
#     HF = sum_{r >= rho*rmax} |F|^p / sum_all |F|^p
#
#     higher => more high-frequency texture/noise
#
#     输出：
#       hf_ratio_mean
#       hf_ratio_L10
#       hf_ratio_L90
#       n_images_valid
#     """
#     hf_list = []
#
#     for img_path in iter_images(images_dir):
#         gray01 = read_gray01(img_path)
#         if gray01 is None:
#             continue
#
#         # -------- 统一尺寸 --------
#         if gray01.shape[0] != target_size or gray01.shape[1] != target_size:
#             gray01 = cv2.resize(
#                 gray01,
#                 (target_size, target_size),
#                 interpolation=cv2.INTER_LINEAR
#             )
#
#         # -------- FFT --------
#         f = np.fft.fft2(gray01)
#         fshift = np.fft.fftshift(f)
#         mag = np.abs(fshift)
#         if use_power:
#             mag = mag ** 2
#
#         h, w = gray01.shape
#         cy, cx = h // 2, w // 2
#         yy, xx = np.ogrid[:h, :w]
#         r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
#         rmax = float(np.max(r)) + eps
#
#         hf_mask = r >= (rho * rmax)
#         num = float(np.sum(mag[hf_mask]))
#         den = float(np.sum(mag)) + eps
#
#         hf = num / den
#         hf_list.append(hf)
#
#     arr = np.asarray(hf_list, dtype=np.float64)
#     score = 1 - arr.mean()
#
#     return {
#         "hf_ratio_score": float(score),
#         "hf_ratio_mean": float(arr.mean()),
#         "hf_ratio_L10": float(np.percentile(arr, 10)),
#         "hf_ratio_L90": float(np.percentile(arr, 90)),
#         "n_images_valid": float(arr.size),
#     }
# ============================================================
# 8) 自相似度缺失
# 衡量图像中是否都是类似的图片
# ============================================================
def best_ncc_in_neighborhood(
    gray: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
    radius: int,
    exclude_margin: int = 4,
) -> float:
    """
    取以(x,y)为中心的 ps×ps patch，在半径radius邻域里做 NCC 匹配，取最大值。
    返回 best NCC 映射到 [0,1]（由 [-1,1] -> [0,1]）。
    """
    h, w = gray.shape[:2]
    ps = int(patch_size)
    half = ps // 2

    x1, y1 = x - half, y - half
    x2, y2 = x1 + ps, y1 + ps
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return 0.0

    templ = gray[y1:y2, x1:x2].astype(np.float32)

    nx1 = max(0, x - radius - half)
    ny1 = max(0, y - radius - half)
    nx2 = min(w, x + radius + half)
    ny2 = min(h, y + radius + half)

    roi = gray[ny1:ny2, nx1:nx2].astype(np.float32)
    if roi.shape[0] < ps + 2 or roi.shape[1] < ps + 2:
        return 0.0

    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)  # [-1,1]
    if res.size == 0:
        return 0.0

    # 排除“自己附近”的平凡匹配
    tl_x, tl_y = (x1 - nx1), (y1 - ny1)
    rx = int(np.clip(tl_x, 0, res.shape[1] - 1))
    ry = int(np.clip(tl_y, 0, res.shape[0] - 1))

    ex = int(exclude_margin)
    x_lo, x_hi = max(0, rx - ex), min(res.shape[1], rx + ex + 1)
    y_lo, y_hi = max(0, ry - ex), min(res.shape[0], ry + ex + 1)
    res[y_lo:y_hi, x_lo:x_hi] = -1.0

    best = float(np.max(res))          # [-1,1]
    best01 = 0.5 * (best + 1.0)        # -> [0,1]
    return float(np.clip(best01, 0.0, 1.0))


def _sample_patch_centers(
    h: int,
    w: int,
    n_patches: int,
    patch_size: int,
    seed: int = 0,
) -> np.ndarray:
    """
    在可取 patch 的有效区域内采样中心点 (x,y)，避免越界。
    返回 shape (K,2) int，列为 [x,y]
    """
    ps = int(patch_size)
    half = ps // 2
    if h < ps + 2 or w < ps + 2:
        return np.zeros((0, 2), dtype=np.int32)

    x_min, x_max = half, w - half - 1
    y_min, y_max = half, h - half - 1
    if x_max <= x_min or y_max <= y_min:
        return np.zeros((0, 2), dtype=np.int32)

    rng = np.random.default_rng(seed)
    xs = rng.integers(x_min, x_max + 1, size=n_patches, dtype=np.int32)
    ys = rng.integers(y_min, y_max + 1, size=n_patches, dtype=np.int32)
    return np.stack([xs, ys], axis=1)


def ssi_dataset(
    images_dir: str | Path,
    n_patches: int = 64,
    patch_size: int = 32,
    search_radius: int = 80,
    exclude_margin: int = 4,
    resize_to: int = 640,
    seed: int = 0,
) -> Dict[str, float]:
    """
    数据集级 Self-Similarity Insufficiency (SSI).

    单图：
      对 K 个 patch，计算 best_ncc，然后：
        ssi_img = 1 - mean(best_ncc)
      （best_ncc 越低 -> 越找不到相似块 -> ssi 越大 -> 越“杂乱/缺乏重复结构”）

    数据集：
      对每张图得到 ssi_img，输出 mean/L10/L90

    返回：
      ssi_mean
      ssi_L10
      ssi_L90
      n_images_valid
    """
    ssi_list = []
    img_idx = 0

    for img_path in iter_images(images_dir):
        gray = read_gray_u8(img_path)
        if gray is None:
            continue

        # 统一尺度（可比性 + 控制搜索半径/patch大小的物理意义）
        if resize_to is not None and resize_to > 0:
            if gray.shape[0] != resize_to or gray.shape[1] != resize_to:
                gray = cv2.resize(gray, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)

        h, w = gray.shape[:2]
        centers = _sample_patch_centers(
            h, w, n_patches=n_patches, patch_size=patch_size, seed=(seed + img_idx)
        )
        img_idx += 1
        if centers.shape[0] == 0:
            continue

        ncc_vals = []
        for x, y in centers:
            ncc = best_ncc_in_neighborhood(
                gray=gray,
                x=int(x),
                y=int(y),
                patch_size=patch_size,
                radius=search_radius,
                exclude_margin=exclude_margin,
            )
            ncc_vals.append(ncc)

        if not ncc_vals:
            continue

        ncc_arr = np.asarray(ncc_vals, dtype=np.float64)
        ssi_img = float(1.0 - np.mean(ncc_arr))  # 越大越“自相似不足”
        ssi_list.append(ssi_img)

    if not ssi_list:
        return {
            "ssi_mean": float("nan"),
            "ssi_L10": float("nan"),
            "ssi_L90": float("nan"),
            "n_images_valid": 0.0,
        }

    arr = np.asarray(ssi_list, dtype=np.float64)
    score = 1 - arr.mean()

    return {
        "ssi_score": float(score),
        "ssi_mean": float(arr.mean()),
        "ssi_L10": float(np.percentile(arr, 10)),
        "ssi_L90": float(np.percentile(arr, 90)),
        "n_images_valid": float(arr.size),
    }

def spatial_metrics(images_dir, labels_dir):
    print("\n========== Spatial Structure Metrics ==========\n")
    # ======================================================
    # 几何稳定性（空间稳定性）
    # ======================================================
    scale_metrics = scale_cv_score(images_dir, labels_dir)
    cv_score = scale_metrics["scale_score"]
    print("Scale Variation (CV-score):")
    print(cv_score, "\n")

    pos_metrics = position_entropy_score(images_dir, labels_dir, grid_m=8)
    entropy_score = pos_metrics["pos_score"]
    print("Position Dispersion (Entropy-score):")
    print(entropy_score, "\n")

    stable_scale = np.exp(np.mean(np.log(np.array([cv_score, entropy_score]))))
    print("stable scale:", stable_scale, "\n")
    # ======================================================
    # 空间信息不足
    # ======================================================
    print("---- Spatial Information Insufficiency ----")

    hf_metrics = hf_insufficiency_score(images_dir, labels_dir)
    hf_score = hf_metrics["hf_score"]
    print("High-Frequency Insufficiency :")
    print(hf_score, "\n")

    rs_metrics = resample_interpolation_blur_L10(
        images_dir,
        scale=0.5,
        tau_L0_min=0.005,
    )
    rs_score = rs_metrics["rs_interp_D_mean"]
    print("Resample Interpolation Blur (D-L10):")
    print(rs_score, "\n")

    px_metrics = min_pixel_sufficiency_prob(
        images_dir,
        labels_dir,
        tau_r=8.0,
        shrink=1.0,
    )
    px_score = px_metrics["px_prob_r_lt_tau"]
    print("Minimum Pixel Insufficiency:")
    print(px_score, "\n")

    infro_score = np.exp(np.mean(np.log(np.array([hf_score, rs_score, px_score]))))
    print("infro_scale:", infro_score, "\n")
    # ======================================================
    # 空间复杂度
    # ======================================================
    print("---- Spatial Complexity ----")

    edge_metrics = edge_density_dataset(images_dir)
    edge_score = edge_metrics["edge_density_score"]
    print("Edge Density:")
    print(edge_score, "\n")

    # hf_energy_metrics = high_freq_energy_ratio_dataset(
    #     images_dir,
    #     rho=0.55,
    #     target_size=640,
    # )
    # hf_energy_score = hf_energy_metrics["hf_ratio_score"]
    # print("High-Frequency Energy Ratio:")
    # print(hf_energy_score, "\n")

    ssi_metrics = ssi_dataset(
        images_dir,
        n_patches=64,
        patch_size=32,
        search_radius=80,
        resize_to=640,
    )
    ssi_score = ssi_metrics["ssi_score"]
    print("Self-Similarity Insufficiency:")
    print(ssi_score, "\n")

    # conmplex_scale = np.exp(np.mean(np.log(np.array([edge_score, hf_energy_score, ssi_score]))))
    conmplex_scale = np.exp(np.mean(np.log(np.array([edge_score, ssi_score]))))
    print("conmplex_scale:", conmplex_scale, "\n")

    # ======================================================
    # 空间总得分
    # ======================================================
    spatial_score = np.exp(np.mean(np.log(np.array([stable_scale, infro_score, conmplex_scale]))))
    print("spatial_score:", spatial_score, "\n")
    print("==============================================\n")

    return spatial_score

