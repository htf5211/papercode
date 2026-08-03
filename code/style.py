from __future__ import annotations

from typing import Dict, Union
import os
import cv2

from utils import *

PathLike = Union[str, os.PathLike, Path]

"""
风格一致特性
    颜色一致性
    调子一致性
    噪声压缩一致性
    纹理一致性
"""
# =========================
# 工具函数
# =========================
# 计算偏度
def _skew(x: np.ndarray) -> float:
    x = x.astype(np.float32).ravel()
    mu = float(x.mean())
    sig = float(x.std() + 1e-12)
    return float(((x - mu) ** 3).mean() / (sig ** 3))


'''
    一致性的指标都是统计均值和方差之类的离散度
    只不过由于数据格式不同，有两种计算离散度的方式而已
'''
# 计算数据集离散度
def _dataset_dispersion(vecs: np.ndarray) -> float:
    """
    Mean L2 distance to dataset mean. Lower => more consistent.
    vecs: (N,D)
    """
    m = vecs.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(vecs - m, axis=1).mean())


# 计算标量序列的离散度
def _mae_to_mean(x: np.ndarray) -> float:
    """
    Mean absolute deviation to mean. Lower => more consistent.
    x: (N,)
    """
    mu = float(x.mean())
    return float(np.mean(np.abs(x - mu)))


# =========================
# 单图特征提取
# =========================
# 提取颜色风格向量（6维）
# 先转换为LAB（L,A,B）格式，然后分别计算LAB格式的均值和方差，形成6维向量
def extract_color_vec_lab_mean_std(bgr_u8: np.ndarray) -> np.ndarray:
    """
    Color style feature vector: Lab channel mean + std (6D).
    """
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)  # (6,)


# 提取调子风格向量（3维）
# 同样转换为LAB格式，只要L，计算均值、方差、偏度，形成3维向量
def extract_tone_vec_L_mean_std_skew(bgr_u8: np.ndarray) -> np.ndarray:
    """
    Tone style feature vector: L-channel mean/std/skew (3D).
    """
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]  # 0..255
    return np.array([L.mean(), L.std(), _skew(L)], dtype=np.float32)  # (3,)


# 计算基础LBP
# 一种计算纹路的经典方法，想了解的话查一下就行
def _lbp_codes_u8(gray_u8: np.ndarray) -> np.ndarray:
    """
    Basic LBP (P=8, R=1), output code in [0,255], shape (H-2,W-2).
    """
    g = gray_u8.astype(np.uint8)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)

    # neighbors (clockwise from top-left)
    code |= ((g[:-2, :-2] >= c) << 7).astype(np.uint8)
    code |= ((g[:-2, 1:-1] >= c) << 6).astype(np.uint8)
    code |= ((g[:-2, 2:] >= c) << 5).astype(np.uint8)
    code |= ((g[1:-1, 2:] >= c) << 4).astype(np.uint8)
    code |= ((g[2:, 2:] >= c) << 3).astype(np.uint8)
    code |= ((g[2:, 1:-1] >= c) << 2).astype(np.uint8)
    code |= ((g[2:, :-2] >= c) << 1).astype(np.uint8)
    code |= ((g[1:-1, :-2] >= c) << 0).astype(np.uint8)
    return code


# 对上述获得的LBP矩阵，计算直方图，并计算概率
def extract_texture_hist_lbp(gray_u8: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Texture statistical feature: LBP histogram (L1-normalized).
    """
    if gray_u8.shape[0] < 3 or gray_u8.shape[1] < 3:
        h = np.zeros((bins,), dtype=np.float32)
        h[0] = 1.0
        return h

    codes = _lbp_codes_u8(gray_u8)
    hist = np.bincount(codes.ravel(), minlength=bins).astype(np.float32)
    hist /= float(hist.sum() + 1e-12)
    return hist  # (256,)


# 计算拉普拉斯方差
def extract_sharpness_lapvar(gray_f32: np.ndarray) -> float:
    """
    Sharpening style proxy: Laplacian variance (LapVar).
    """
    lap = cv2.Laplacian(gray_f32, ddepth=cv2.CV_32F, ksize=3)
    return float(lap.var())


# 计算 高频残差能量一致性
# noise_sigma:高斯滤波器的标准差，控制滤波强度，1.2是根据经验设置的
# 高斯卷积核大小，如果不指定，就自动算一个
def extract_noise_residual_energy(
    gray_f32: np.ndarray,
    *,
    noise_sigma: float = 1.2,
    noise_ksize: int = 0,
) -> float:
    """
    Noise/compression proxy: residual energy after Gaussian blur.
    """
    if noise_ksize is None or noise_ksize <= 0:
        # 经典的自动计算卷积核大小的公式
        k = int(round(noise_sigma * 6.0 + 1.0))
        if k % 2 == 0:
            k += 1
        noise_ksize = max(3, k)

    blur = cv2.GaussianBlur(gray_f32, ksize=(noise_ksize, noise_ksize), sigmaX=noise_sigma)
    resid = gray_f32 - blur
    return float((resid ** 2).mean())


# 计算压缩一致性
# 经典的JPEG格式图片一般是以8*8像素块为基础单位进行压缩的
# 若块边界跳变显著，提示存在较强 JPEG 块伪影（通常与低质量压缩或多次压缩相关），用于评估压缩风格一致性。
# 所以计算一下每隔8位的块与相邻块的差距，若差距较大，则说明有较大压缩
def extract_blockiness_score(gray_u8: np.ndarray, block: int = 8) -> float:
    """
    JPEG blockiness proxy: mean abs diff across 8x8 block boundaries.
    """
    g = gray_u8.astype(np.float32)
    h, w = g.shape
    if h < block + 1 or w < block + 1:
        return 0.0

    cols = np.arange(block, w, block)
    v = np.abs(g[:, cols] - g[:, cols - 1]).mean() if cols.size > 0 else 0.0

    rows = np.arange(block, h, block)
    hdiff = np.abs(g[rows, :] - g[rows - 1, :]).mean() if rows.size > 0 else 0.0

    return float(0.5 * (v + hdiff))


# 将计算得到的值转化为0-1.且越接近1一致性越好
def inconsistency_to_score(x: float) -> float:
    """
    Lower-is-better inconsistency -> higher-is-better score in (0,1].
    Uses log compression to handle very different raw scales.
    """
    x = max(float(x), 0.0)
    return float(1.0 / (1.0 + np.log1p(x)))
# =========================
# 数据集级指标函数
# =========================
# 计算颜色一致性
def metric_color_consistency(
    img_paths: List[Path],
    *,
    resize_long_side: int = 640,
) -> float:
    feats = []
    for p in img_paths:
        bgr = to_uint8_bgr(read_bgr_u8(p))
        bgr = maybe_resize_long_side_u8(bgr, resize_long_side)
        feats.append(extract_color_vec_lab_mean_std(bgr))
    vecs = np.vstack(feats).astype(np.float32)  # (N,6)
    return _dataset_dispersion(vecs)


# 计算调子一致性
def metric_tone_consistency(
    img_paths: List[Path],
    *,
    resize_long_side: int = 640,
) -> float:
    feats = []
    for p in img_paths:
        bgr = to_uint8_bgr(read_bgr_u8(p))
        bgr = maybe_resize_long_side_u8(bgr, resize_long_side)
        feats.append(extract_tone_vec_L_mean_std_skew(bgr))
    vecs = np.vstack(feats).astype(np.float32)  # (N,3)
    return _dataset_dispersion(vecs)


# 计算纹理一致性（LBP）
def metric_texture_consistency_lbp(
    img_paths: List[Path],
    *,
    resize_long_side: int = 640,
) -> float:
    feats = []
    for p in img_paths:
        bgr = to_uint8_bgr(read_bgr_u8(p))
        bgr = maybe_resize_long_side_u8(bgr, resize_long_side)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        feats.append(extract_texture_hist_lbp(gray, bins=256))
    vecs = np.vstack(feats).astype(np.float32)  # (N,256)
    return _dataset_dispersion(vecs)


# 计算噪声/压缩一致性
def metric_noise_compression_consistency(
    img_paths: List[Path],
    *,
    resize_long_side: int = 640,
    noise_sigma: float = 1.2,
    noise_ksize: int = 0,
    include_blockiness: bool = True,
) -> Dict[str, float]:
    """
    Returns:
      - C_noise: residual-energy dispersion
      - C_blk  : (optional) blockiness dispersion
    """
    noise_vals: List[float] = []
    blk_vals: List[float] = []

    for p in img_paths:
        bgr = to_uint8_bgr(read_bgr_u8(p))
        bgr = maybe_resize_long_side_u8(bgr, resize_long_side)
        gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_f32 = gray_u8.astype(np.float32)

        noise_vals.append(
            extract_noise_residual_energy(gray_f32, noise_sigma=noise_sigma, noise_ksize=noise_ksize)
        )
        if include_blockiness:
            blk_vals.append(extract_blockiness_score(gray_u8, block=8))

    n = np.asarray(noise_vals, dtype=np.float32)
    out = {"C_noise": _mae_to_mean(n)}

    if include_blockiness:
        b = np.asarray(blk_vals, dtype=np.float32)
        out["C_blk"] = _mae_to_mean(b)

    return out


# 计算锐化一致性
# def metric_sharpening_style_consistency(
#     img_paths: List[Path],
#     *,
#     resize_long_side: int = 640,
# ) -> float:
#     vals = []
#     for p in img_paths:
#         bgr = to_uint8_bgr(read_bgr_u8(p))
#         bgr = maybe_resize_long_side_u8(bgr, resize_long_side)
#         gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#         vals.append(extract_sharpness_lapvar(gray_u8.astype(np.float32)))
#     s = np.asarray(vals, dtype=np.float32)
#     # print(min(vals), np.median(vals), max(vals))
#     return _mae_to_mean(s)


# =========================
# 主入口
# =========================
def style_consistency(
    img_dir: PathLike,
    *,
    resize_long_side: int = 640,
    noise_sigma: float = 1.2,
    noise_ksize: int = 0,
    include_blockiness: bool = True,
):
    """
    Main entry: compute style consistency metrics from an image directory.

    Returns (lower => more consistent):
      - C_color
      - C_tone
      - C_tex
      - C_noise
      - C_blk (optional)
      - C_sharp
    """
    img_paths = list_images(img_dir)

    out: Dict[str, float] = {"C_color": metric_color_consistency(img_paths, resize_long_side=resize_long_side),
                             "C_tone": metric_tone_consistency(img_paths, resize_long_side=resize_long_side),
                             "C_tex": metric_texture_consistency_lbp(img_paths, resize_long_side=resize_long_side)}

    nc = metric_noise_compression_consistency(
        img_paths,
        resize_long_side=resize_long_side,
        noise_sigma=noise_sigma,
        noise_ksize=noise_ksize,
        include_blockiness=include_blockiness,
    )
    out.update(nc)

    # out["C_sharp"] = metric_sharpening_style_consistency(img_paths, resize_long_side=resize_long_side)

    print("\n==== Style Consistency Result ====")
    for k, v in out.items():
        print(f"{k:10s}: {v:.6f}")
    print("==================================\n")

    # 将指标值转化为0-1
    # ---- individual scores ----
    color_score = inconsistency_to_score(out["C_color"])
    tone_score = inconsistency_to_score(out["C_tone"])
    texture_score = inconsistency_to_score(out["C_tex"])
    noise_score = inconsistency_to_score(out["C_noise"])
    # sharp_score = inconsistency_to_score(out["C_sharp"])

    # ---- optional blockiness ----
    if "C_blk" in out:
        block_score = inconsistency_to_score(out["C_blk"])
        compression_score = float(np.exp(np.mean(np.log(np.clip([
            noise_score,
            block_score
        ], 1e-6, 1.0)))))
    else:
        compression_score = noise_score

    # ---- final style score ----
    style_score = float(np.exp(np.mean(np.log(np.clip([
        color_score,
        tone_score,
        texture_score,
        compression_score,
        # sharp_score
    ], 1e-6, 1.0)))))

    # 输出分数
    print("\n========== Style Consistency Scores ==========\n")
    print("Color Consistency Score:")
    print(color_score, "\n")

    print("Tone Consistency Score:")
    print(tone_score, "\n")

    print("Texture Consistency Score:")
    print(texture_score, "\n")

    print("compression_score:")
    print(compression_score, "\n")

    # print("Sharpening Consistency Score:")
    # print(sharp_score, "\n")

    print("style_score:")
    print(style_score, "\n")

    print("==============================================\n")

    return style_score


