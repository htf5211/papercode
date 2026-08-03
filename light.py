from __future__ import annotations

import math
from typing import Dict
import cv2

from utils import *
"""
光学平衡特性
    动态范围截断
        过曝区域占比
        过曝强度
        过暗区域占比
        过暗强度
    空间照度不均
        照度场变异系数
    方向性模糊退化
        动态模糊方向各异性
        定向高频损失
"""
# =========================================================
# 一些配套使用的函数
# =========================================================
# 亮度图转换函数，将彩色图换为[0,1]
# 常规的RGB图对于亮度的展示不是太好，所以要根据需要将其转换为hsv或者ycc图，能更好的展示亮度
def _brightness_map01(bgr: np.ndarray, channel: str = "y") -> np.ndarray:
    """
    Return brightness I in [0,1]
    channel:
      - "y": Y channel of YCrCb (recommended)
      - "v": V channel of HSV
    """
    ch = channel.lower()
    if ch == "v":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2].astype(np.float32) / 255.0
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycc[:, :, 0].astype(np.float32) / 255.0


# 统计L10、L90函数
def _l10_l90_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "L10": float(np.percentile(arr, 10)),
        "L90": float(np.percentile(arr, 90)),
    }
# =========================================================
# A) 动态范围截断：过曝欠曝指标
# 这里的阈值是设置的绝对阈值0.90和0.05，是按照经验设置的两个阈值
# TODO 这里的过曝强度和过暗强度是绝对的，数值比较小，可以看看需不需要调整
# =========================================================
def exposure_saturation_dataset(
    images_dir: str | Path,
    t_over: float = 0.90,
    t_under: float = 0.05,
    brightness_channel: str = "y",
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Dataset-level saturation (clipping) metrics:
      - R_over: ratio of pixels >= t_over
      - I_over: mean exceedance over t_over within over region
      - R_under: ratio of pixels <= t_under
      - I_under: mean deficit under t_under within under region

    Output: mean/L10/L90 for each + n_images_valid
    """
    R_over_list, I_over_list = [], []
    R_under_list, I_under_list = [], []

    for img_path in iter_images(images_dir):
        bgr = read_bgr_u8(img_path)
        if bgr is None:
            continue
        I = _brightness_map01(bgr, channel=brightness_channel)

        over = I >= float(t_over)
        under = I <= float(t_under)

        R_over = float(np.mean(over))
        R_under = float(np.mean(under))

        if np.any(over):
            I_over = float(np.mean(I[over] - float(t_over)))
        else:
            I_over = 0.0

        if np.any(under):
            I_under = float(np.mean(float(t_under) - I[under]))
        else:
            I_under = 0.0

        R_over_list.append(R_over)
        I_over_list.append(I_over)
        R_under_list.append(R_under)
        I_under_list.append(I_under)

    R_over_arr = np.asarray(R_over_list, dtype=np.float64)
    I_over_arr = np.asarray(I_over_list, dtype=np.float64)
    R_under_arr = np.asarray(R_under_list, dtype=np.float64)
    I_under_arr = np.asarray(I_under_list, dtype=np.float64)

    sRov = _l10_l90_stats(R_over_arr)
    sIov = _l10_l90_stats(I_over_arr)
    sRun = _l10_l90_stats(R_under_arr)
    sIun = _l10_l90_stats(I_under_arr)

    return {
        "R_over_mean": sRov["mean"], "R_over_L10": sRov["L10"], "R_over_L90": sRov["L90"],
        "I_over_mean": sIov["mean"], "I_over_L10": sIov["L10"], "I_over_L90": sIov["L90"],
        "R_under_mean": sRun["mean"], "R_under_L10": sRun["L10"], "R_under_L90": sRun["L90"],
        "I_under_mean": sIun["mean"], "I_under_L10": sIun["L10"], "I_under_L90": sIun["L90"],
        "n_images_valid": float(len(R_over_arr)),
    }


# =========================================================
# B) 空间照度不均
# 先用高斯函数把图模糊一下，假设其近似于照度场
# 然后求变异系数，用来表示波动程度
# sigamx为高斯分布标准差
# =========================================================
def illumination_nonuniformity_dataset(
    images_dir: str | Path,
    brightness_channel: str = "y",
    sigma: float = 25.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Illumination field non-uniformity per image:
      1) Estimate low-frequency illumination L via Gaussian blur (sigma)
      2) IU = std(L) / (mean(L) + eps)

    Output: mean/L10/L90 + n_images_valid
    """
    iu_list = []

    for img_path in iter_images(images_dir):
        bgr = read_bgr_u8(img_path)
        if bgr is None:
            continue
        I = _brightness_map01(bgr, channel=brightness_channel)  # [0,1]
        L = cv2.GaussianBlur(I.astype(np.float32), (0, 0), sigmaX=float(sigma))
        iu = float(L.std() / (L.mean() + eps))
        iu_list.append(iu)

    arr = np.asarray(iu_list, dtype=np.float64)
    s = _l10_l90_stats(arr)
    return {
        "illum_IU_mean": s["mean"],
        "illum_IU_L10": s["L10"],
        "illum_IU_L90": s["L90"],
        "n_images_valid": float(arr.size),
    }


# =========================================================
# C) 动态模糊各向异性
# 在具体实现中，使用了过曝和欠曝过滤，为了防止极端区域对数据造成污染
# 然后利用sobel统计了各方向的梯度，之后筛选出强梯度
# 再用强梯度计算熵，用来表示各向异性
# =========================================================
# 单图计算各向异性
def _goe_blur_score(
    gray01: np.ndarray,
    t_over: float = 0.90,
    t_under: float = 0.05,
    mag_percentile: float = 70.0,
    n_bins: int = 36,
    eps: float = 1e-12,
) -> float:
    """
    Gradient Orientation Entropy (GOE) blur score:
      - compute gradient orientations in valid brightness region (exclude over/under)
      - select strong gradients by magnitude percentile
      - entropy H of orientation histogram
      - score = 1 - H_norm in [0,1], higher => more direction-concentrated => more motion blur
    """
    I = gray01.astype(np.float32)
    valid = (I < float(t_over)) & (I > float(t_under))
    if float(np.mean(valid)) < 0.05:
        return 0.0

    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
    # 梯度强度
    mag = np.sqrt(gx * gx + gy * gy)
    # 梯度方向
    ang = (np.arctan2(gy, gx) + np.pi)  # [0,2pi)

    mag_v = mag[valid]
    if mag_v.size == 0:
        return 0.0

    # 筛选强梯度，参与后续计算，防止弱梯度（噪声）干扰
    thr = float(np.percentile(mag_v, float(mag_percentile)))
    # 得到一个布尔矩阵
    sel = valid & (mag >= thr)
    if float(np.mean(sel)) < 0.01:
        return 0.0

    # 利用方向的熵来展现各向异性
    # 这两步先计算梯度方向的离散概率分布，将方向分为36份，再计算不同方向的分布情况
    ang_sel = ang[sel]
    hist, _ = np.histogram(ang_sel, bins=int(n_bins), range=(0, 2 * np.pi), density=False)
    p = hist.astype(np.float32) / (float(hist.sum()) + eps)

    # 计算熵
    H = float(-np.sum(p * np.log(p + eps)))
    H_norm = float(H / (math.log(float(n_bins)) + eps))
    return float(np.clip(1.0 - H_norm, 0.0, 1.0))


# 数据集级别的统计
def motion_blur_anisotropy_goe_dataset(
    images_dir: str | Path,
    t_over: float = 0.90,
    t_under: float = 0.05,
    mag_percentile: float = 70.0,
    n_bins: int = 36,
) -> Dict[str, float]:
    """
    Dataset-level motion blur anisotropy (GOE):
      Blur_GOE in [0,1], higher => stronger directional blur tendency

    Output: mean/L10/L90 + n_images_valid
    """
    blur_list = []

    for img_path in iter_images(images_dir):
        gray01 = read_gray01(img_path)
        if gray01 is None:
            continue
        blur_list.append(
            _goe_blur_score(
                gray01,
                t_over=t_over,
                t_under=t_under,
                mag_percentile=mag_percentile,
                n_bins=n_bins,
            )
        )

    if not blur_list:
        return {"Blur_GOE_mean": float("nan"), "Blur_GOE_L10": float("nan"), "Blur_GOE_L90": float("nan"), "n_images_valid": 0.0}

    arr = np.asarray(blur_list, dtype=np.float64)
    s = _l10_l90_stats(arr)
    return {
        "Blur_GOE_mean": s["mean"],
        "Blur_GOE_L10": s["L10"],
        "Blur_GOE_L90": s["L90"],
        "n_images_valid": float(arr.size),
    }


# =========================================================
# D) 定向高频损失
# 上一个指标用于判定是否存在动态模糊，这个指标用于判断动态模糊强度
# 计算逻辑很简单，单图自动计算边缘的最大模糊方向，然后计算损失
# =========================================================
# 把某个角度转换为方向向量
def _unit_vec(theta_deg: float) -> Tuple[float, float]:
    th = np.deg2rad(float(theta_deg))
    return float(np.cos(th)), float(np.sin(th))


# 计算图像在方向u上的二阶方向导数
def _directional_second_derivative(img_f01: np.ndarray, theta_deg: float) -> np.ndarray:
    u0, u1 = _unit_vec(theta_deg)
    Ixx = cv2.Sobel(img_f01, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(img_f01, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(img_f01, cv2.CV_32F, 1, 1, ksize=3)
    return (u0 * u0) * Ixx + (2.0 * u0 * u1) * Ixy + (u1 * u1) * Iyy


# 在给定方向下输出一个方向性高频能量损失分数
def dhfel_dir(gray01: np.ndarray, theta_deg: float, eps: float = 1e-8) -> float:
    """
    DHFEL in [0,1]:
      loss = 1 - E_par / (E_iso + eps)
    higher => stronger directional loss (more anisotropic degradation).
    """
    img_f = gray01.astype(np.float32)
    Iuu = _directional_second_derivative(img_f, theta_deg)
    E_par = float(np.mean(Iuu * Iuu))

    L = cv2.Laplacian(img_f, cv2.CV_32F, ksize=3)
    E_iso = float(np.mean(L * L))

    loss = 1.0 - (E_par / (E_iso + eps))
    return float(np.clip(loss, 0.0, 1.0))


# 估计整张图的主要结构方向
def estimate_dominant_orientation(gray01: np.ndarray, ksize: int = 9, eps: float = 1e-8) -> float:
    """
    Structure-tensor dominant orientation in [0,180).
    """
    I = gray01.astype(np.float32)
    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)

    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    gxx = cv2.GaussianBlur(gx * gx, (k, k), 0)
    gyy = cv2.GaussianBlur(gy * gy, (k, k), 0)
    gxy = cv2.GaussianBlur(gx * gy, (k, k), 0)

    a = float(np.mean(gxx))
    b = float(np.mean(gxy))
    c = float(np.mean(gyy))

    theta = 0.5 * np.arctan2(2.0 * b, (a - c + eps))
    return float(np.rad2deg(theta) % 180.0)


# 自动选择运动模糊方向并计算损失
def dhfel_auto(gray01: np.ndarray) -> float:
    theta_g = estimate_dominant_orientation(gray01)
    theta_motion = (theta_g + 90.0) % 180.0
    return dhfel_dir(gray01, theta_motion)


# 统计数据集级指标
def directional_hf_energy_loss_dataset(
    images_dir: str | Path,
    use_auto: bool = True,
    theta_deg: float = 90.0,
) -> Dict[str, float]:
    """
    Dataset-level DHFEL:
      - use_auto=True: estimate dominant orientation and take orthogonal as motion direction
      - use_auto=False: fixed direction theta_deg (0=horizontal, 90=vertical)

    Output: mean/L10/L90 + n_images_valid
    """
    loss_list = []

    for img_path in iter_images(images_dir):
        gray01 = read_gray01(img_path)
        if gray01 is None:
            continue
        loss = dhfel_auto(gray01) if use_auto else dhfel_dir(gray01, theta_deg=float(theta_deg))
        loss_list.append(loss)

    if not loss_list:
        return {"DHFEL_mean": float("nan"), "DHFEL_L10": float("nan"), "DHFEL_L90": float("nan"), "n_images_valid": 0.0}

    arr = np.asarray(loss_list, dtype=np.float64)
    s = _l10_l90_stats(arr)
    return {
        "DHFEL_mean": s["mean"],
        "DHFEL_L10": s["L10"],
        "DHFEL_L90": s["L90"],
        "n_images_valid": float(arr.size),
    }


def light_metrics(images_dir, labels_dir):
    # ======================================================
    # A) 动态范围截断
    # ======================================================
    print("---- Dynamic Range Clipping (Over/Under Exposure) ----")
    sat = exposure_saturation_dataset(
        images_dir,
        t_over=0.90,
        t_under=0.05,
        brightness_channel="y",
    )
    print(sat, "\n")

    E_over = np.exp(np.mean(np.log([sat["R_over_mean"], sat["I_over_mean"]])))
    E_under = np.exp(np.mean(np.log([sat["R_under_mean"], sat["I_under_mean"]])))
    E_clip = E_over + E_under
    exposure_score = 1.0 - E_clip

    # ======================================================
    # B) 照度场不均
    # ======================================================
    print("---- Illumination Non-uniformity ----")
    iu = illumination_nonuniformity_dataset(
        images_dir,
        brightness_channel="y",
        sigma=25.0,
    )
    print(iu, "\n")
    illum_score = 1.0 - iu["illum_IU_mean"]
    # TODO 各向异性值偏低，要么是这个值本来就偏低，要么就是那个强弱边缘筛选的阈值选的不合适，后续试验下
    # ======================================================
    # C) 动态模糊方向各向异性
    # ======================================================
    print("---- Motion Blur Anisotropy (GOE) ----")
    goe = motion_blur_anisotropy_goe_dataset(
        images_dir,
        t_over=0.90,
        t_under=0.05,
        mag_percentile=70.0,
        n_bins=36,
    )
    print(goe, "\n")

    # ======================================================
    # D) 定向高频能量损失
    # ======================================================
    print("---- Directional High-Frequency Energy Loss (DHFEL) ----")
    dhfel = directional_hf_energy_loss_dataset(
        images_dir,
        use_auto=True,
        theta_deg=90.0,
    )
    print(dhfel, "\n")
    eps = 1e-12  # 防止log(0)
    g = goe["Blur_GOE_mean"]
    d = dhfel["DHFEL_mean"]

    geo_mean = np.exp((np.log(g + eps) + np.log(d + eps)) / 2.0)

    blur_score = 1.0 - geo_mean

    # ======================================================
    # 计算总得分
    # ======================================================
    optical_score = float(np.exp(np.mean(np.log(np.clip([
        exposure_score,
        illum_score,
        blur_score,
    ], 1e-6, 1.0)))))

    print("exposure_score:", exposure_score)
    print("illum_score:", illum_score)
    print("blur_score:", blur_score)
    print("optical_score:", optical_score)

    return optical_score

