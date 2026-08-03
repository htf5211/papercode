from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from utils import *

"""
标注一致性（最终 7 指标）
    类别划分一致性
        1) 类别可分性（Fisher separability in feature space）
        2) 近邻混杂率（kNN impurity）
    标注准则一致性
        3) 结构一致性（Split Consistency; 连通组拆分数 CV）
        4) 内容覆盖一致性（B2; multi-ring 低信息占比的类内离散）
    标注分布一致性
        6) 有效类别数 N_eff_norm
    标注覆盖稳定性
        7) 正样本图比例 P_img
"""
# =========================================================
# 预训练特征提取器：ResNet50 (ImageNet)
# =========================================================
def build_extractor(device: torch.device) -> nn.Module:
    """
    ResNet50 pretrained on ImageNet.
    Output: [B, 2048] embedding (avgpool output).
    """
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    extractor = nn.Sequential(*list(m.children())[:-1]).to(device)  # remove fc
    extractor.eval()
    return extractor


# 输入标准预处理
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# =========================================================
# 样本结构：每个标注实例
# =========================================================
@dataclass
class PatchItem:
    img_path: Path
    cls: int
    bbox_xyxy: Tuple[float, float, float, float]  # pixel coords
    emb: Optional[np.ndarray] = None              # (D,)
    cover_ratio_b2: Optional[float] = None        # B2 content coverage proxy


# =========================================================
# 图像/梯度工具
# =========================================================
# 计算边缘密度，用来在内容覆盖一致性中使用
def _sobel_gradmag(gray01: np.ndarray) -> np.ndarray:
    """
    gray01: float32 in [0,1], shape (H,W)
    Return: gradient magnitude float32, shape (H,W)
    """
    g = gray01.astype(np.float32)
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    gx = (
        kx[0, 0] * gp[:-2, :-2] + kx[0, 1] * gp[:-2, 1:-1] + kx[0, 2] * gp[:-2, 2:] +
        kx[1, 0] * gp[1:-1, :-2] + kx[1, 1] * gp[1:-1, 1:-1] + kx[1, 2] * gp[1:-1, 2:] +
        kx[2, 0] * gp[2:, :-2] + kx[2, 1] * gp[2:, 1:-1] + kx[2, 2] * gp[2:, 2:]
    )
    gy = (
        ky[0, 0] * gp[:-2, :-2] + ky[0, 1] * gp[:-2, 1:-1] + ky[0, 2] * gp[:-2, 2:] +
        ky[1, 0] * gp[1:-1, :-2] + ky[1, 1] * gp[1:-1, 1:-1] + ky[1, 2] * gp[1:-1, 2:] +
        ky[2, 0] * gp[2:, :-2] + ky[2, 1] * gp[2:, 1:-1] + ky[2, 2] * gp[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return mag


# 安全裁剪函数
def _crop_u8(img_u8: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    if x2 <= x1 or y2 <= y1:
        return None
    return img_u8[y1:y2, x1:x2]


# 获取边缘环形掩码，用来在内容覆盖一致性中使用
def _ring_mask(h: int, w: int, t: int) -> Optional[np.ndarray]:
    """
    Create ring mask for a patch of size (h,w) with border thickness t (pixels):
    ring = patch - inner_rect(t).
    """
    if t <= 0:
        return None
    if h <= 2 * t or w <= 2 * t:
        # too small, ring degenerates
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :] = True
    mask[t:h - t, t:w - t] = False
    return mask


# 计算单个标注内容覆盖一致性，及标注框内是否有一段低信息的背景，背景比例是否相近
# TODO 这里使用了相对阈值，即q=0.30，意味着将梯度分布的30%分位数视为低结构区域
# 对于不同像素下的结果，使用稳健聚合（中位数聚合）的方式
# TODO 这个函数有些问题，其实并不能完全证明背景比例，但当同类缺陷进行比较时，可以粗粒度的进行比较
def cover_ratio_b2_from_patch(
    patch_rgb: Image.Image,
    *,
    ring_ws: Tuple[int, ...] = (2, 4, 8, 12),
    q: float = 0.30,
    agg: str = "median",  # "median" or "mean"
) -> float:
    """
    Content coverage consistency (B2):
      - compute gradient magnitude on grayscale
      - for each ring width w: ratio of low-gradient pixels within ring
        (threshold by ring-internal quantile q)
      - aggregate across w by median/mean
    """
    g = np.asarray(patch_rgb.convert("L"), dtype=np.float32) / 255.0  # (H,W)
    mag = _sobel_gradmag(g)

    h, w = mag.shape
    vals: List[float] = []

    for t in ring_ws:
        m = _ring_mask(h, w, t)
        if m is None:
            continue
        ring_mag = mag[m]
        if ring_mag.size < 16:
            continue
        # 获取梯度后30%的阈值
        thr = float(np.quantile(ring_mag, q))
        # TODO 计算低梯度的比例，由于可能存在的重复数值，所以比例不一定为30%
        ratio = float((ring_mag <= thr).mean())
        vals.append(ratio)

    if not vals:
        # fallback: whole patch quantile ratio
        thr = float(np.quantile(mag, q))
        return float((mag <= thr).mean())

    arr = np.asarray(vals, dtype=np.float32)
    if agg == "mean":
        return float(arr.mean())
    return float(np.median(arr))


# =========================================================
# Step 1: 加载标注实例 + 特征 + B2 内容覆盖 proxy
# =========================================================
@torch.no_grad()
# 为多个指标服务的数据预处理函数
# 从数据集中提取所有标注目标的patch，并为每个patch计算两个关键特征：B2内容覆盖率+深度特征
def collect_patches_and_features(
    img_dir: Path,
    lbl_dir: Path,
    *,
    # 对bbox扩展一点再裁剪，以防止有些缺陷完全无背景信息
    pad_ratio: float = 0.10,
    # 最多提取多少个bbox（等于是取样）
    max_patches: Optional[int] = 10000,
    batch_size: int = 64,
    ring_ws: Tuple[int, ...] = (2, 4, 8, 12),
    ring_q: float = 0.30,
    ring_agg: str = "median",
) -> List[PatchItem]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = build_extractor(device)

    items: List[PatchItem] = []
    tensor_buf: List[torch.Tensor] = []
    # 这个索引是为了防止某些patch是空的之类，所以记录一下原始索引的位置
    idx_buf: List[int] = []

    img_paths = list_images(img_dir)
    for ip in img_paths:
        lp = lbl_dir / f"{ip.stem}.txt"
        labels = read_yolo_labels(lp)
        if not labels:
            continue

        img = Image.open(ip).convert("RGB")
        W, H = img.size

        for cls, cx, cy, bw, bh in labels:
            # bbox in pixel
            x1f, y1f, x2f, y2f = yolo_to_xyxy_float(cx, cy, bw, bh, W, H)

            # padded crop (for feature + B2)
            bw_px = (x2f - x1f)
            bh_px = (y2f - y1f)
            x1 = x1f - bw_px * pad_ratio
            y1 = y1f - bh_px * pad_ratio
            x2 = x2f + bw_px * pad_ratio
            y2 = y2f + bh_px * pad_ratio

            x1i = clamp_int(x1, 0, W)
            y1i = clamp_int(y1, 0, H)
            x2i = clamp_int(x2, 0, W)
            y2i = clamp_int(y2, 0, H)

            if x2i <= x1i or y2i <= y1i:
                continue

            patch = img.crop((x1i, y1i, x2i, y2i))
            # 创建patch类
            pi = PatchItem(img_path=ip, cls=cls, bbox_xyxy=(x1f, y1f, x2f, y2f))

            # B2 content coverage proxy (multi-ring on the patch)
            pi.cover_ratio_b2 = cover_ratio_b2_from_patch(
                patch, ring_ws=ring_ws, q=ring_q, agg=ring_agg
            )

            # 保存patch，包括其对象、图像和索引
            items.append(pi)
            tensor_buf.append(_preprocess(patch))
            idx_buf.append(len(items) - 1)

            if max_patches is not None and len(items) >= max_patches:
                break
        if max_patches is not None and len(items) >= max_patches:
            break

    if len(items) < 10:
        raise RuntimeError(
            f"Too few patches extracted: {len(items)}. "
            f"Check img_dir/label_dir or reduce filters."
        )

    # 获取patch对象中剩下的特征
    # batch extract embeddings
    for i in range(0, len(tensor_buf), batch_size):
        x = torch.stack(tensor_buf[i:i + batch_size], dim=0).to(device)
        feat = extractor(x)              # [B, 2048, 1, 1]
        feat = feat.flatten(1)           # [B, 2048]
        z = feat.cpu().numpy().astype(np.float32)

        # L2 normalize => cosine similarity = dot product
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

        for j, emb in enumerate(z):
            # 这里搞了个双重索引，是为了防止某些patch是空的之类
            items[idx_buf[i + j]].emb = emb

    return items


# =========================================================
# 指标 1/2：类别划分一致性（Sep + kNN impurity）
# =========================================================
# 计算类别可分性的经典方法，利用类内散度和类间散度之比来计算，越大表示越可分
def fisher_separability(Z: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    # y是类别数
    classes = np.unique(y)
    # Z是N个样本的D维向量
    mu = Z.mean(axis=0)
    Sw = 0.0
    Sb = 0.0
    for c in classes:
        idx = (y == c)
        Zc = Z[idx]
        if len(Zc) < 2:
            continue
        muc = Zc.mean(axis=0)
        Sw += float(((Zc - muc) ** 2).sum())
        Sb += float(len(Zc) * ((muc - mu) ** 2).sum())
    return float(Sb / (Sw + eps))


# 利用余弦相似度来计算最相似的邻居中，多少是同类别，越高则越差
def knn_impurity(
    Z: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    chunk_size: int = 512,
) -> float:
    n = Z.shape[0]
    if n <= 1:
        return 0.0

    k = int(min(k, n - 1))
    same_sum = 0.0
    count = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Zi = Z[start:end]               # [B, D]
        sim = Zi @ Z.T                  # [B, N]

        # 排除自身匹配
        for local_i, global_i in enumerate(range(start, end)):
            sim[local_i, global_i] = -1.0

        # top-k neighbors
        nn_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

        same = (y[nn_idx] == y[start:end, None]).mean(axis=1)
        same_sum += float(same.sum())
        count += (end - start)

        del Zi, sim, nn_idx, same

    return float(1.0 - same_sum / (count + 1e-12))


# =========================================================
# 指标 3：结构一致性（Split Consistency）
# =========================================================
# 对每张图、每个类别，把所有标注框按 IoU≥阈值连成图，求连通分量大小（一个缺陷区域被拆成了多少个框）
# 最后统计这些大小在全数据集上的变异系数 CV，CV 越大说明拆分规则越不一致。
# 为了能够统计那种离得很近，但是不接触的，先扩框再判断IOU
def split_consistency_cv(
    items: List[PatchItem],
    iou_thr: float = 0.10,
    *,
    expand_ratio: float = 0.10,   # ✅ 新增：扩框比例（相对bbox宽高）
) -> float:
    """
    For each image and each class:
      - build connected components among boxes by IoU>=iou_thr
      - each component size = number of boxes used to annotate one connected defect region
    Metric: CV of component sizes pooled across dataset.

    Note:
      - To handle "gapped" split annotations (boxes do not overlap but are spatially contiguous),
        boxes are expanded by `expand_ratio` before IoU computation.
    """
    from collections import defaultdict

    def _expand_xyxy(box: Tuple[float, float, float, float], r: float) -> Tuple[float, float, float, float]:
        """Expand xyxy box by ratio r (relative to box width/height). No clamping needed here."""
        if r <= 0:
            return box
        x1, y1, x2, y2 = box
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        dx = 0.5 * r * w
        dy = 0.5 * r * h
        return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

    # 按同一张图、同一类别进行分组
    groups = defaultdict(list)
    for it in items:
        groups[(it.img_path, it.cls)].append(it.bbox_xyxy)

    ms: List[int] = []
    for (_, _), boxes in groups.items():
        n = len(boxes)
        if n <= 1:
            ms.append(1)
            continue

        # 预先扩框，避免在双层循环里重复计算
        boxes_e = [_expand_xyxy(b, expand_ratio) for b in boxes]

        # 把相互连通的框归为同一个缺陷区域
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if iou_xyxy(boxes_e[i], boxes_e[j]) >= iou_thr:
                    adj[i].append(j)
                    adj[j].append(i)

        # 图遍历，找连通分量
        vis = [False] * n
        for i in range(n):
            if vis[i]:
                continue
            stack = [i]
            vis[i] = True
            cnt = 0
            while stack:
                u = stack.pop()
                cnt += 1
                for v in adj[u]:
                    if not vis[v]:
                        vis[v] = True
                        stack.append(v)
            ms.append(cnt)

    if len(ms) < 2:
        return 0.0
    # ms，或者说m，表示每个连通区域，被标成了多少个框
    m = np.asarray(ms, dtype=np.float32)
    return float(m.std() / (m.mean() + 1e-12))

# =========================================================
# 指标 4：内容覆盖一致性（B2, 类内离散）
# =========================================================
# 调用前面的函数，计算内容覆盖一致性
def content_coverage_b2_cv(items: List[PatchItem], min_per_class: int = 10) -> float:
    """
    Compute per-class CV of B2 cover ratio proxy, then average over classes with enough samples.
    This measures class-internal stability (not cross-class).
    """
    from collections import defaultdict
    # 按类别分组
    cls2vals = defaultdict(list)
    for it in items:
        if it.cover_ratio_b2 is not None:
            cls2vals[it.cls].append(float(it.cover_ratio_b2))

    # 计算每一类的波动程度（B2是已经计算好存起来，直接传入使用的，这里不用计算）
    cvs: List[float] = []
    for c, vals in cls2vals.items():
        if len(vals) < min_per_class:
            continue
        v = np.asarray(vals, dtype=np.float32)
        cvs.append(float(v.std() / (v.mean() + 1e-12)))

    if not cvs:
        return 0.0
    return float(np.mean(cvs))


# =========================================================
# 指标 5/6：标注分布一致性（H_norm, N_eff_norm）
# =========================================================
# 判断数据集中各类别实例数量是否均衡
# 计算类别信息熵和有效类别数
def class_distribution_metrics(y: np.ndarray) -> Dict[str, float]:
    """
    y: instance labels of extracted items (instance-level distribution).
    Returns:
      - H_norm in [0,1] (if C_present<=1 => 1.0 by convention)
      - N_eff_norm in (0,1] (if C_present<=1 => 1.0)
    """
    # 统计每一类的数量
    classes, counts = np.unique(y, return_counts=True)
    # 计算类别数量
    C = int(len(classes))
    if C <= 1:
        return {"H_norm": 1.0, "N_eff_norm": 1.0, "num_classes_present": float(C)}

    # 计算类别概率
    p = counts.astype(np.float64)
    p /= (p.sum() + 1e-12)

    # 计算信息熵
    H = float(-(p * np.log(p + 1e-12)).sum())
    # 归一化熵
    H_norm = float(H / (math.log(C) + 1e-12))

    # 计算有效类别数
    Neff = float(1.0 / float((p * p).sum() + 1e-12))
    # 归一化有效类别数
    N_eff_norm = float(Neff / C)

    return {"H_norm": H_norm, "N_eff_norm": N_eff_norm, "num_classes_present": float(C)}


# =========================================================
# 指标 7：正样本图比例 P_img
# =========================================================
# 计算数据集中“含有缺陷的图像比例”
def positive_image_ratio(img_dir: Path, lbl_dir: Path) -> float:
    """
    P_img = (#images with >=1 label) / (#images total)
    """
    img_paths = list_images(img_dir)
    if not img_paths:
        return 0.0

    pos = 0
    for ip in img_paths:
        lp = lbl_dir / f"{ip.stem}.txt"
        labels = read_yolo_labels(lp)
        if len(labels) > 0:
            pos += 1
    return float(pos / len(img_paths))


# =========================================================
# 主入口：annotation_consistency_metrics
# =========================================================
def annotation_consistency_metrics(
    img_dir: Path,
    lbl_dir: Path,
    *,
    # feature/patch sampling
    pad_ratio: float = 0.10,
    max_patches: Optional[int] = 10000,
    batch_size: int = 64,
    # kNN
    knn_k: int = 10,
    knn_chunk_size: int = 512,
    # split consistency
    split_iou_thr: float = 0.10,
    # B2 coverage
    ring_ws: Tuple[int, ...] = (2, 4, 8, 12),
    ring_q: float = 0.30,
    ring_agg: str = "median",
    b2_min_per_class: int = 10,
    eps: float = 1e-12
):
    """
    Compute 7 annotation-consistency metrics for a YOLO-format dataset split.
    Returns a flat dict; higher/lower meaning:
      - A_sep: higher => more separable
      - A_impurity: higher => more mixed (worse)
      - Rule_split_CV: higher => split rules less consistent (worse)
      - Rule_cover_B2_CV: higher => boundary tightness less consistent (worse)
      - H_norm: higher => more balanced
      - N_eff_norm: higher => more effective classes
      - P_img: higher => more positive images
    """
    # 7) P_img (image-level)
    P_img = positive_image_ratio(img_dir, lbl_dir)

    # instance extraction + feature + B2 proxy
    items = collect_patches_and_features(
        img_dir, lbl_dir,
        pad_ratio=pad_ratio,
        max_patches=max_patches,
        batch_size=batch_size,
        ring_ws=ring_ws,
        ring_q=ring_q,
        ring_agg=ring_agg,
    )

    Z = np.stack([it.emb for it in items], axis=0).astype(np.float32)
    y = np.asarray([it.cls for it in items], dtype=np.int64)

    # 1) separability
    A_sep = fisher_separability(Z, y)

    # 2) kNN impurity
    A_imp = knn_impurity(Z, y, k=knn_k, chunk_size=knn_chunk_size)

    # 3) split consistency
    Rule_split_CV = split_consistency_cv(items, iou_thr=split_iou_thr)

    # 4) content coverage (B2)
    Rule_cover_B2_CV = content_coverage_b2_cv(items, min_per_class=b2_min_per_class)

    # 5/6) distribution metrics
    dist = class_distribution_metrics(y)

    # 归一化得分
    # =========================================================
    # 归一化为 0-1 得分（全部越大越好）
    # =========================================================
    A_sep_score = A_sep / (1.0 + A_sep)
    # 防止只有1类时为0
    if A_sep_score == 0:
        A_sep_score = 1.0
    A_imp_score = 1.0 - A_imp
    Rule_split_score = 1.0 / (1.0 + Rule_split_CV)
    Rule_cover_score = 1.0 / (1.0 + Rule_cover_B2_CV)

    # H_score = float(dist["H_norm"])
    N_eff_score = float(dist["N_eff_norm"])
    P_img_score = float(P_img)

    scores = np.array([
        A_sep_score,
        A_imp_score,
        Rule_split_score,
        Rule_cover_score,
        # H_score,
        N_eff_score,
        P_img_score
    ], dtype=np.float64)

    # =========================================================
    # 几何平均总得分
    # =========================================================
    eps = 1e-12
    annotation_quality_score = float(
        np.exp(np.mean(np.log(scores + eps)))
    )

    print("\n==== Annotation Consistency Result ====")

    print(f"num_instances_used: {len(items):.6f}")
    print(f"num_classes_present: {dist['num_classes_present']:.6f}")

    print("\n---- Raw Metrics ----")
    print(f"A_sep             : {A_sep:.6f}")
    print(f"A_impurity@k={knn_k:<2}   : {A_imp:.6f}")
    print(f"Rule_split_CV     : {Rule_split_CV:.6f}")
    print(f"Rule_cover_B2_CV  : {Rule_cover_B2_CV:.6f}")
    # print(f"H_norm            : {dist['H_norm']:.6f}")
    print(f"N_eff_norm        : {dist['N_eff_norm']:.6f}")
    print(f"P_img             : {P_img:.6f}")

    print("\n---- Normalized Scores ----")
    print(f"A_sep_score       : {A_sep_score:.6f}")
    print(f"A_impurity_score  : {A_imp_score:.6f}")
    print(f"Rule_split_score  : {Rule_split_score:.6f}")
    print(f"Rule_cover_score  : {Rule_cover_score:.6f}")
    # print(f"H_score           : {H_score:.6f}")
    print(f"N_eff_score       : {N_eff_score:.6f}")
    print(f"P_img_score       : {P_img_score:.6f}")

    print("\n---- Overall Score ----")
    print(f"Annotation Quality Score : {annotation_quality_score:.6f}")
    print("======================================")

    return annotation_quality_score


