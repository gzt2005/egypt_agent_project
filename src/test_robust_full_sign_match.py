from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"
AUG_FEATURE_DIR = SIGN_DIR / "sign_features_augmented"

AUG_METADATA_CSV = AUG_FEATURE_DIR / "augmented_feature_metadata.csv"
AUG_BINARY_HOG_PATH = AUG_FEATURE_DIR / "augmented_binary_hog_features.npy"
AUG_EDGE_HOG_PATH = AUG_FEATURE_DIR / "augmented_edge_hog_features.npy"
AUG_HU_PATH = AUG_FEATURE_DIR / "augmented_hu_moments_features.npy"

OUTPUT_SIZE = 128


# =========================
# 2. 支持中文路径读取图片
# =========================
def read_image_gray(image_path: Path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    return img


# =========================
# 3. 基础裁剪居中
# =========================
def crop_center_resize(binary_img, output_size: int = 128):
    coords = cv2.findNonZero(binary_img)

    if coords is None:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary_img[y:y + h, x:x + w]

    pad = 20
    cropped = cv2.copyMakeBorder(
        cropped,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    h2, w2 = cropped.shape
    scale = (output_size - 20) / max(h2, w2)

    new_w = max(1, int(w2 * scale))
    new_h = max(1, int(h2 * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    start_x = (output_size - new_w) // 2
    start_y = (output_size - new_h) // 2

    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return canvas


# =========================
# 4. 多种输入图像预处理
# =========================
def preprocess_binary_otsu(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    _, binary_inv = cv2.threshold(
        img_blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((2, 2), np.uint8)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

    return crop_center_resize(binary_inv, OUTPUT_SIZE)


def preprocess_adaptive_threshold(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    adaptive = cv2.adaptiveThreshold(
        img_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    kernel = np.ones((2, 2), np.uint8)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    return crop_center_resize(adaptive, OUTPUT_SIZE)


def preprocess_canny_edge(img):
    h, w = img.shape
    max_side = max(h, w)

    if max_side > 800:
        scale = 800 / max_side
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(img)

    filtered = cv2.bilateralFilter(
        enhanced,
        d=7,
        sigmaColor=50,
        sigmaSpace=50
    )

    edges = cv2.Canny(
        filtered,
        threshold1=40,
        threshold2=140
    )

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    edges = cv2.dilate(
        edges,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    return crop_center_resize(edges, OUTPUT_SIZE)


def preprocess_laplacian_edge(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    lap = cv2.Laplacian(img_blur, cv2.CV_64F)
    lap_abs = np.uint8(np.absolute(lap))

    _, binary = cv2.threshold(
        lap_abs,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return crop_center_resize(binary, OUTPUT_SIZE)


def preprocess_skeleton_like(img):
    """
    简化骨架风格：先自适应阈值，再细化腐蚀一次。
    不依赖额外库。
    """
    binary = preprocess_adaptive_threshold(img)

    kernel = np.ones((2, 2), np.uint8)
    thin = cv2.erode(binary, kernel, iterations=1)

    return thin


def generate_input_views(img):
    """
    给同一张输入图片生成多个通用预处理视图。
    """
    views = {
        "otsu_binary": preprocess_binary_otsu(img),
        "adaptive_binary": preprocess_adaptive_threshold(img),
        "canny_edge": preprocess_canny_edge(img),
        "laplacian_edge": preprocess_laplacian_edge(img),
        "skeleton_like": preprocess_skeleton_like(img),
    }

    return views


# =========================
# 5. 特征提取
# =========================
def extract_hog_feature(img):
    img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    img_norm = img.astype("float32") / 255.0

    feature = hog(
        img_norm,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return feature


def extract_hu_moments(img):
    img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    img_binary = (img > 0).astype("uint8") * 255

    moments = cv2.moments(img_binary)
    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu


def hu_similarity(hu1, hu2):
    distance = np.linalg.norm(hu1 - hu2)
    return float(1 / (1 + distance))


def extract_structure_features(img):
    """
    通用结构特征，不针对任何单个符号：
    - 宽高比
    - 像素密度
    - 轮廓数量
    - 孔洞数量
    - 横向投影峰值
    - 纵向投影峰值
    """
    img_binary = (img > 0).astype("uint8") * 255
    h, w = img_binary.shape

    coords = cv2.findNonZero(img_binary)

    if coords is None:
        return np.zeros(6, dtype=np.float32)

    x, y, bw, bh = cv2.boundingRect(coords)

    aspect_ratio = bh / max(bw, 1)
    density = np.count_nonzero(img_binary) / img_binary.size

    contours, hierarchy = cv2.findContours(
        img_binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_count = len(contours)

    hole_count = 0
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for item in hierarchy:
            parent = item[3]
            if parent != -1:
                hole_count += 1

    row_projection = np.sum(img_binary > 0, axis=1)
    col_projection = np.sum(img_binary > 0, axis=0)

    horizontal_peak = row_projection.max() / max(w, 1)
    vertical_peak = col_projection.max() / max(h, 1)

    features = np.array([
        aspect_ratio,
        density,
        contour_count / 20.0,
        hole_count / 10.0,
        horizontal_peak,
        vertical_peak
    ], dtype=np.float32)

    return features


def structure_similarity(f1, f2):
    distance = np.linalg.norm(f1 - f2)
    return float(1 / (1 + distance))


# =========================
# 6. 加载增强特征缓存
# =========================
def load_augmented_feature_cache():
    if not AUG_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到增强元数据：{AUG_METADATA_CSV}")

    if not AUG_BINARY_HOG_PATH.exists():
        raise FileNotFoundError(f"未找到增强 binary HOG 特征：{AUG_BINARY_HOG_PATH}")

    if not AUG_EDGE_HOG_PATH.exists():
        raise FileNotFoundError(f"未找到增强 edge HOG 特征：{AUG_EDGE_HOG_PATH}")

    if not AUG_HU_PATH.exists():
        raise FileNotFoundError(f"未找到增强 Hu 特征：{AUG_HU_PATH}")

    metadata = pd.read_csv(AUG_METADATA_CSV, dtype=str).fillna("")
    binary_hog_matrix = np.load(AUG_BINARY_HOG_PATH)
    edge_hog_matrix = np.load(AUG_EDGE_HOG_PATH)
    hu_matrix = np.load(AUG_HU_PATH)

    return metadata, binary_hog_matrix, edge_hog_matrix, hu_matrix


# =========================
# 7. Rank Fusion 工具
# =========================
def reciprocal_rank_fusion(score_array, k=60):
    """
    将一组相似度分数转成 RRF 排名分。
    分数越高，排名越靠前。
    """
    order = np.argsort(score_array)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(score_array) + 1)

    rrf_scores = 1.0 / (k + ranks)

    return rrf_scores


def get_confidence_level(score):
    score = float(score)

    if score >= 0.035:
        return "高"
    elif score >= 0.025:
        return "中"
    else:
        return "低"


# =========================
# 8. 鲁棒匹配主函数
# =========================
def match_robust_full_sign_image(image_path: Path, top_k: int = 10):
    metadata, binary_hog_matrix, edge_hog_matrix, hu_matrix = load_augmented_feature_cache()

    img = read_image_gray(image_path)
    views = generate_input_views(img)

    all_rrf_scores = np.zeros(len(metadata), dtype=float)

    # 保存一些可解释的最大原始分数
    max_binary_hog = np.zeros(len(metadata), dtype=float)
    max_edge_hog = np.zeros(len(metadata), dtype=float)
    max_hu = np.zeros(len(metadata), dtype=float)
    max_structure = np.zeros(len(metadata), dtype=float)

    # 为库中的增强样本计算结构特征。
    # 为避免额外缓存，这里用 Hu 不足以表达结构，所以临时从已有特征中不取图像。
    # 当前版本结构特征只对输入视图之间稳定性做辅助，不直接使用库结构缓存。
    # 因此主要融合 HOG 与 Hu。
    for view_name, view_img in views.items():
        input_hog = extract_hog_feature(view_img).reshape(1, -1)
        input_hu = extract_hu_moments(view_img)

        # 根据视图类型选择匹配矩阵
        if "edge" in view_name or "laplacian" in view_name:
            hog_scores = cosine_similarity(input_hog, edge_hog_matrix)[0]
        else:
            hog_scores = cosine_similarity(input_hog, binary_hog_matrix)[0]

        hu_scores = np.array([
            hu_similarity(input_hu, hu_matrix[i])
            for i in range(len(metadata))
        ])

        # RRF 排名融合
        hog_rrf = reciprocal_rank_fusion(hog_scores)
        hu_rrf = reciprocal_rank_fusion(hu_scores)

        view_rrf = 0.75 * hog_rrf + 0.25 * hu_rrf

        all_rrf_scores += view_rrf

        max_binary_hog = np.maximum(max_binary_hog, hog_scores)
        max_edge_hog = np.maximum(max_edge_hog, hog_scores)
        max_hu = np.maximum(max_hu, hu_scores)

    # 归一化一下，方便解释
    all_rrf_scores = all_rrf_scores / len(views)

    scored = metadata.copy()
    scored["robust_score"] = all_rrf_scores
    scored["max_hog_score"] = max_binary_hog
    scored["max_edge_or_binary_hog_score"] = max_edge_hog
    scored["max_hu_score"] = max_hu

    # 按 sign_id 聚合，每个符号取最高 robust_score 的增强变体
    idx_best = scored.groupby("sign_id")["robust_score"].idxmax()
    best_per_sign = scored.loc[idx_best].copy()

    best_per_sign = best_per_sign.sort_values(
        by="robust_score",
        ascending=False
    ).head(top_k)

    best_per_sign["confidence_level"] = best_per_sign["robust_score"].apply(
        get_confidence_level
    )

    return best_per_sign, views


# =========================
# 9. 输出结果
# =========================
def print_match_results(results):
    print("\n鲁棒匹配结果 Top-K：")
    print("-" * 120)

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        gardiner_code = row.get("gardiner_code", "")
        zh_name = row.get("zh_name", "")
        en_name = row.get("en_name", "")
        unicode_char = row.get("unicode_char", "")
        unicode_codepoint = row.get("unicode_codepoint", "")
        auto_label = row.get("auto_label", "")
        related_terms = row.get("related_terms", "")
        has_manual_annotation = row.get("has_manual_annotation", "")
        variant_type = row.get("variant_type", "")
        variant_param = row.get("variant_param", "")

        display_code = gardiner_code if gardiner_code else auto_label
        display_zh = zh_name if zh_name else "暂无中文注释"
        display_en = en_name if en_name else "暂无英文注释"

        print(f"Rank {rank}")
        print("显示编号:", display_code)
        print("Unicode:", unicode_char)
        print("Codepoint:", unicode_codepoint)
        print("Auto label:", auto_label)
        print("Gardiner:", gardiner_code if gardiner_code else "暂无人工注释")
        print("中文名:", display_zh)
        print("英文名:", display_en)
        print("相关检索词:", related_terms if related_terms else "暂无")
        print("是否人工注释:", has_manual_annotation)
        print("最佳增强类型:", variant_type)
        print("增强参数:", variant_param)
        print("Robust Rank-Fusion 分数:", round(float(row.get("robust_score", 0)), 6))
        print("最大 HOG 分数:", round(float(row.get("max_hog_score", 0)), 4))
        print("最大 Hu 分数:", round(float(row.get("max_hu_score", 0)), 4))
        print("置信度:", row.get("confidence_level", ""))
        print("-" * 120)


# =========================
# 10. 主程序
# =========================
def main():
    print("=" * 120)
    print("V2.7 鲁棒版全量 Unicode 古埃及符号图像匹配测试")
    print("=" * 120)
    print("候选库：1072 个符号 × 12 个增强版本 = 12864 个增强样本")
    print("策略：多预处理视图 + HOG/Hu + Reciprocal Rank Fusion + sign_id 聚合")
    print("不包含任何单独符号加分，不包含人工注释加权。")
    print("\n示例：")
    print(r"C:\Users\GE ZITONG\Desktop\安卡1.png")
    print(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project\data_sign_demo\sign_png_full\U_13000.png")

    while True:
        user_input = input("\n请输入图片路径 / q：").strip().strip('"')

        if user_input.lower() == "q":
            print("已退出。")
            break

        image_path = Path(user_input)

        try:
            results, _ = match_robust_full_sign_image(
                image_path=image_path,
                top_k=10
            )
            print_match_results(results)

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()