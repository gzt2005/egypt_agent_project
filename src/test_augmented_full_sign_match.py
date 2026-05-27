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
    """
    支持 Windows 中文路径读取图片。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    return img


# =========================
# 3. 输入图片预处理
# =========================
def crop_center_resize(binary_img, output_size: int = 128):
    """
    对二值图或边缘图进行裁剪、缩放和居中。
    """
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


def preprocess_binary_style(img, output_size: int = 128):
    """
    适合白底黑字、截图、标准符号图的预处理。
    """
    img = cv2.GaussianBlur(img, (3, 3), 0)

    _, binary_inv = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((2, 2), np.uint8)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

    return crop_center_resize(binary_inv, output_size)


def preprocess_photo_edge_style(img, output_size: int = 128):
    """
    适合浮雕照片、真实图片、光照不均图片的边缘预处理。
    """
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
        threshold1=50,
        threshold2=150
    )

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    edges = cv2.dilate(
        edges,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    return crop_center_resize(edges, output_size)


# =========================
# 4. 特征提取
# =========================
def extract_hog_feature(img):
    """
    提取 HOG 特征。
    """
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
    """
    提取 Hu Moments 特征。
    """
    img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    img_binary = (img > 0).astype("uint8") * 255

    moments = cv2.moments(img_binary)
    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu


def hu_similarity(hu1, hu2):
    """
    Hu Moments 距离转相似度。
    """
    distance = np.linalg.norm(hu1 - hu2)
    return float(1 / (1 + distance))


# =========================
# 5. 加载增强特征缓存
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
# 6. 置信度
# =========================
def get_confidence_level(score):
    score = float(score)

    if score >= 0.70:
        return "高"
    elif score >= 0.50:
        return "中"
    else:
        return "低"


# =========================
# 7. 增强库匹配 + 按 sign_id 聚合
# =========================
def match_augmented_full_sign_image(image_path: Path, top_k: int = 10):
    """
    使用增强特征库进行全量符号匹配。
    先匹配 12864 个增强样本，再按 sign_id 聚合，每个符号取最高分。
    """
    metadata, binary_hog_matrix, edge_hog_matrix, hu_matrix = load_augmented_feature_cache()

    img = read_image_gray(image_path)

    input_binary = preprocess_binary_style(img, OUTPUT_SIZE)
    input_edge = preprocess_photo_edge_style(img, OUTPUT_SIZE)

    input_binary_hog = extract_hog_feature(input_binary).reshape(1, -1)
    input_edge_hog = extract_hog_feature(input_edge).reshape(1, -1)
    input_hu = extract_hu_moments(input_binary)

    binary_hog_scores = cosine_similarity(
        input_binary_hog,
        binary_hog_matrix
    )[0]

    edge_hog_scores = cosine_similarity(
        input_edge_hog,
        edge_hog_matrix
    )[0]

    hu_scores = np.array([
        hu_similarity(input_hu, hu_matrix[i])
        for i in range(len(metadata))
    ])

    # 通用融合分数：不加入任何单一符号加分
    final_scores = (
        0.40 * binary_hog_scores +
        0.40 * edge_hog_scores +
        0.20 * hu_scores
    )

    scored = metadata.copy()
    scored["binary_hog_score"] = binary_hog_scores
    scored["edge_hog_score"] = edge_hog_scores
    scored["hu_score"] = hu_scores
    scored["final_score"] = final_scores

    # 每个 sign_id 只保留分数最高的增强变体
    idx_best = scored.groupby("sign_id")["final_score"].idxmax()
    best_per_sign = scored.loc[idx_best].copy()

    best_per_sign = best_per_sign.sort_values(
        by="final_score",
        ascending=False
    ).head(top_k)

    best_per_sign["confidence_level"] = best_per_sign["final_score"].apply(
        get_confidence_level
    )

    return best_per_sign, input_binary, input_edge


# =========================
# 8. 输出结果
# =========================
def print_match_results(results):
    print("\n匹配结果 Top-K：")
    print("-" * 110)

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
        print("Binary HOG:", round(float(row.get("binary_hog_score", 0)), 4))
        print("Edge HOG:", round(float(row.get("edge_hog_score", 0)), 4))
        print("Hu:", round(float(row.get("hu_score", 0)), 4))
        print("最终融合分数:", round(float(row.get("final_score", 0)), 4))
        print("置信度:", row.get("confidence_level", ""))
        print("-" * 110)


# =========================
# 9. 主程序
# =========================
def main():
    print("=" * 110)
    print("增强版全量 Unicode 古埃及符号图像匹配测试")
    print("=" * 110)
    print("候选库：1072 个符号 × 12 个增强版本 = 12864 个增强样本")
    print("匹配策略：全量增强样本匹配后，按 sign_id 聚合，每个符号取最高分。")
    print("输入图片路径进行测试，输入 q 退出。")
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
            results, _, _ = match_augmented_full_sign_image(
                image_path=image_path,
                top_k=10
            )
            print_match_results(results)

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()