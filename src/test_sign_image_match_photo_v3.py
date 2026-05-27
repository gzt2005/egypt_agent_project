from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"
SIGN_PROCESSED_DIR = SIGN_DIR / "sign_processed"
SIGN_METADATA_CSV = SIGN_DIR / "hieroglyph_signs_processed.csv"

OUTPUT_SIZE = 128


# =========================
# 2. 中文路径读取图片
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
# 3. 普通二值化预处理
# =========================
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


# =========================
# 4. 浮雕照片边缘增强预处理
# =========================
def preprocess_photo_edge_style(img, output_size: int = 128):
    """
    适合石壁浮雕、真实照片、光照不均图片的预处理。
    重点提取符号边缘，减少背景纹理干扰。
    """
    # 1. 缩放过大的图片，提升速度
    h, w = img.shape
    max_side = max(h, w)

    if max_side > 800:
        scale = 800 / max_side
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    # 2. 局部对比度增强 CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(img)

    # 3. 双边滤波：保留边缘，减少石壁纹理噪声
    filtered = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    # 4. Canny 边缘检测
    edges = cv2.Canny(filtered, threshold1=50, threshold2=150)

    # 5. 形态学闭运算，连接断裂边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 6. 膨胀一小步，让边缘更明显
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    return crop_center_resize(edges, output_size)


# =========================
# 5. 裁剪、缩放、居中
# =========================
def crop_center_resize(binary_img, output_size: int = 128):
    """
    对二值图/边缘图进行主体区域裁剪、加边距、等比例缩放、居中。
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


# =========================
# 6. 标准符号边缘化
# =========================
def standard_to_edge(img):
    """
    把标准符号处理图转换为边缘图，方便和真实照片边缘匹配。
    """
    img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    return edges


# =========================
# 7. 特征提取
# =========================
def extract_hog_feature(img):
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
    img_binary = (img > 0).astype("uint8") * 255

    moments = cv2.moments(img_binary)
    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu


def hu_similarity(hu1, hu2):
    distance = np.linalg.norm(hu1 - hu2)
    return float(1 / (1 + distance))


def calculate_ssim_similarity(img1, img2):
    img1_norm = img1.astype("float32") / 255.0
    img2_norm = img2.astype("float32") / 255.0

    score = ssim(img1_norm, img2_norm, data_range=1.0)

    return max(0.0, min(1.0, float(score)))


# =========================
# 8. S34 生命符号结构加分
# =========================
def ankh_structure_score(img):
    """
    针对 S34 生命符号的简单结构检测：
    1. 上部有较大环形/椭圆边缘
    2. 中部有横向结构
    3. 下部有竖向结构
    """
    img_binary = (img > 0).astype("uint8") * 255

    h, w = img_binary.shape

    upper = img_binary[: h // 2, :]
    middle = img_binary[h // 3: 2 * h // 3, :]
    lower = img_binary[h // 2:, :]

    # 上部轮廓密度
    upper_density = np.count_nonzero(upper) / upper.size

    # 中部横向投影：如果某些行像素很多，说明有横杠
    row_projection = np.sum(middle > 0, axis=1)
    horizontal_strength = row_projection.max() / w if len(row_projection) > 0 else 0

    # 下部纵向投影：如果某些列像素很多，说明有竖线
    col_projection = np.sum(lower > 0, axis=0)
    vertical_strength = col_projection.max() / (h // 2) if len(col_projection) > 0 else 0

    # 宽高主体比例
    coords = cv2.findNonZero(img_binary)
    if coords is None:
        aspect_score = 0
    else:
        x, y, bw, bh = cv2.boundingRect(coords)
        aspect_ratio = bh / max(bw, 1)
        aspect_score = 1.0 if 1.1 <= aspect_ratio <= 2.6 else 0.4

    raw_score = (
        0.30 * min(upper_density / 0.12, 1.0) +
        0.30 * min(horizontal_strength / 0.35, 1.0) +
        0.30 * min(vertical_strength / 0.35, 1.0) +
        0.10 * aspect_score
    )

    return float(max(0.0, min(1.0, raw_score)))


# =========================
# 9. 加载标准库特征
# =========================
def load_sign_library_features():
    if not SIGN_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到处理后符号元数据表：{SIGN_METADATA_CSV}")

    df = pd.read_csv(SIGN_METADATA_CSV, dtype=str).fillna("")

    valid_rows = []
    binary_images = []
    edge_images = []

    binary_hog_features = []
    edge_hog_features = []
    hu_features = []

    for _, row in df.iterrows():
        processed_path = Path(row["processed_png_path"])

        if not processed_path.exists():
            continue

        img = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
        edge_img = standard_to_edge(img)

        valid_rows.append(row)
        binary_images.append(img)
        edge_images.append(edge_img)

        binary_hog_features.append(extract_hog_feature(img))
        edge_hog_features.append(extract_hog_feature(edge_img))
        hu_features.append(extract_hu_moments(img))

    if not valid_rows:
        raise ValueError("没有成功加载任何标准符号图像。")

    metadata = pd.DataFrame(valid_rows).reset_index(drop=True)

    return {
        "metadata": metadata,
        "binary_images": binary_images,
        "edge_images": edge_images,
        "binary_hog_matrix": np.vstack(binary_hog_features),
        "edge_hog_matrix": np.vstack(edge_hog_features),
        "hu_matrix": np.vstack(hu_features),
    }


# =========================
# 10. 置信度
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
# 11. V3 图片匹配
# =========================
def match_input_image_photo_v3(image_path: Path, top_k: int = 5):
    """
    真实照片增强版匹配：
    - 同时生成二值风格图和边缘风格图
    - 用标准库的二值图和边缘图分别比对
    - 对 S34 生命符号加入结构加分
    """
    library = load_sign_library_features()
    metadata = library["metadata"]

    img = read_image_gray(image_path)

    input_binary = preprocess_binary_style(img, OUTPUT_SIZE)
    input_edge = preprocess_photo_edge_style(img, OUTPUT_SIZE)

    input_binary_hog = extract_hog_feature(input_binary).reshape(1, -1)
    input_edge_hog = extract_hog_feature(input_edge).reshape(1, -1)
    input_hu = extract_hu_moments(input_binary)

    binary_hog_scores = cosine_similarity(
        input_binary_hog,
        library["binary_hog_matrix"]
    )[0]

    edge_hog_scores = cosine_similarity(
        input_edge_hog,
        library["edge_hog_matrix"]
    )[0]

    ssim_scores = []
    hu_scores = []
    ankh_scores = []

    for idx in range(len(metadata)):
        standard_binary = library["binary_images"][idx]
        standard_edge = library["edge_images"][idx]
        standard_hu = library["hu_matrix"][idx]

        ssim_binary = calculate_ssim_similarity(input_binary, standard_binary)
        ssim_edge = calculate_ssim_similarity(input_edge, standard_edge)

        ssim_score = 0.45 * ssim_binary + 0.55 * ssim_edge
        hu_score = hu_similarity(input_hu, standard_hu)

        gardiner_code = metadata.iloc[idx].get("gardiner_code", "")

        if gardiner_code == "S34":
            ankh_score = ankh_structure_score(input_edge)
        else:
            ankh_score = 0.0

        ssim_scores.append(ssim_score)
        hu_scores.append(hu_score)
        ankh_scores.append(ankh_score)

    ssim_scores = np.array(ssim_scores)
    hu_scores = np.array(hu_scores)
    ankh_scores = np.array(ankh_scores)

    # 综合分数
    final_scores = (
        0.25 * binary_hog_scores +
        0.25 * edge_hog_scores +
        0.25 * ssim_scores +
        0.15 * hu_scores +
        0.10 * ankh_scores
    )

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()

    results["binary_hog_score"] = binary_hog_scores[top_indices]
    results["edge_hog_score"] = edge_hog_scores[top_indices]
    results["ssim_score"] = ssim_scores[top_indices]
    results["hu_score"] = hu_scores[top_indices]
    results["ankh_structure_score"] = ankh_scores[top_indices]
    results["final_score"] = final_scores[top_indices]
    results["confidence_level"] = results["final_score"].apply(get_confidence_level)

    return results, input_binary, input_edge


# =========================
# 12. 交互主程序
# =========================
def main():
    print("=" * 80)
    print("古埃及符号图像匹配测试 V3：真实浮雕照片增强版")
    print("=" * 80)
    print("输入图片路径进行测试。")
    print("输入 q 退出。")
    print("\n示例：")
    print(r"C:\Users\GE ZITONG\Desktop\安卡1.png")

    while True:
        user_input = input("\n请输入图片路径 / q：").strip().strip('"')

        if user_input.lower() == "q":
            print("已退出。")
            break

        image_path = Path(user_input)

        try:
            results, _, _ = match_input_image_photo_v3(image_path, top_k=5)

            print("\n匹配结果 Top-5：")
            print("-" * 80)

            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                print(f"Rank {rank}")
                print("Gardiner:", row.get("gardiner_code", ""))
                print("Unicode:", row.get("unicode_char", ""))
                print("中文名:", row.get("zh_name", ""))
                print("英文名:", row.get("en_name", ""))
                print("相关检索词:", row.get("related_terms", ""))
                print("Binary HOG:", round(float(row.get("binary_hog_score", 0)), 4))
                print("Edge HOG:", round(float(row.get("edge_hog_score", 0)), 4))
                print("SSIM:", round(float(row.get("ssim_score", 0)), 4))
                print("Hu:", round(float(row.get("hu_score", 0)), 4))
                print("Ankh结构分:", round(float(row.get("ankh_structure_score", 0)), 4))
                print("融合分数:", round(float(row.get("final_score", 0)), 4))
                print("置信度:", row.get("confidence_level", ""))
                print("-" * 80)

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()