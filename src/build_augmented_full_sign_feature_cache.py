from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

PROCESSED_LIBRARY_CSV = SIGN_DIR / "sign_library_merged_processed.csv"

AUG_FEATURE_DIR = SIGN_DIR / "sign_features_augmented"
AUG_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

AUG_METADATA_OUT = AUG_FEATURE_DIR / "augmented_feature_metadata.csv"
AUG_BINARY_HOG_OUT = AUG_FEATURE_DIR / "augmented_binary_hog_features.npy"
AUG_EDGE_HOG_OUT = AUG_FEATURE_DIR / "augmented_edge_hog_features.npy"
AUG_HU_OUT = AUG_FEATURE_DIR / "augmented_hu_moments_features.npy"

OUTPUT_SIZE = 128


# =========================
# 2. 图像读取
# =========================
def read_image_gray(image_path: Path):
    """
    支持 Windows 中文路径读取图片。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return None

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    return img


# =========================
# 3. 基础图像工具
# =========================
def ensure_128(img):
    """
    保证图像尺寸为 128×128。
    """
    if img is None:
        return None

    if img.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
        img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

    return img


def standard_to_edge(img):
    """
    将标准预处理图转换为边缘图。
    """
    img = ensure_128(img)

    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    return edges


def rotate_image_keep_canvas(img, angle):
    """
    在 128×128 画布内旋转图像。
    用于模拟真实图片中的轻微倾斜。
    """
    img = ensure_128(img)

    center = (OUTPUT_SIZE // 2, OUTPUT_SIZE // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        matrix,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated


def thicken_image(img):
    """
    笔画加粗，模拟浮雕边缘、粗线条或截图放大。
    """
    img = ensure_128(img)

    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.dilate(img, kernel, iterations=1)

    return thick


def thin_image(img):
    """
    笔画变细，模拟线条残缺或较浅浮雕。
    """
    img = ensure_128(img)

    kernel = np.ones((2, 2), np.uint8)
    thin = cv2.erode(img, kernel, iterations=1)

    return thin


def blur_image(img):
    """
    轻微模糊，模拟拍照或截图造成的模糊。
    """
    img = ensure_128(img)

    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    return blurred


def shift_image(img, dx, dy):
    """
    轻微平移，模拟用户裁剪时符号不完全居中。
    """
    img = ensure_128(img)

    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    shifted = cv2.warpAffine(
        img,
        matrix,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return shifted


# =========================
# 4. 数据增强
# =========================
def generate_variants(img):
    """
    为单个标准符号生成多个视觉变体。
    每个变体返回：
    variant_type, variant_param, image
    """
    img = ensure_128(img)

    variants = []

    # 原始图
    variants.append(("original", "0", img))

    # 轻微旋转
    for angle in [-10, -5, 5, 10]:
        variants.append((f"rotate_{angle}", str(angle), rotate_image_keep_canvas(img, angle)))

    # 笔画粗细变化
    variants.append(("thicken", "1", thicken_image(img)))
    variants.append(("thin", "1", thin_image(img)))

    # 轻微模糊
    variants.append(("blur", "3x3", blur_image(img)))

    # 轻微平移
    variants.append(("shift_left", "-4,0", shift_image(img, -4, 0)))
    variants.append(("shift_right", "4,0", shift_image(img, 4, 0)))
    variants.append(("shift_up", "0,-4", shift_image(img, 0, -4)))
    variants.append(("shift_down", "0,4", shift_image(img, 0, 4)))

    return variants


# =========================
# 5. 特征提取
# =========================
def extract_hog_feature(img):
    """
    提取 HOG 特征。
    """
    img = ensure_128(img)

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
    img = ensure_128(img)

    img_binary = (img > 0).astype("uint8") * 255

    moments = cv2.moments(img_binary)
    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu


# =========================
# 6. 主流程
# =========================
def main():
    if not PROCESSED_LIBRARY_CSV.exists():
        raise FileNotFoundError(f"未找到全量预处理符号库：{PROCESSED_LIBRARY_CSV}")

    df = pd.read_csv(PROCESSED_LIBRARY_CSV, dtype=str).fillna("")

    print("=" * 90)
    print("开始构建全量古埃及符号增强特征缓存")
    print("=" * 90)
    print("输入元数据：", PROCESSED_LIBRARY_CSV)
    print("输入符号数量：", len(df))
    print("输出目录：", AUG_FEATURE_DIR)
    print("增强策略：original + rotation + thickness + blur + shift")
    print("=" * 90)

    augmented_rows = []
    binary_hog_features = []
    edge_hog_features = []
    hu_features = []

    success_sign_count = 0
    failed_sign_count = 0
    total_variant_count = 0

    for idx, row in df.iterrows():
        processed_path = row.get("processed_png_path", "")

        if not processed_path:
            failed_sign_count += 1
            continue

        img = read_image_gray(Path(processed_path))

        if img is None:
            failed_sign_count += 1
            continue

        img = ensure_128(img)

        variants = generate_variants(img)

        for variant_type, variant_param, variant_img in variants:
            edge_img = standard_to_edge(variant_img)

            binary_hog = extract_hog_feature(variant_img)
            edge_hog = extract_hog_feature(edge_img)
            hu = extract_hu_moments(variant_img)

            new_row = row.to_dict()
            new_row["variant_type"] = variant_type
            new_row["variant_param"] = variant_param
            new_row["base_processed_png_path"] = processed_path

            augmented_rows.append(new_row)
            binary_hog_features.append(binary_hog)
            edge_hog_features.append(edge_hog)
            hu_features.append(hu)

            total_variant_count += 1

        success_sign_count += 1

        if (idx + 1) % 100 == 0:
            print(
                f"已处理符号 {idx + 1} / {len(df)}，"
                f"当前增强样本数：{total_variant_count}"
            )

    if not augmented_rows:
        raise ValueError("没有成功生成任何增强特征，请检查 processed_png_path 是否有效。")

    augmented_metadata = pd.DataFrame(augmented_rows).reset_index(drop=True)

    binary_hog_matrix = np.vstack(binary_hog_features)
    edge_hog_matrix = np.vstack(edge_hog_features)
    hu_matrix = np.vstack(hu_features)

    augmented_metadata.to_csv(AUG_METADATA_OUT, index=False, encoding="utf-8-sig")
    np.save(AUG_BINARY_HOG_OUT, binary_hog_matrix)
    np.save(AUG_EDGE_HOG_OUT, edge_hog_matrix)
    np.save(AUG_HU_OUT, hu_matrix)

    print("\n" + "=" * 90)
    print("全量增强符号特征缓存构建完成！")
    print("=" * 90)
    print("成功处理符号数量：", success_sign_count)
    print("失败符号数量：", failed_sign_count)
    print("增强样本总数：", len(augmented_metadata))
    print("平均每个符号增强样本数：", round(len(augmented_metadata) / max(success_sign_count, 1), 2))

    print("\n特征矩阵 shape：")
    print("binary_hog_matrix:", binary_hog_matrix.shape)
    print("edge_hog_matrix:", edge_hog_matrix.shape)
    print("hu_matrix:", hu_matrix.shape)

    print("\n输出文件：")
    print("1.", AUG_METADATA_OUT)
    print("2.", AUG_BINARY_HOG_OUT)
    print("3.", AUG_EDGE_HOG_OUT)
    print("4.", AUG_HU_OUT)

    print("\n人工注释数量统计：")
    if "has_manual_annotation" in augmented_metadata.columns:
        print(augmented_metadata["has_manual_annotation"].value_counts())
    else:
        print("未找到 has_manual_annotation 字段。")

    print("\n增强类型统计：")
    print(augmented_metadata["variant_type"].value_counts())


if __name__ == "__main__":
    main()