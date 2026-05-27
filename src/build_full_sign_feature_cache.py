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

FEATURE_DIR = SIGN_DIR / "sign_features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

METADATA_OUT = FEATURE_DIR / "full_sign_feature_metadata.csv"
BINARY_HOG_OUT = FEATURE_DIR / "binary_hog_features.npy"
EDGE_HOG_OUT = FEATURE_DIR / "edge_hog_features.npy"
HU_OUT = FEATURE_DIR / "hu_moments_features.npy"

OUTPUT_SIZE = 128


# =========================
# 2. 图片读取
# =========================
def read_image_gray(image_path: Path):
    """
    支持中文路径读取图片。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return None

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    return img


# =========================
# 3. 标准符号边缘图
# =========================
def standard_to_edge(img):
    """
    将标准预处理图转换为边缘图。
    """
    img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    return edges


# =========================
# 4. HOG 特征
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


# =========================
# 5. Hu Moments 特征
# =========================
def extract_hu_moments(img):
    """
    提取 Hu Moments 轮廓矩特征。
    """
    img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
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

    print("=" * 80)
    print("开始构建全量古埃及符号特征缓存")
    print("=" * 80)
    print("输入元数据：", PROCESSED_LIBRARY_CSV)
    print("输入记录数：", len(df))
    print("输出目录：", FEATURE_DIR)

    valid_rows = []
    binary_hog_features = []
    edge_hog_features = []
    hu_features = []

    for idx, row in df.iterrows():
        processed_path = row.get("processed_png_path", "")

        if not processed_path:
            continue

        img = read_image_gray(Path(processed_path))

        if img is None:
            continue

        img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
        edge_img = standard_to_edge(img)

        binary_hog = extract_hog_feature(img)
        edge_hog = extract_hog_feature(edge_img)
        hu = extract_hu_moments(img)

        valid_rows.append(row)
        binary_hog_features.append(binary_hog)
        edge_hog_features.append(edge_hog)
        hu_features.append(hu)

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1} / {len(df)} 条")

    if not valid_rows:
        raise ValueError("没有成功生成任何特征，请检查 processed_png_path 是否有效。")

    metadata = pd.DataFrame(valid_rows).reset_index(drop=True)

    binary_hog_matrix = np.vstack(binary_hog_features)
    edge_hog_matrix = np.vstack(edge_hog_features)
    hu_matrix = np.vstack(hu_features)

    metadata.to_csv(METADATA_OUT, index=False, encoding="utf-8-sig")
    np.save(BINARY_HOG_OUT, binary_hog_matrix)
    np.save(EDGE_HOG_OUT, edge_hog_matrix)
    np.save(HU_OUT, hu_matrix)

    print("\n" + "=" * 80)
    print("全量符号特征缓存构建完成！")
    print("=" * 80)
    print("有效符号数量：", len(metadata))
    print("binary_hog_matrix shape:", binary_hog_matrix.shape)
    print("edge_hog_matrix shape:", edge_hog_matrix.shape)
    print("hu_matrix shape:", hu_matrix.shape)

    print("\n输出文件：")
    print("1.", METADATA_OUT)
    print("2.", BINARY_HOG_OUT)
    print("3.", EDGE_HOG_OUT)
    print("4.", HU_OUT)

    print("\n人工注释数量：")
    if "has_manual_annotation" in metadata.columns:
        print(metadata["has_manual_annotation"].value_counts())
    else:
        print("未找到 has_manual_annotation 字段。")


if __name__ == "__main__":
    main()