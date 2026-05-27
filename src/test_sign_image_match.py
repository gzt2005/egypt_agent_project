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
SIGN_PROCESSED_DIR = SIGN_DIR / "sign_processed"
SIGN_METADATA_CSV = SIGN_DIR / "hieroglyph_signs_processed.csv"

OUTPUT_SIZE = 128


# =========================
# 2. 图像预处理函数
# =========================
def preprocess_input_image(image_path: Path, output_size: int = 128):
    """
    对用户输入图片进行与标准符号库一致的预处理：
    1. 灰度读取
    2. 二值化
    3. 提取符号区域
    4. 裁剪、加边距、缩放、居中
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    # Otsu 二值化 + 反色：符号为白，背景为黑
    _, binary_inv = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary_inv)

    if coords is None:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary_inv[y:y + h, x:x + w]

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
# 3. HOG 特征提取
# =========================
def extract_hog_feature(img):
    """
    提取 HOG 特征。
    输入 img 应为 128×128 灰度图。
    """
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
# 4. 加载标准符号库特征
# =========================
def load_sign_library_features():
    if not SIGN_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到处理后符号元数据表：{SIGN_METADATA_CSV}")

    df = pd.read_csv(SIGN_METADATA_CSV, dtype=str).fillna("")

    features = []
    valid_rows = []

    for _, row in df.iterrows():
        processed_path = Path(row["processed_png_path"])

        if not processed_path.exists():
            continue

        img = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        feature = extract_hog_feature(img)

        features.append(feature)
        valid_rows.append(row)

    feature_matrix = np.vstack(features)
    metadata = pd.DataFrame(valid_rows).reset_index(drop=True)

    return metadata, feature_matrix


# =========================
# 5. 图像匹配
# =========================
def match_input_image(image_path: Path, top_k: int = 5):
    """
    匹配用户输入图片，返回 Top-K 相似符号。
    """
    metadata, feature_matrix = load_sign_library_features()

    input_img = preprocess_input_image(image_path, OUTPUT_SIZE)
    input_feature = extract_hog_feature(input_img).reshape(1, -1)

    similarities = cosine_similarity(input_feature, feature_matrix)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results, input_img


# =========================
# 6. 主程序：交互测试
# =========================
def main():
    print("=" * 60)
    print("古埃及符号图像匹配测试")
    print("=" * 60)
    print("说明：请输入一张古埃及符号图片路径。")
    print("建议先用 data_sign_demo/sign_png 里的标准符号测试。")
    print("例如：")
    print(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project\data_sign_demo\sign_png\N5.png")
    print("输入 q 退出。")

    while True:
        user_input = input("\n请输入图片路径：").strip().strip('"')

        if user_input.lower() == "q":
            print("已退出。")
            break

        image_path = Path(user_input)

        if not image_path.exists():
            print("图片不存在，请重新输入。")
            continue

        try:
            results, _ = match_input_image(image_path, top_k=5)

            print("\n匹配结果 Top-5：")
            print("-" * 80)

            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                print(f"Rank {rank}")
                print("Gardiner:", row.get("gardiner_code", ""))
                print("Unicode:", row.get("unicode_char", ""))
                print("中文名:", row.get("zh_name", ""))
                print("英文名:", row.get("en_name", ""))
                print("相关检索词:", row.get("related_terms", ""))
                print("相似度:", round(float(row.get("similarity", 0)), 4))
                print("-" * 80)

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()