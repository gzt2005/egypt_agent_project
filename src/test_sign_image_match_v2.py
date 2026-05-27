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
# 2. 图像预处理
# =========================
def preprocess_input_image(image_path: Path, output_size: int = 128):
    """
    对用户输入图片进行标准化处理。
    使用 np.fromfile + cv2.imdecode 读取图片，避免 Windows 中文路径乱码问题。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    # 关键修改：支持中文路径读取
    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    # 轻微去噪，降低截图或压缩噪声影响
    img = cv2.GaussianBlur(img, (3, 3), 0)

    _, binary_inv = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 形态学闭运算，让断裂笔画稍微连接
    kernel = np.ones((2, 2), np.uint8)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

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
# 3. HOG 特征
# =========================
def extract_hog_feature(img):
    """
    提取 HOG 特征，用于描述边缘方向和局部形状。
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
# 4. Hu Moments 特征
# =========================
def extract_hu_moments(img):
    """
    提取 Hu Moments 轮廓矩特征。
    Hu Moments 对缩放、平移和一定程度旋转更稳。
    """
    img_binary = (img > 0).astype("uint8") * 255

    moments = cv2.moments(img_binary)
    hu = cv2.HuMoments(moments).flatten()

    # log 变换，压缩数量级
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu


def hu_similarity(hu1, hu2):
    """
    将 Hu Moments 距离转换为 0-1 相似度。
    距离越小，相似度越高。
    """
    distance = np.linalg.norm(hu1 - hu2)

    # 距离转相似度，避免出现负数
    similarity = 1 / (1 + distance)

    return float(similarity)


# =========================
# 5. SSIM 图像结构相似度
# =========================
def calculate_ssim_similarity(img1, img2):
    """
    计算两个标准化二值图之间的结构相似度。
    """
    img1_norm = img1.astype("float32") / 255.0
    img2_norm = img2.astype("float32") / 255.0

    score = ssim(img1_norm, img2_norm, data_range=1.0)

    # SSIM 可能略低于 0，统一压到 0-1
    score = max(0.0, min(1.0, float(score)))

    return score


# =========================
# 6. 加载标准符号库
# =========================
def load_sign_library_features():
    """
    加载标准符号库，预先计算 HOG、Hu Moments 和标准图像。
    """
    if not SIGN_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到处理后符号元数据表：{SIGN_METADATA_CSV}")

    df = pd.read_csv(SIGN_METADATA_CSV, dtype=str).fillna("")

    valid_rows = []
    hog_features = []
    hu_features = []
    processed_images = []

    for _, row in df.iterrows():
        processed_path = Path(row["processed_png_path"])

        if not processed_path.exists():
            continue

        img = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # 保证尺寸一致
        img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

        hog_feature = extract_hog_feature(img)
        hu_feature = extract_hu_moments(img)

        valid_rows.append(row)
        hog_features.append(hog_feature)
        hu_features.append(hu_feature)
        processed_images.append(img)

    if not valid_rows:
        raise ValueError("没有成功加载任何标准符号图像。")

    metadata = pd.DataFrame(valid_rows).reset_index(drop=True)
    hog_matrix = np.vstack(hog_features)
    hu_matrix = np.vstack(hu_features)

    return metadata, hog_matrix, hu_matrix, processed_images


# =========================
# 7. 多特征融合匹配
# =========================
def match_input_image_v2(image_path: Path, top_k: int = 5):
    """
    使用 HOG + SSIM + Hu Moments 多特征融合进行图像匹配。
    """
    metadata, hog_matrix, hu_matrix, processed_images = load_sign_library_features()

    input_img = preprocess_input_image(image_path, OUTPUT_SIZE)

    input_hog = extract_hog_feature(input_img).reshape(1, -1)
    input_hu = extract_hu_moments(input_img)

    hog_scores = cosine_similarity(input_hog, hog_matrix)[0]

    ssim_scores = []
    hu_scores = []

    for idx in range(len(metadata)):
        standard_img = processed_images[idx]
        standard_hu = hu_matrix[idx]

        ssim_score = calculate_ssim_similarity(input_img, standard_img)
        hu_score = hu_similarity(input_hu, standard_hu)

        ssim_scores.append(ssim_score)
        hu_scores.append(hu_score)

    ssim_scores = np.array(ssim_scores)
    hu_scores = np.array(hu_scores)

    # 多特征融合权重
    final_scores = (
        0.45 * hog_scores +
        0.35 * ssim_scores +
        0.20 * hu_scores
    )

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()

    results["hog_score"] = hog_scores[top_indices]
    results["ssim_score"] = ssim_scores[top_indices]
    results["hu_score"] = hu_scores[top_indices]
    results["final_score"] = final_scores[top_indices]

    results["confidence_level"] = results["final_score"].apply(get_confidence_level)

    return results, input_img


def get_confidence_level(score):
    """
    根据最终融合分数给出置信度等级。
    """
    score = float(score)

    if score >= 0.75:
        return "高"
    elif score >= 0.55:
        return "中"
    else:
        return "低"


# =========================
# 8. 批量自测
# =========================
def batch_test_standard_signs():
    """
    用标准 sign_png 中的图片做自测，检查 Top-1 是否识别正确。
    """
    print("\n" + "=" * 80)
    print("标准符号库自测")
    print("=" * 80)

    df = pd.read_csv(SIGN_METADATA_CSV, dtype=str).fillna("")

    total = 0
    correct = 0
    records = []

    for _, row in df.iterrows():
        gardiner_code = row["gardiner_code"]

        # 原始标准图路径
        original_png = SIGN_DIR / "sign_png" / f"{gardiner_code}.png"

        if not original_png.exists():
            continue

        results, _ = match_input_image_v2(original_png, top_k=5)

        top1 = results.iloc[0]
        predicted = top1["gardiner_code"]
        final_score = float(top1["final_score"])
        confidence = top1["confidence_level"]

        is_correct = predicted == gardiner_code

        total += 1
        correct += int(is_correct)

        records.append({
            "true_code": gardiner_code,
            "predicted_code": predicted,
            "correct": is_correct,
            "final_score": round(final_score, 4),
            "confidence_level": confidence
        })

        mark = "✅" if is_correct else "❌"
        print(
            f"{mark} true={gardiner_code:<5} pred={predicted:<5} "
            f"score={final_score:.4f} confidence={confidence}"
        )

    acc = correct / total if total > 0 else 0

    print("\n" + "-" * 80)
    print(f"自测数量：{total}")
    print(f"Top-1 正确数：{correct}")
    print(f"Top-1 准确率：{acc:.2%}")

    out_path = SIGN_DIR / "sign_match_v2_self_test.csv"
    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"自测结果已保存：{out_path}")


# =========================
# 9. 主程序：交互测试
# =========================
def main():
    print("=" * 80)
    print("古埃及符号图像匹配测试 V2：HOG + SSIM + Hu Moments")
    print("=" * 80)
    print("输入图片路径进行测试。")
    print("输入 test 运行标准符号库自测。")
    print("输入 q 退出。")
    print("\n示例：")
    print(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project\data_sign_demo\sign_png\N5.png")

    while True:
        user_input = input("\n请输入图片路径 / test / q：").strip().strip('"')

        if user_input.lower() == "q":
            print("已退出。")
            break

        if user_input.lower() == "test":
            batch_test_standard_signs()
            continue

        image_path = Path(user_input)

        if not image_path.exists():
            print("图片不存在，请重新输入。")
            continue

        try:
            results, _ = match_input_image_v2(image_path, top_k=5)

            print("\n匹配结果 Top-5：")
            print("-" * 80)

            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                print(f"Rank {rank}")
                print("Gardiner:", row.get("gardiner_code", ""))
                print("Unicode:", row.get("unicode_char", ""))
                print("中文名:", row.get("zh_name", ""))
                print("英文名:", row.get("en_name", ""))
                print("相关检索词:", row.get("related_terms", ""))
                print("HOG:", round(float(row.get("hog_score", 0)), 4))
                print("SSIM:", round(float(row.get("ssim_score", 0)), 4))
                print("Hu:", round(float(row.get("hu_score", 0)), 4))
                print("融合分数:", round(float(row.get("final_score", 0)), 4))
                print("置信度:", row.get("confidence_level", ""))
                print("-" * 80)

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()