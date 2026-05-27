from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

CLIP_RELIEF_DIR = SIGN_DIR / "sign_clip_relief_features"
CLIP_RELIEF_METADATA_CSV = CLIP_RELIEF_DIR / "clip_relief_metadata.csv"
CLIP_RELIEF_EMBEDDINGS_PATH = CLIP_RELIEF_DIR / "clip_relief_embeddings.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"

VIEW_SIZE = 224


# =========================================================
# 2. 图片读取与保存兼容中文路径
# =========================================================
def read_image_bgr(image_path: Path):
    """
    支持 Windows 中文路径读取图片，返回 BGR 图。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    return img


def cv_bgr_to_pil_rgb(img_bgr):
    """
    OpenCV BGR 转 PIL RGB。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def cv_gray_to_pil_rgb(img_gray):
    """
    灰度图转 PIL RGB。
    """
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb)


# =========================================================
# 3. 通用图像处理函数
# =========================================================
def resize_keep_ratio_to_square(img, size=224, bg_color=255):
    """
    等比例缩放到正方形画布，避免直接拉伸。
    支持灰度图和 BGR 图。
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(img.shape) == 2:
        canvas = np.ones((size, size), dtype=np.uint8) * bg_color
        y = (size - new_h) // 2
        x = (size - new_w) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
    else:
        canvas = np.ones((size, size, 3), dtype=np.uint8) * bg_color
        y = (size - new_h) // 2
        x = (size - new_w) // 2
        canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def crop_symbol_region_from_gray(gray):
    """
    尝试从真实图片里裁剪主体符号区域。
    这是通用裁剪，不针对任何符号。
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 自适应阈值更适合光照不均的浮雕图
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(binary)

    if coords is None:
        return gray

    x, y, w, h = cv2.boundingRect(coords)

    # 加边距，避免裁掉符号边缘
    pad = int(max(w, h) * 0.12)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)

    cropped = gray[y1:y2, x1:x2]

    if cropped.size == 0:
        return gray

    return cropped


def make_stone_like_from_mask(mask, size=224):
    """
    将输入图的二值 mask 转成类似浮雕风格的查询图。
    这是为了让查询图与浮雕增强库风格更接近。
    """
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    mask_f = mask.astype("float32") / 255.0

    # 石壁背景
    base = np.ones((size, size, 3), dtype=np.float32) * np.array([185, 155, 115], dtype=np.float32)

    # 背景纹理
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 13, (size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=8, sigmaY=8)

    for c in range(3):
        base[:, :, c] += noise

    # 阴影/高光
    matrix_high = np.float32([[1, 0, -3], [0, 1, -3]])
    matrix_shadow = np.float32([[1, 0, 4], [0, 1, 4]])

    high = cv2.warpAffine(mask_f, matrix_high, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shadow = cv2.warpAffine(mask_f, matrix_shadow, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    for c in range(3):
        base[:, :, c] += high * 55
        base[:, :, c] -= shadow * 55
        base[:, :, c] -= mask_f * 18

    edge = cv2.Canny(mask, 50, 150).astype("float32") / 255.0
    edge = cv2.GaussianBlur(edge, (3, 3), 0)

    for c in range(3):
        base[:, :, c] -= edge * 25

    base = np.clip(base, 0, 255).astype(np.uint8)
    return base


# =========================================================
# 4. 生成查询图片多视图
# =========================================================
def generate_query_views(image_path: Path):
    """
    对用户输入图片生成多个 CLIP 查询视图。
    不针对任何单个符号，属于通用预处理。
    """
    img_bgr = read_image_bgr(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    views = {}

    # 1. 原始图等比例缩放
    original_square = resize_keep_ratio_to_square(img_bgr, VIEW_SIZE, bg_color=255)
    views["original"] = cv_bgr_to_pil_rgb(original_square)

    # 2. 主体裁剪后的原图风格
    cropped_gray = crop_symbol_region_from_gray(gray)
    cropped_square = resize_keep_ratio_to_square(cropped_gray, VIEW_SIZE, bg_color=255)
    views["cropped_gray"] = cv_gray_to_pil_rgb(cropped_square)

    # 3. CLAHE 对比度增强图
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cropped_gray)
    enhanced_square = resize_keep_ratio_to_square(enhanced, VIEW_SIZE, bg_color=255)
    views["clahe_gray"] = cv_gray_to_pil_rgb(enhanced_square)

    # 4. Canny 边缘图
    filtered = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
    edges = cv2.Canny(filtered, threshold1=40, threshold2=140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_square = resize_keep_ratio_to_square(edges, VIEW_SIZE, bg_color=0)
    views["canny_edge"] = cv_gray_to_pil_rgb(edges_square)

    # 5. 自适应阈值二值图
    blur = cv2.GaussianBlur(cropped_gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary_square = resize_keep_ratio_to_square(binary, VIEW_SIZE, bg_color=0)
    views["binary_mask"] = cv_gray_to_pil_rgb(binary_square)

    # 6. 查询图自身的浮雕风格模拟图
    relief_query = make_stone_like_from_mask(binary_square, VIEW_SIZE)
    views["query_relief_style"] = cv_bgr_to_pil_rgb(cv2.cvtColor(relief_query, cv2.COLOR_RGB2BGR))

    return views


# =========================================================
# 5. 加载 CLIP 模型
# =========================================================
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("加载 CLIP 模型")
    print("=" * 80)
    print("模型名称：", MODEL_NAME)
    print("运行设备：", device)

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return model, processor, device


def encode_images_clip(images, model, processor, device):
    """
    输入 PIL Image 列表，输出归一化后的 CLIP 图像向量。
    """
    inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    if isinstance(outputs, torch.Tensor):
        image_features = outputs
    elif hasattr(outputs, "image_embeds"):
        image_features = outputs.image_embeds
    elif hasattr(outputs, "pooler_output"):
        image_features = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        image_features = outputs.last_hidden_state[:, 0, :]
    else:
        raise TypeError(f"无法识别 CLIP 输出类型：{type(outputs)}")

    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-12)

    return image_features.cpu().numpy()


# =========================================================
# 6. 加载浮雕增强 CLIP 库
# =========================================================
def load_clip_relief_library():
    if not CLIP_RELIEF_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到浮雕 CLIP 元数据：{CLIP_RELIEF_METADATA_CSV}")

    if not CLIP_RELIEF_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"未找到浮雕 CLIP embedding：{CLIP_RELIEF_EMBEDDINGS_PATH}")

    metadata = pd.read_csv(CLIP_RELIEF_METADATA_CSV, dtype=str).fillna("")
    embeddings = np.load(CLIP_RELIEF_EMBEDDINGS_PATH)

    return metadata, embeddings


# =========================================================
# 7. 置信度
# =========================================================
def get_confidence_level(score):
    score = float(score)

    if score >= 0.35:
        return "高"
    elif score >= 0.25:
        return "中"
    else:
        return "低"


# =========================================================
# 8. 多视图 CLIP 浮雕匹配
# =========================================================
# =========================================================
# 8. 多视图 CLIP 浮雕匹配
# =========================================================
def match_clip_relief_multiview(image_path: Path, top_k: int = 50):
    """
    输入真实浮雕图片：
    1. 生成多个查询视图
    2. 每个查询视图提取 CLIP embedding
    3. 与浮雕增强库进行相似度匹配
    4. 按 sign_id 聚合，每个符号保留最高分
    5. 输出 Top-K，并诊断 S34 的全局排名
    """
    metadata, relief_embeddings = load_clip_relief_library()

    model, processor, device = load_clip_model()

    views = generate_query_views(image_path)

    view_names = list(views.keys())
    view_images = [views[name] for name in view_names]

    query_embeddings = encode_images_clip(
        view_images,
        model,
        processor,
        device
    )

    scored_all_views = []

    for view_idx, view_name in enumerate(view_names):
        query_embedding = query_embeddings[view_idx].reshape(1, -1)

        similarities = cosine_similarity(
            query_embedding,
            relief_embeddings
        )[0]

        scored = metadata.copy()
        scored["view_name"] = view_name
        scored["view_score"] = similarities

        scored_all_views.append(scored)

    scored_all = pd.concat(scored_all_views, ignore_index=True)

    # 每个 sign_id 在所有查询视图、所有浮雕风格里取最高分
    idx_best = scored_all.groupby("sign_id")["view_score"].idxmax()
    best_per_sign = scored_all.loc[idx_best].copy()

    best_per_sign = best_per_sign.sort_values(
        by="view_score",
        ascending=False
    ).reset_index(drop=True)

    best_per_sign["global_rank"] = best_per_sign.index + 1

    # 单独检查 S34 的全局排名
    s34_rows = best_per_sign[
        best_per_sign["gardiner_code"].astype(str).str.strip() == "S34"
    ]

    if len(s34_rows) > 0:
        s34_rank = int(s34_rows.iloc[0]["global_rank"])
        s34_score = float(s34_rows.iloc[0]["view_score"])

        print("\n" + "=" * 80)
        print("S34｜生命符号 检索诊断")
        print("=" * 80)
        print(f"S34 当前全局排名：{s34_rank}")
        print(f"S34 当前相似度：{s34_score:.4f}")
        print("最佳查询视图：", s34_rows.iloc[0].get("view_name", ""))
        print("最佳浮雕风格：", s34_rows.iloc[0].get("relief_variant", ""))
    else:
        print("\n未在全量库中找到 S34 记录，请检查人工注释表。")

    top_results = best_per_sign.head(top_k).copy()

    top_results["confidence_level"] = top_results["view_score"].apply(
        get_confidence_level
    )

    return top_results, view_names

# =========================================================
# 9. 输出结果
# =========================================================
def print_match_results(results, view_names):
    out_path = PROJECT_DIR / "data_sign_demo" / "clip_multiview_top_results.csv"
    results.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nTop-K 结果已保存到：{out_path}")
    print("\nCLIP 浮雕增强多视图匹配结果 Top-K：")
    print("查询视图：", ", ".join(view_names))
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
        relief_variant = row.get("relief_variant", "")
        relief_png_path = row.get("relief_png_path", "")
        view_name = row.get("view_name", "")

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
        print("最佳查询视图:", view_name)
        print("最佳浮雕风格:", relief_variant)
        print("浮雕增强图路径:", relief_png_path)
        print("CLIP 多视图相似度:", round(float(row.get("view_score", 0)), 4))
        print("置信度:", row.get("confidence_level", ""))
        print("-" * 120)


# =========================================================
# 10. 主程序
# =========================================================
def main():
    print("=" * 120)
    print("V2.11 CLIP 浮雕增强多视图古埃及符号图像匹配测试")
    print("=" * 120)
    print("候选库：1072 个符号 × 6 种浮雕风格 = 6432 个增强样本")
    print("查询侧：原始图 / 裁剪图 / 增强灰度 / 边缘图 / 二值图 / 查询浮雕模拟图")
    print("策略：多查询视图 CLIP embedding + 浮雕增强库 + sign_id 聚合")
    print("不包含任何单独符号加分，不包含人工注释加权。")
    print("\n示例：")
    print(r"C:\Users\GE ZITONG\Desktop\安卡1.png")

    while True:
        user_input = input("\n请输入图片路径 / q：").strip().strip('"')

        if user_input.lower() == "q":
            print("已退出。")
            break

        image_path = Path(user_input)

        try:
            results, view_names = match_clip_relief_multiview(
            image_path=image_path,
            top_k=50
)

            print_match_results(results, view_names)

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()