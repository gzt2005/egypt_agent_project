from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

DINO_RELIEF_DIR = SIGN_DIR / "sign_dinov2_relief_features"
DINO_RELIEF_METADATA_CSV = DINO_RELIEF_DIR / "dinov2_relief_metadata.csv"
DINO_RELIEF_EMBEDDINGS_PATH = DINO_RELIEF_DIR / "dinov2_relief_embeddings.npy"

DINO_RESULT_CSV = SIGN_DIR / "dinov2_multiview_results.csv"

MODEL_NAME = "facebook/dinov2-small"

VIEW_SIZE = 224


# =========================================================
# 2. 图片读取与格式转换
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
# 3. 查询图多视图生成
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
    通用主体裁剪，不针对任何单个符号。
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

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
    将查询 mask 转成浮雕风格查询图。
    """
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    mask_f = mask.astype("float32") / 255.0

    base = np.ones((size, size, 3), dtype=np.float32) * np.array(
        [185, 155, 115],
        dtype=np.float32
    )

    rng = np.random.default_rng(123)
    noise = rng.normal(0, 13, (size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=8, sigmaY=8)

    for c in range(3):
        base[:, :, c] += noise

    matrix_high = np.float32([[1, 0, -3], [0, 1, -3]])
    matrix_shadow = np.float32([[1, 0, 4], [0, 1, 4]])

    high = cv2.warpAffine(
        mask_f,
        matrix_high,
        (size, size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    shadow = cv2.warpAffine(
        mask_f,
        matrix_shadow,
        (size, size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

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


def generate_query_views(image_path: Path):
    """
    生成 DINOv2 查询视图。
    和 CLIP 多视图版本保持一致，便于对比实验。
    """
    img_bgr = read_image_bgr(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    views = {}

    original_square = resize_keep_ratio_to_square(
        img_bgr,
        VIEW_SIZE,
        bg_color=255
    )
    views["original"] = cv_bgr_to_pil_rgb(original_square)

    cropped_gray = crop_symbol_region_from_gray(gray)
    cropped_square = resize_keep_ratio_to_square(
        cropped_gray,
        VIEW_SIZE,
        bg_color=255
    )
    views["cropped_gray"] = cv_gray_to_pil_rgb(cropped_square)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(cropped_gray)
    enhanced_square = resize_keep_ratio_to_square(
        enhanced,
        VIEW_SIZE,
        bg_color=255
    )
    views["clahe_gray"] = cv_gray_to_pil_rgb(enhanced_square)

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
    edges = cv2.dilate(
        edges,
        np.ones((2, 2), np.uint8),
        iterations=1
    )
    edges_square = resize_keep_ratio_to_square(
        edges,
        VIEW_SIZE,
        bg_color=0
    )
    views["canny_edge"] = cv_gray_to_pil_rgb(edges_square)

    blur = cv2.GaussianBlur(cropped_gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    )
    binary_square = resize_keep_ratio_to_square(
        binary,
        VIEW_SIZE,
        bg_color=0
    )
    views["binary_mask"] = cv_gray_to_pil_rgb(binary_square)

    relief_query = make_stone_like_from_mask(binary_square, VIEW_SIZE)
    views["query_relief_style"] = cv_bgr_to_pil_rgb(
        cv2.cvtColor(relief_query, cv2.COLOR_RGB2BGR)
    )

    return views


# =========================================================
# 4. DINOv2 模型与向量
# =========================================================
def load_dinov2_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 90)
    print("加载 DINOv2 模型")
    print("=" * 90)
    print("模型名称：", MODEL_NAME)
    print("运行设备：", device)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return model, processor, device


def encode_images_dinov2(images, model, processor, device):
    """
    输入 PIL Image 列表，输出归一化后的 DINOv2 图像向量。
    """
    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "last_hidden_state"):
        image_features = outputs.last_hidden_state[:, 0, :]
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        image_features = outputs.pooler_output
    else:
        raise TypeError(f"无法识别 DINOv2 输出类型：{type(outputs)}")

    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-12)

    return image_features.cpu().numpy()


def load_dinov2_relief_library():
    if not DINO_RELIEF_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到 DINOv2 元数据：{DINO_RELIEF_METADATA_CSV}")

    if not DINO_RELIEF_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"未找到 DINOv2 embedding：{DINO_RELIEF_EMBEDDINGS_PATH}")

    metadata = pd.read_csv(DINO_RELIEF_METADATA_CSV, dtype=str).fillna("")
    embeddings = np.load(DINO_RELIEF_EMBEDDINGS_PATH)

    return metadata, embeddings


# =========================================================
# 5. DINOv2 多视图检索
# =========================================================
def match_dinov2_multiview(image_path: Path, top_k: int = 50):
    metadata, relief_embeddings = load_dinov2_relief_library()

    model, processor, device = load_dinov2_model()

    views = generate_query_views(image_path)

    view_names = list(views.keys())
    view_images = [views[name] for name in view_names]

    query_embeddings = encode_images_dinov2(
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
        scored["dinov2_score"] = similarities

        scored_all_views.append(scored)

    scored_all = pd.concat(scored_all_views, ignore_index=True)

    # 按 sign_id 聚合，每个符号只保留最高分
    idx_best = scored_all.groupby("sign_id")["dinov2_score"].idxmax()
    best_per_sign = scored_all.loc[idx_best].copy()

    best_per_sign = best_per_sign.sort_values(
        by="dinov2_score",
        ascending=False
    ).reset_index(drop=True)

    best_per_sign["global_rank"] = best_per_sign.index + 1

    # S34 诊断
    s34_rows = best_per_sign[
        best_per_sign["gardiner_code"].astype(str).str.strip() == "S34"
    ]

    print("\n" + "=" * 90)
    print("DINOv2 检索诊断")
    print("=" * 90)

    if len(s34_rows) > 0:
        print("S34｜生命符号 当前全局排名：", int(s34_rows.iloc[0]["global_rank"]))
        print("S34｜生命符号 当前相似度：", round(float(s34_rows.iloc[0]["dinov2_score"]), 4))
        print("S34 最佳查询视图：", s34_rows.iloc[0].get("view_name", ""))
        print("S34 最佳浮雕风格：", s34_rows.iloc[0].get("relief_variant", ""))
    else:
        print("未在全量库中找到 S34 记录，请检查人工注释。")

    top_results = best_per_sign.head(top_k).copy()

    return top_results, view_names, best_per_sign


# =========================================================
# 6. 输出
# =========================================================
def get_confidence_level(score):
    score = float(score)

    if score >= 0.75:
        return "高"
    elif score >= 0.55:
        return "中"
    else:
        return "低"


def print_results(results: pd.DataFrame, view_names):
    results = results.copy()
    results["confidence_level"] = results["dinov2_score"].apply(get_confidence_level)

    results.to_csv(DINO_RESULT_CSV, index=False, encoding="utf-8-sig")

    print("\n结果已保存到：", DINO_RESULT_CSV)

    print("\n" + "=" * 90)
    print("DINOv2 多视图匹配结果 Top-K")
    print("=" * 90)
    print("查询视图：", ", ".join(view_names))

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        gardiner_code = row.get("gardiner_code", "")
        zh_name = row.get("zh_name", "")
        en_name = row.get("en_name", "")
        unicode_char = row.get("unicode_char", "")
        unicode_codepoint = row.get("unicode_codepoint", "")
        auto_label = row.get("auto_label", "")
        related_terms = row.get("related_terms", "")
        has_manual_annotation = row.get("has_manual_annotation", "")

        display_code = gardiner_code if gardiner_code else auto_label
        display_zh = zh_name if zh_name else "暂无中文注释"
        display_en = en_name if en_name else "暂无英文注释"

        print("-" * 90)
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
        print("最佳查询视图:", row.get("view_name", ""))
        print("最佳浮雕风格:", row.get("relief_variant", ""))
        print("DINOv2 相似度:", round(float(row.get("dinov2_score", 0)), 4))
        print("置信度:", row.get("confidence_level", ""))


def search_uploaded_image(image_bgr, top_k=10):
    """
    Streamlit 上传图片后调用的入口函数。
    输入：OpenCV BGR 图像
    输出：Top-K 检索结果 DataFrame
    """
    results, view_names, full_results = match_dinov2_multiview_from_array(
        image_bgr=image_bgr,
        top_k=top_k
    )

    return results, view_names, full_results