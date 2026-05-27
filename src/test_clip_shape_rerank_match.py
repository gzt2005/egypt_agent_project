from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

CLIP_RELIEF_DIR = SIGN_DIR / "sign_clip_relief_features"
CLIP_RELIEF_METADATA_CSV = CLIP_RELIEF_DIR / "clip_relief_metadata.csv"
CLIP_RELIEF_EMBEDDINGS_PATH = CLIP_RELIEF_DIR / "clip_relief_embeddings.npy"

RERANK_OUTPUT_CSV = SIGN_DIR / "clip_shape_rerank_results.csv"

MODEL_NAME = "openai/clip-vit-base-patch32"

VIEW_SIZE = 224
SHAPE_SIZE = 128


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


def read_image_gray(image_path: Path):
    """
    支持 Windows 中文路径读取灰度图。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return None

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    return img


def cv_bgr_to_pil_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def cv_gray_to_pil_rgb(img_gray):
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb)


# =========================================================
# 3. 查询图多视图生成：用于 CLIP 召回
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
    将输入图 mask 转成类似浮雕风格的查询图。
    这是通用查询风格对齐，不针对任何符号。
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
    给输入真实图像生成多个 CLIP 查询视图。
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
# 4. 查询图形状预处理：用于第二阶段重排序
# =========================================================
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


def preprocess_query_for_shape(image_path: Path):
    """
    将真实查询图转成更适合形状比较的二值图和边缘图。
    """
    img_bgr = read_image_bgr(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cropped = crop_symbol_region_from_gray(gray)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(cropped)

    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

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

    query_binary = crop_center_resize(binary, SHAPE_SIZE)

    edges = cv2.Canny(
        enhanced,
        threshold1=40,
        threshold2=140
    )
    edges = cv2.dilate(
        edges,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    query_edge = crop_center_resize(edges, SHAPE_SIZE)

    return query_binary, query_edge


def standard_to_edge(img):
    img = cv2.resize(
        img,
        (SHAPE_SIZE, SHAPE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    edges = cv2.Canny(
        img,
        threshold1=50,
        threshold2=150
    )

    edges = cv2.dilate(
        edges,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    return edges


# =========================================================
# 5. 通用形状特征
# =========================================================
def extract_hog_feature(img):
    img = cv2.resize(
        img,
        (SHAPE_SIZE, SHAPE_SIZE),
        interpolation=cv2.INTER_AREA
    )

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
    img = cv2.resize(
        img,
        (SHAPE_SIZE, SHAPE_SIZE),
        interpolation=cv2.INTER_AREA
    )

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
    通用结构特征，不针对任何符号：
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


def minmax_normalize(values):
    arr = np.array(values, dtype=np.float32)

    v_min = arr.min()
    v_max = arr.max()

    if abs(v_max - v_min) < 1e-12:
        return np.ones_like(arr) * 0.5

    return (arr - v_min) / (v_max - v_min)


# =========================================================
# 6. CLIP 模型和向量库
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


def load_clip_relief_library():
    if not CLIP_RELIEF_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到浮雕 CLIP 元数据：{CLIP_RELIEF_METADATA_CSV}")

    if not CLIP_RELIEF_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"未找到浮雕 CLIP embedding：{CLIP_RELIEF_EMBEDDINGS_PATH}")

    metadata = pd.read_csv(CLIP_RELIEF_METADATA_CSV, dtype=str).fillna("")
    embeddings = np.load(CLIP_RELIEF_EMBEDDINGS_PATH)

    return metadata, embeddings


# =========================================================
# 7. 第一阶段：CLIP 多视图召回
# =========================================================
def clip_multiview_recall(image_path: Path, recall_k: int = 100):
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
        scored["clip_score"] = similarities

        scored_all_views.append(scored)

    scored_all = pd.concat(scored_all_views, ignore_index=True)

    idx_best = scored_all.groupby("sign_id")["clip_score"].idxmax()
    best_per_sign = scored_all.loc[idx_best].copy()

    best_per_sign = best_per_sign.sort_values(
        by="clip_score",
        ascending=False
    ).reset_index(drop=True)

    best_per_sign["clip_recall_rank"] = best_per_sign.index + 1

    recall_results = best_per_sign.head(recall_k).copy()

    return recall_results, best_per_sign, view_names


# =========================================================
# 8. 第二阶段：Top-N 候选通用形状重排序
# =========================================================
def rerank_candidates_by_shape(image_path: Path, recall_results: pd.DataFrame):
    query_binary, query_edge = preprocess_query_for_shape(image_path)

    query_binary_hog = extract_hog_feature(query_binary).reshape(1, -1)
    query_edge_hog = extract_hog_feature(query_edge).reshape(1, -1)
    query_hu = extract_hu_moments(query_binary)
    query_structure = extract_structure_features(query_binary)

    records = []

    for _, row in recall_results.iterrows():
        processed_path = row.get("processed_png_path", "")

        if not processed_path:
            continue

        candidate_img = read_image_gray(Path(processed_path))

        if candidate_img is None:
            continue

        candidate_img = cv2.resize(
            candidate_img,
            (SHAPE_SIZE, SHAPE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        candidate_edge = standard_to_edge(candidate_img)

        candidate_binary_hog = extract_hog_feature(candidate_img).reshape(1, -1)
        candidate_edge_hog = extract_hog_feature(candidate_edge).reshape(1, -1)

        candidate_hu = extract_hu_moments(candidate_img)
        candidate_structure = extract_structure_features(candidate_img)

        binary_hog_score = cosine_similarity(
            query_binary_hog,
            candidate_binary_hog
        )[0][0]

        edge_hog_score = cosine_similarity(
            query_edge_hog,
            candidate_edge_hog
        )[0][0]

        hu_score = hu_similarity(query_hu, candidate_hu)
        structure_score = structure_similarity(query_structure, candidate_structure)

        new_row = row.to_dict()
        new_row["binary_hog_score"] = binary_hog_score
        new_row["edge_hog_score"] = edge_hog_score
        new_row["hu_score"] = hu_score
        new_row["structure_score"] = structure_score

        records.append(new_row)

    rerank_df = pd.DataFrame(records)

    if len(rerank_df) == 0:
        return rerank_df

    rerank_df["clip_norm"] = minmax_normalize(rerank_df["clip_score"])
    rerank_df["binary_hog_norm"] = minmax_normalize(rerank_df["binary_hog_score"])
    rerank_df["edge_hog_norm"] = minmax_normalize(rerank_df["edge_hog_score"])
    rerank_df["hu_norm"] = minmax_normalize(rerank_df["hu_score"])
    rerank_df["structure_norm"] = minmax_normalize(rerank_df["structure_score"])

    # 通用二阶段重排序分数：
    # CLIP 负责召回，形状特征负责精排。
    # 不包含 S34 特殊规则，不包含人工注释加权。
    # CLIP 主导型重排序：
# 当前实验表明传统形状特征在真实浮雕图上不够稳定，
# 因此让 CLIP 多视图召回分数作为主导，形状特征只做弱辅助。
    rerank_df["final_rerank_score"] = (
    0.85 * rerank_df["clip_norm"] +
    0.06 * rerank_df["binary_hog_norm"] +
    0.04 * rerank_df["edge_hog_norm"] +
    0.03 * rerank_df["hu_norm"] +
    0.02 * rerank_df["structure_norm"]
)

    rerank_df = rerank_df.sort_values(
        by="final_rerank_score",
        ascending=False
    ).reset_index(drop=True)

    rerank_df["final_rank"] = rerank_df.index + 1

    return rerank_df


# =========================================================
# 9. 置信度与输出
# =========================================================
def get_confidence_level(score):
    score = float(score)

    if score >= 0.75:
        return "高"
    elif score >= 0.55:
        return "中"
    else:
        return "低"


def print_results(rerank_df: pd.DataFrame, full_recall_df: pd.DataFrame, top_k: int = 20):
    if len(rerank_df) == 0:
        print("没有可输出的重排序结果。")
        return

    rerank_df["confidence_level"] = rerank_df["final_rerank_score"].apply(
        get_confidence_level
    )

    rerank_df.to_csv(RERANK_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("二阶段检索诊断")
    print("=" * 100)

    s34_recall = full_recall_df[
        full_recall_df["gardiner_code"].astype(str).str.strip() == "S34"
    ]

    if len(s34_recall) > 0:
        print("S34 在 CLIP 召回阶段排名：", int(s34_recall.iloc[0]["clip_recall_rank"]))
        print("S34 在 CLIP 召回阶段分数：", round(float(s34_recall.iloc[0]["clip_score"]), 4))
    else:
        print("S34 未出现在 CLIP 全量召回结果中。")

    s34_rerank = rerank_df[
        rerank_df["gardiner_code"].astype(str).str.strip() == "S34"
    ]

    if len(s34_rerank) > 0:
        print("S34 在形状重排序后排名：", int(s34_rerank.iloc[0]["final_rank"]))
        print("S34 最终重排序分数：", round(float(s34_rerank.iloc[0]["final_rerank_score"]), 4))
    else:
        print("S34 未进入本次重排序候选集。")

    print("\n结果已保存到：", RERANK_OUTPUT_CSV)

    print("\n" + "=" * 100)
    print(f"CLIP 召回 + 通用形状重排序结果 Top-{top_k}")
    print("=" * 100)

    show_df = rerank_df.head(top_k).copy()

    for rank, (_, row) in enumerate(show_df.iterrows(), start=1):
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

        print("-" * 100)
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

        print("CLIP 召回排名:", row.get("clip_recall_rank", ""))
        print("最佳查询视图:", row.get("view_name", ""))
        print("最佳浮雕风格:", row.get("relief_variant", ""))

        print("CLIP 原始分数:", round(float(row.get("clip_score", 0)), 4))
        print("Binary HOG:", round(float(row.get("binary_hog_score", 0)), 4))
        print("Edge HOG:", round(float(row.get("edge_hog_score", 0)), 4))
        print("Hu:", round(float(row.get("hu_score", 0)), 4))
        print("结构分:", round(float(row.get("structure_score", 0)), 4))
        print("最终重排序分数:", round(float(row.get("final_rerank_score", 0)), 4))
        print("置信度:", row.get("confidence_level", ""))


# =========================================================
# 10. 主程序
# =========================================================
def main():
    print("=" * 100)
    print("V2.12 CLIP 召回 + 通用形状重排序匹配测试")
    print("=" * 100)
    print("第一阶段：CLIP 多视图 + 浮雕增强库召回 Top100")
    print("第二阶段：在 Top100 内用 HOG / Hu / 结构特征进行通用重排序")
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
            recall_results, full_recall_df, view_names = clip_multiview_recall(
                image_path=image_path,
                recall_k=100
            )

            rerank_df = rerank_candidates_by_shape(
                image_path=image_path,
                recall_results=recall_results
            )

            print_results(
                rerank_df=rerank_df,
                full_recall_df=full_recall_df,
                top_k=20
            )

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()