from pathlib import Path
import re
import time
import sqlite3

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity

# ResNet18 单符号识别模块
from src.resnet18_hieroglyph_predictor import HieroglyphResNet18Predictor


# =========================
# 1. 页面设置
# =========================
st.set_page_config(
    page_title="古埃及文字智能检索系统",
    page_icon="𓂀",
    layout="wide"
)


# =========================
# 2. 路径设置
# =========================
PROJECT_DIR = Path(__file__).parent

DB_PATH = PROJECT_DIR / "database_demo" / "egypt_demo.db"

SEMANTIC_DIR = PROJECT_DIR / "data_semantic_demo"
SEMANTIC_EMBEDDINGS_PATH = SEMANTIC_DIR / "semantic_embeddings.npy"
SEMANTIC_METADATA_PATH = SEMANTIC_DIR / "semantic_metadata.csv"
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EVALUATION_DIR = PROJECT_DIR / "evaluation_results"
EVALUATION_RESULTS_CSV = EVALUATION_DIR / "evaluation_results.csv"

SIGN_DIR = PROJECT_DIR / "data_sign_demo"
SIGN_PNG_DIR = SIGN_DIR / "sign_png"
SIGN_PROCESSED_DIR = SIGN_DIR / "sign_processed"
SIGN_METADATA_PROCESSED_CSV = SIGN_DIR / "hieroglyph_signs_processed.csv"
SIGN_OUTPUT_SIZE = 128

DINO_RELIEF_DIR = SIGN_DIR / "sign_dinov2_relief_features"
DINO_RELIEF_METADATA_CSV = DINO_RELIEF_DIR / "dinov2_relief_metadata.csv"
DINO_RELIEF_EMBEDDINGS_PATH = DINO_RELIEF_DIR / "dinov2_relief_embeddings.npy"
DINO_MODEL_NAME = "facebook/dinov2-small"
DINO_VIEW_SIZE = 224



# =========================
# 3. 工具函数
# =========================
def normalize_query_term(term: str) -> str:
    """
    将用户输入或扩展词归一化。
    例如：nṯr -> ntr；ḏd -> dd；ꜥnḫ -> anh；Wsjr -> wsjr。
    """
    if not isinstance(term, str):
        return ""

    term = term.strip().lower()

    mapping = {
        "ꜣ": "a",
        "ꜥ": "a",
        "ȝ": "a",
        "ʾ": "a",
        "ḏ": "d",
        "ḥ": "h",
        "ḫ": "h",
        "ẖ": "h",
        "ḳ": "q",
        "š": "s",
        "ṯ": "t",
        "ṱ": "t",
        "ỉ": "i",
        "ī": "i",
        "ū": "u",
        "ꞽ": "i",
    }

    for old, new in mapping.items():
        term = term.replace(old, new)

    term = re.sub(r"[^a-z0-9\.\-_]", "", term)
    return term


def contains_chinese(text: str) -> bool:
    """判断是否包含中文。"""
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


def limit_topic_tags(topic_tags_zh: str, max_tags: int = 5) -> str:
    """限制中文主题标签展示数量，避免页面标签过多。"""
    if not isinstance(topic_tags_zh, str) or not topic_tags_zh.strip():
        return ""

    tags = [tag.strip() for tag in topic_tags_zh.split("；") if tag.strip()]
    tags = list(dict.fromkeys(tags))
    return "；".join(tags[:max_tags])


def generate_chinese_hint(query, matched_terms, matched_fields):
    """根据查询词、命中词和命中字段生成简单中文解释。"""
    if not matched_terms:
        matched_terms = "相关词项"

    if not matched_fields:
        matched_fields = "相关字段"

    if contains_chinese(query):
        return (
            f"该结果与“{query}”主题相关，系统通过扩展词 {matched_terms} 命中原始语料。"
            f"命中字段包括 {matched_fields}，可结合原始译文、古埃及转写、lemma、MDC 和中文主题摘要判断其主题相关性。"
        )

    return (
        f"该结果命中检索词 {matched_terms}，命中字段包括 {matched_fields}。"
        f"可结合原始译文、古埃及转写、lemma、MDC 和中文知识增强信息判断其文本证据价值。"
    )


def get_sqlite_connection():
    """获取 SQLite 数据库连接。"""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"数据库文件不存在：{DB_PATH}")
    return sqlite3.connect(DB_PATH)


# =========================
# 4. SQLite 系统信息加载
# =========================
@st.cache_data(show_spinner="正在读取 SQLite 数据库信息...")
def load_sqlite_counts():
    """从 SQLite 数据库读取各表记录数量。"""
    default_counts = {
        "main_documents": 0,
        "term_dictionary": 0,
        "inverted_file": 0,
        "query_expansion": 0,
        "topic_taxonomy_zh": 0,
        "chinese_annotations": 0,
    }

    if not DB_PATH.exists():
        return default_counts

    conn = sqlite3.connect(DB_PATH)
    try:
        counts = {}
        for table in default_counts.keys():
            try:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table};")
                counts[table] = cursor.fetchone()[0]
            except Exception:
                counts[table] = 0
        return counts
    finally:
        conn.close()


# =========================
# 4.1 缓存加载 AI 语义检索资源
# =========================
@st.cache_resource(show_spinner="正在加载 AI 语义检索模型，请稍等...")
def load_semantic_model():
    return SentenceTransformer(SEMANTIC_MODEL_NAME)


@st.cache_data(show_spinner="正在加载语义向量索引，请稍等...")
def load_semantic_index():
    embeddings = np.load(SEMANTIC_EMBEDDINGS_PATH)
    metadata = pd.read_csv(SEMANTIC_METADATA_PATH, dtype=str).fillna("")
    return embeddings, metadata


# =========================
# 4.2 缓存加载系统测评结果
# =========================
@st.cache_data(show_spinner="正在加载系统性能测评结果...")
def load_evaluation_results():
    if not EVALUATION_RESULTS_CSV.exists():
        return pd.DataFrame()

    eval_df = pd.read_csv(EVALUATION_RESULTS_CSV, dtype=str).fillna("")

    if "elapsed_time_sec" in eval_df.columns:
        eval_df["elapsed_time_sec"] = pd.to_numeric(
            eval_df["elapsed_time_sec"],
            errors="coerce"
        ).fillna(0)

    if "result_count" in eval_df.columns:
        eval_df["result_count"] = pd.to_numeric(
            eval_df["result_count"],
            errors="coerce"
        ).fillna(0).astype(int)

    return eval_df


# =========================
# 4.3 DINOv2 图像符号检索资源加载与匹配
# =========================
def read_uploaded_image_bgr(image_bytes):
    """读取 Streamlit 上传图片，返回 OpenCV BGR 图像。"""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("无法读取上传图片，请确认图片格式为 PNG/JPG/JPEG。")

    return image_bgr


def cv_bgr_to_pil_rgb(img_bgr):
    """OpenCV BGR 图像转 PIL RGB。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def cv_gray_to_pil_rgb(img_gray):
    """灰度图转 PIL RGB。"""
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb)


def resize_keep_ratio_to_square(img, size=224, bg_color=255):
    """等比例缩放到正方形画布，避免拉伸符号。"""
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
    """通用主体裁剪，不针对任何单个符号。"""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

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
    return cropped if cropped.size > 0 else gray


def make_stone_like_from_mask(mask, size=224):
    """将查询 mask 转成浮雕风格查询图，用于与浮雕增强库风格对齐。"""
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

    return np.clip(base, 0, 255).astype(np.uint8)


def generate_query_views_from_bgr(image_bgr):
    """生成 DINOv2 查询视图。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    views = {}

    original_square = resize_keep_ratio_to_square(
        image_bgr,
        DINO_VIEW_SIZE,
        bg_color=255
    )
    views["original"] = cv_bgr_to_pil_rgb(original_square)

    cropped_gray = crop_symbol_region_from_gray(gray)
    cropped_square = resize_keep_ratio_to_square(
        cropped_gray,
        DINO_VIEW_SIZE,
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
        DINO_VIEW_SIZE,
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
        DINO_VIEW_SIZE,
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
        DINO_VIEW_SIZE,
        bg_color=0
    )
    views["binary_mask"] = cv_gray_to_pil_rgb(binary_square)

    relief_query = make_stone_like_from_mask(binary_square, DINO_VIEW_SIZE)
    views["query_relief_style"] = cv_bgr_to_pil_rgb(
        cv2.cvtColor(relief_query, cv2.COLOR_RGB2BGR)
    )

    return views


@st.cache_resource(show_spinner="正在加载 DINOv2 图像识别模型，请稍等...")
def load_dinov2_model():
    """加载 DINOv2 模型。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    model = AutoModel.from_pretrained(DINO_MODEL_NAME)
    model.to(device)
    model.eval()
    return model, processor, device


@st.cache_data(show_spinner="正在加载 DINOv2 浮雕增强符号向量库...")
def load_dinov2_relief_library():
    """加载 DINOv2 浮雕增强符号向量库。"""
    if not DINO_RELIEF_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到 DINOv2 元数据：{DINO_RELIEF_METADATA_CSV}")

    if not DINO_RELIEF_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"未找到 DINOv2 embedding：{DINO_RELIEF_EMBEDDINGS_PATH}")

    metadata = pd.read_csv(DINO_RELIEF_METADATA_CSV, dtype=str).fillna("")
    embeddings = np.load(DINO_RELIEF_EMBEDDINGS_PATH)
    return metadata, embeddings


def encode_images_dinov2(images, model, processor, device):
    """输入 PIL Image 列表，输出归一化后的 DINOv2 图像向量。"""
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


def get_dinov2_confidence(score):
    """根据 DINOv2 相似度给出置信度。"""
    score = float(score)
    if score >= 0.75:
        return "高"
    if score >= 0.55:
        return "中"
    return "低"


def search_uploaded_image_dinov2(image_bytes, top_k=10):
    """DINOv2 多视图图像符号检索入口。"""
    image_bgr = read_uploaded_image_bgr(image_bytes)
    metadata, relief_embeddings = load_dinov2_relief_library()
    model, processor, device = load_dinov2_model()

    views = generate_query_views_from_bgr(image_bgr)
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

    idx_best = scored_all.groupby("sign_id")["dinov2_score"].idxmax()
    best_per_sign = scored_all.loc[idx_best].copy()

    best_per_sign = best_per_sign.sort_values(
        by="dinov2_score",
        ascending=False
    ).reset_index(drop=True)

    best_per_sign["global_rank"] = best_per_sign.index + 1
    best_per_sign["confidence_level"] = best_per_sign["dinov2_score"].apply(
        get_dinov2_confidence
    )

    top_results = best_per_sign.head(top_k).copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return top_results, view_names, best_per_sign, image_rgb


# =========================
# 4.4 ResNet18 单符号识别资源加载
# =========================
@st.cache_resource(show_spinner="正在加载 ResNet18 单符号识别模型，请稍等...")
def load_resnet18_predictor():
    """加载 ResNet18 古埃及单符号识别器。"""
    return HieroglyphResNet18Predictor()


def predict_uploaded_image_resnet18(uploaded_file, top_k=5):
    """对 Streamlit 上传图片执行 ResNet18 Top-K 预测。"""
    image = Image.open(uploaded_file).convert("RGB")
    predictor = load_resnet18_predictor()
    results = predictor.predict_pil_image(image, top_k=top_k)
    return image, pd.DataFrame(results)


# =========================
# 5. 中文查询扩展：SQLite 版
# =========================
def expand_chinese_query_sqlite(conn, query: str):
    """从 SQLite 的 query_expansion 表中读取中文查询扩展词。"""
    query = query.strip()

    exact_hit = pd.read_sql_query(
        """
        SELECT *
        FROM query_expansion
        WHERE query_zh = ?
        LIMIT 1;
        """,
        conn,
        params=(query,)
    ).fillna("")

    if len(exact_hit) > 0:
        row = exact_hit.iloc[0]
    else:
        all_expansion = pd.read_sql_query(
            """
            SELECT *
            FROM query_expansion;
            """,
            conn
        ).fillna("")

        fuzzy_hit = all_expansion[
            all_expansion["query_zh"].apply(lambda x: x in query or query in x)
        ]

        if len(fuzzy_hit) == 0:
            return [], "未在中文查询扩展表中找到该主题。"

        row = fuzzy_hit.iloc[0]

    expanded_terms = [
        normalize_query_term(t)
        for t in str(row.get("expanded_terms", "")).split(",")
        if normalize_query_term(t)
    ]

    explanation = row.get("explanation_zh", "")
    return expanded_terms, explanation


# =========================
# 6. 单词 DIALOG 检索：SQLite 版
# =========================
def dialog_search_single_term_sqlite(conn, term):
    """对单个 term 执行 SQLite-backed DIALOG 风格检索，并加入字段加权排序。"""
    term_norm = normalize_query_term(term)

    if not term_norm:
        return pd.DataFrame(), None

    term_info = pd.read_sql_query(
        """
        SELECT term_id, term, df, total_tf, fields
        FROM term_dictionary
        WHERE term = ?
        LIMIT 1;
        """,
        conn,
        params=(term_norm,)
    ).fillna("")

    if len(term_info) == 0:
        return pd.DataFrame(), None

    term_info_row = term_info.iloc[0]
    term_id = term_info_row["term_id"]

    postings = pd.read_sql_query(
        """
        SELECT term_id, term, doc_id, field, tf, positions
        FROM inverted_file
        WHERE term_id = ?;
        """,
        conn,
        params=(term_id,)
    ).fillna("")

    if len(postings) == 0:
        return pd.DataFrame(), term_info_row

    postings["tf"] = pd.to_numeric(postings["tf"], errors="coerce").fillna(0).astype(int)

    field_weights = {
        "lemma_forms": 5,
        "normalized_transliteration": 4,
        "mdc": 3,
        "translation": 2
    }

    postings["field_weight"] = postings["field"].map(field_weights).fillna(1)
    postings["weighted_tf"] = postings["tf"] * postings["field_weight"]

    doc_scores = (
        postings
        .groupby("doc_id")
        .agg(
            total_tf=("tf", "sum"),
            weighted_score=("weighted_tf", "sum"),
            matched_fields=("field", lambda x: ", ".join(sorted(set(x)))),
            positions=("positions", lambda x: " | ".join(map(str, x)))
        )
        .reset_index()
    )

    doc_scores["matched_term"] = term_norm
    return doc_scores, term_info_row


def fetch_main_documents_sqlite(conn, doc_scores):
    """根据 doc_scores 回主文档表取完整文本信息，并连接中文知识增强表。"""
    if len(doc_scores) == 0:
        return pd.DataFrame()

    doc_ids = doc_scores["doc_id"].tolist()
    placeholders = ",".join(["?"] * len(doc_ids))

    main_docs = pd.read_sql_query(
        f"""
        SELECT
            m.*,
            c.topic_tags_zh,
            c.summary_zh,
            c.matched_chinese_rules,
            c.annotation_source
        FROM main_documents m
        LEFT JOIN chinese_annotations c
        ON m.doc_id = c.doc_id
        WHERE m.doc_id IN ({placeholders});
        """,
        conn,
        params=doc_ids
    ).fillna("")

    results = doc_scores.merge(main_docs, on="doc_id", how="left")
    return results


# =========================
# 7. 关键词检索：SQLite 版
# =========================
def keyword_search(query, top_k=10):
    """SQLite-backed 关键词检索，支持中文主题词、英文关键词、古埃及转写词检索。"""
    query = query.strip()
    conn = get_sqlite_connection()

    try:
        if contains_chinese(query):
            expanded_terms, explanation = expand_chinese_query_sqlite(conn, query)

            if not expanded_terms:
                return {
                    "query": query,
                    "mode": "关键词检索",
                    "sub_mode": "中文检索",
                    "expanded_terms": [],
                    "explanation": explanation,
                    "term_infos": [],
                    "results": pd.DataFrame()
                }

            all_doc_scores = []
            term_infos = []

            for term in expanded_terms:
                doc_scores, term_info = dialog_search_single_term_sqlite(conn, term)

                if term_info is not None:
                    term_infos.append(term_info)

                if len(doc_scores) > 0:
                    all_doc_scores.append(doc_scores)

            if not all_doc_scores:
                return {
                    "query": query,
                    "mode": "关键词检索",
                    "sub_mode": "中文检索",
                    "expanded_terms": expanded_terms,
                    "explanation": explanation,
                    "term_infos": term_infos,
                    "results": pd.DataFrame()
                }

            combined = pd.concat(all_doc_scores, ignore_index=True)

            combined_grouped = (
                combined
                .groupby("doc_id")
                .agg(
                    total_tf=("total_tf", "sum"),
                    weighted_score=("weighted_score", "sum"),
                    matched_terms=("matched_term", lambda x: ", ".join(sorted(set(x)))),
                    matched_fields=("matched_fields", lambda x: ", ".join(sorted(set(", ".join(x).split(", "))))),
                    positions=("positions", lambda x: " || ".join(map(str, x)))
                )
                .reset_index()
            )

            combined_grouped["matched_term_count"] = combined_grouped["matched_terms"].apply(
                lambda x: len(x.split(", ")) if isinstance(x, str) and x else 0
            )

            combined_grouped = combined_grouped.sort_values(
                by=["matched_term_count", "weighted_score", "total_tf"],
                ascending=[False, False, False]
            ).head(top_k)

            results = fetch_main_documents_sqlite(conn, combined_grouped)

            return {
                "query": query,
                "mode": "关键词检索",
                "sub_mode": "中文检索",
                "expanded_terms": expanded_terms,
                "explanation": explanation,
                "term_infos": term_infos,
                "results": results
            }

        term_norm = normalize_query_term(query)
        doc_scores, term_info = dialog_search_single_term_sqlite(conn, term_norm)

        if len(doc_scores) == 0:
            return {
                "query": query,
                "mode": "关键词检索",
                "sub_mode": "普通检索",
                "expanded_terms": [term_norm],
                "explanation": "",
                "term_infos": [term_info] if term_info is not None else [],
                "results": pd.DataFrame()
            }

        doc_scores = doc_scores.sort_values(
            by=["weighted_score", "total_tf", "doc_id"],
            ascending=[False, False, True]
        ).head(top_k)

        results = fetch_main_documents_sqlite(conn, doc_scores)

        return {
            "query": query,
            "mode": "关键词检索",
            "sub_mode": "普通检索",
            "expanded_terms": [term_norm],
            "explanation": "",
            "term_infos": [term_info] if term_info is not None else [],
            "results": results
        }

    finally:
        conn.close()


# =========================
# 8. AI 语义检索
# =========================
def semantic_search(query, top_k=10):
    """基于 sentence-transformers 的 AI 语义检索。"""
    query = query.strip()

    if not query:
        return {
            "query": query,
            "mode": "AI语义检索",
            "sub_mode": "语义向量检索",
            "expanded_terms": [],
            "explanation": "请输入有效查询。",
            "term_infos": [],
            "results": pd.DataFrame()
        }

    model = load_semantic_model()
    embeddings, metadata = load_semantic_index()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(embeddings, query_embedding[0])
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()
    results["semantic_score"] = scores[top_indices]

    try:
        conn = get_sqlite_connection()
        doc_ids = results["doc_id"].tolist()
        placeholders = ",".join(["?"] * len(doc_ids))

        chinese_info = pd.read_sql_query(
            f"""
            SELECT
                doc_id,
                topic_tags_zh,
                summary_zh,
                matched_chinese_rules,
                annotation_source
            FROM chinese_annotations
            WHERE doc_id IN ({placeholders});
            """,
            conn,
            params=doc_ids
        ).fillna("")
        conn.close()
        results = results.merge(chinese_info, on="doc_id", how="left").fillna("")

    except Exception:
        results["topic_tags_zh"] = ""
        results["summary_zh"] = ""
        results["matched_chinese_rules"] = ""
        results["annotation_source"] = ""

    return {
        "query": query,
        "mode": "AI语义检索",
        "sub_mode": "语义向量检索",
        "expanded_terms": [],
        "explanation": "系统基于语义向量计算用户查询与古埃及文本记录之间的相似度，返回语义最接近的原始语料证据。",
        "term_infos": [],
        "results": results
    }


# =========================
# 9. 页面主体
# =========================
st.title("𓂀 古埃及文字智能检索系统")
st.caption(
    "Version 2.1｜基于 DIALOG 风格的“主文档—索引文档—倒排档”结构，"
    "支持 SQLite 关键词检索、中文知识增强、AI 语义检索、DINOv2 图像相似检索、ResNet18 单符号识别与系统性能测评。"
)

system_counts = load_sqlite_counts()

with st.sidebar:
    st.header("系统信息")
    st.write("系统版本：", "Version 2.1")
    st.write("数据库状态：", "已连接" if DB_PATH.exists() else "未找到")
    st.write("主文档数量：", system_counts.get("main_documents", 0))
    st.write("索引词条数量：", system_counts.get("term_dictionary", 0))
    st.write("倒排记录数量：", system_counts.get("inverted_file", 0))
    st.write("中文查询词数量：", system_counts.get("query_expansion", 0))
    st.write("中文主题数量：", system_counts.get("topic_taxonomy_zh", 0))
    st.write("中文标注数量：", system_counts.get("chinese_annotations", 0))

    if SEMANTIC_EMBEDDINGS_PATH.exists() and SEMANTIC_METADATA_PATH.exists():
        st.write("语义索引状态：", "已加载")
        st.write("语义索引规模：", "8000 条")
    else:
        st.write("语义索引状态：", "未找到")

    if DINO_RELIEF_METADATA_CSV.exists() and DINO_RELIEF_EMBEDDINGS_PATH.exists():
        st.write("图像符号库：", "DINOv2 已加载")
        st.write("图像候选规模：", "6432 张浮雕增强图")
    else:
        st.write("图像符号库：", "DINOv2 特征未找到")

    if EVALUATION_RESULTS_CSV.exists():
        st.write("性能测评状态：", "已生成")
    else:
        st.write("性能测评状态：", "未生成")

    st.divider()
    st.header("示例查询")
    st.markdown("""
    **关键词检索：**
    - 神
    - 奥西里斯
    - 国王
    - 法老
    - 亡灵书
    - 审判
    - 祭祀
    - ntr
    - wsjr
    - osiris

    **AI语义检索：**
    - 太阳神和国王
    - Osiris and afterlife
    - offering rituals
    - texts about gods and kingship
    - enemies of Osiris

    **图像符号检索：**
    - 上传 N5.png 测试太阳符号
    - 上传 S34.png 测试生命符号
    """)


search_mode = st.radio(
    "请选择检索模式",
    ["关键词检索", "AI语义检索", "DINOv2图像相似检索", "ResNet18单符号识别"],
    horizontal=True
)

uploaded_sign_file = None
query = ""

if search_mode in ["DINOv2图像相似检索", "ResNet18单符号识别"]:
    uploaded_sign_file = st.file_uploader(
        "请上传单个古埃及象形文字符号图片",
        type=["png", "jpg", "jpeg"]
    )
    if search_mode == "DINOv2图像相似检索":
        st.caption("建议上传单个清晰符号图片。DINOv2 模块用于相似符号检索，暂不支持整行碑文 OCR。")
    else:
        st.caption("建议上传裁剪后的单个象形文字符号。ResNet18 模块用于 88 类 Gardiner 符号分类，暂不支持整张碑文自动分割。")
else:
    query = st.text_input(
        "请输入检索词或自然语言问题",
        placeholder="例如：神、奥西里斯、法老、亡灵书、ntr、太阳神和国王"
    )

if search_mode == "DINOv2图像相似检索":
    top_k = st.slider("返回相似符号数量", min_value=3, max_value=20, value=10)
elif search_mode == "ResNet18单符号识别":
    top_k = st.slider("返回候选符号数量", min_value=3, max_value=10, value=5)
else:
    top_k = st.slider("返回结果数量", min_value=3, max_value=20, value=10)

search_button = st.button("开始检索", type="primary")


if search_button:
    # =========================
    # ResNet18 单符号识别模式：监督分类 Top-K
    # =========================
    if search_mode == "ResNet18单符号识别":
        if uploaded_sign_file is None:
            st.warning("请先上传一张裁剪后的单个古埃及符号图片。")
        else:
            start_time = time.perf_counter()
            try:
                uploaded_image, pred_df = predict_uploaded_image_resnet18(
                    uploaded_file=uploaded_sign_file,
                    top_k=top_k
                )
                elapsed_time = time.perf_counter() - start_time

                st.subheader("ResNet18 单符号识别概览")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("识别模式", "ResNet18")
                c2.metric("任务类型", "88类分类")
                c3.metric("返回候选数", len(pred_df))
                c4.metric("识别耗时", f"{elapsed_time:.3f} 秒")

                st.markdown("**上传图片：**")
                st.image(uploaded_image, width=260)

                st.info(
                    "当前 ResNet18 模型适用于裁剪后的单个象形文字符号识别，"
                    "会返回 Top-K Gardiner 编号候选及概率。暂不支持整张碑文自动分割或整行 OCR。"
                )
                with st.expander("查看 ResNet18 模型性能与适用范围", expanded=False):
                    st.markdown(
                    """
                    **模型说明：**  
                    当前图像识别模块基于 ResNet18 迁移学习模型构建，用于对裁剪后的单个古埃及象形文字符号进行分类识别。

                    **训练数据：**
                    - 原始图像数据：9703 张
                    - 原始类别数量：310 类
                    - 第一版训练筛选：每类至少 20 张样本
                    - 最终训练数据：8498 张图像
                    - 最终分类类别：88 个 Gardiner 符号类别

                    **内部测试结果：**
                    - Test Top-1 Accuracy：93.31%
                    - Test Top-3 Accuracy：99.01%
                    - Test Top-5 Accuracy：99.47%

                    **真实外部测试结果：**
                    - 外部测试图像：29 张真实馆藏/浮雕裁剪图
                    - 覆盖类别：D21、D4、G17、G43、I9、M17、N35、S29、V30、X1
                    - External Top-1 Accuracy：96.55%
                    - External Top-3 Accuracy：100.00%
                    - External Top-5 Accuracy：100.00%

                    **适用范围：**
                    - 适用于已经裁剪好的单个象形文字符号；
                    - 适用于浮雕、碑刻、馆藏图像中的单符号识别；
                    - 适合返回 Top-K Gardiner 编号候选，辅助用户判断符号类别。

                    **当前限制：**
                    - 暂不支持整张碑文自动分割；
                    - 暂不支持整行 OCR；
                    - 对破损严重、多个符号粘连或裁剪不完整的图像，Top-1 结果可能不稳定。
                     """
                    )

                st.subheader("Top-K 识别结果")
                display_df = pred_df.copy()
                if "probability" in display_df.columns:
                    display_df["probability"] = display_df["probability"].round(6)
                st.dataframe(display_df, use_container_width=True)

                if len(pred_df) > 0:
                    top1 = pred_df.iloc[0]
                    st.success(
                        f"Top-1 识别结果：{top1['gardiner_code']}｜{top1['english_name']}，"
                        f"概率 {top1['probability_percent']}。"
                    )

            except Exception as e:
                st.error(f"ResNet18 单符号识别失败：{e}")

    # =========================
    # DINOv2 图像相似检索模式：多视图检索
    # =========================
    elif search_mode == "DINOv2图像相似检索":
        if uploaded_sign_file is None:
            st.warning("请先上传一张古埃及符号图片。")
        else:
            image_bytes = uploaded_sign_file.getvalue()
            start_time = time.perf_counter()

            try:
                image_results, view_names, full_results, image_rgb = search_uploaded_image_dinov2(
                    image_bytes=image_bytes,
                    top_k=top_k
                )
                elapsed_time = time.perf_counter() - start_time

                st.subheader("图像识别概览")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("检索模式", "DINOv2图像相似检索")
                c2.metric("子模式", "DINOv2 多视图")
                c3.metric("返回结果数", len(image_results))
                c4.metric("识别耗时", f"{elapsed_time:.3f} 秒")

                st.markdown("**上传图片：**")
                st.image(image_rgb, width=260)

                if view_names:
                    st.write("**查询视图：**", "、".join(view_names))

                st.info(
                    "系统基于 DINOv2 深度视觉特征，对上传图片生成原图、裁剪图、增强灰度图、"
                    "边缘图、二值图和浮雕风格图等多种查询视图，并在 6432 张浮雕增强符号图像库中"
                    "检索最相似的古埃及 Unicode 符号。"
                )

                st.subheader("相似符号 Top-K")

                if len(image_results) == 0:
                    st.warning("没有返回图像识别结果。")
                else:
                    for rank, (_, row) in enumerate(image_results.iterrows(), start=1):
                        gardiner_code = str(row.get("gardiner_code", "")).strip()
                        auto_label = str(row.get("auto_label", "")).strip()
                        display_id = str(row.get("display_id", "")).strip()
                        display_code = display_id or gardiner_code or auto_label or "未知编号"

                        unicode_char = str(row.get("unicode_char", "")).strip()
                        unicode_codepoint = str(row.get("unicode_codepoint", "")).strip()
                        zh_name = str(row.get("zh_name", "")).strip() or "暂无中文注释"
                        en_name = str(row.get("en_name", "")).strip() or "暂无英文注释"
                        related_terms = str(row.get("related_terms", "")).strip() or "暂无"
                        view_name = str(row.get("view_name", "")).strip()
                        relief_variant = str(row.get("relief_variant", "")).strip()
                        has_manual_annotation = str(row.get("has_manual_annotation", "")).strip()
                        score = float(row.get("dinov2_score", 0))
                        confidence = str(row.get("confidence_level", "")).strip() or get_dinov2_confidence(score)

                        with st.container(border=True):
                            st.markdown(f"### Top {rank}｜{display_code}｜{zh_name}")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.write("**Unicode**")
                            col1.write(unicode_char)
                            col2.write("**英文名**")
                            col2.write(en_name)
                            col3.write("**DINOv2 相似度**")
                            col3.write(round(score, 4))
                            col4.write("**置信度**")
                            col4.write(confidence)

                            st.write("**Codepoint：**", unicode_codepoint)
                            st.write("**相关检索词：**", related_terms)
                            st.write("**最佳查询视图：**", view_name)
                            st.write("**最佳浮雕风格：**", relief_variant)
                            st.write("**是否人工注释：**", has_manual_annotation)

                    top1 = image_results.iloc[0]
                    top1_terms = str(top1.get("related_terms", "")).strip()
                    top1_zh = str(top1.get("zh_name", "")).strip() or "暂无中文注释"
                    top1_code = (
                        str(top1.get("gardiner_code", "")).strip()
                        or str(top1.get("auto_label", "")).strip()
                        or "未知编号"
                    )
                    top1_score = round(float(top1.get("dinov2_score", 0)), 4)

                    st.success(
                        f"Top-1 识别结果为 {top1_code}｜{top1_zh}，DINOv2 相似度为 {top1_score}。"
                    )

                    if top1_terms:
                        st.write("**可联动文本检索词：**", top1_terms)
                        related_term_list = [t.strip() for t in top1_terms.split(",") if t.strip()]

                        if related_term_list:
                            link_term = related_term_list[0]
                            st.subheader("联动文本检索结果")
                            st.caption(
                                f"系统自动选择 Top-1 符号相关词 `{link_term}` 作为联动查询词，"
                                "调用 SQLite 关键词检索模块返回相关古埃及文本证据。"
                            )

                            text_output = keyword_search(link_term, top_k=5)
                            text_results = text_output["results"]

                            if len(text_results) == 0:
                                st.warning("未找到联动文本检索结果。")
                            else:
                                for text_rank, (_, text_row) in enumerate(text_results.iterrows(), start=1):
                                    with st.container(border=True):
                                        st.markdown(f"#### 文本结果 {text_rank}｜{text_row.get('doc_id', '')}")
                                        tc1, tc2, tc3, tc4 = st.columns(4)
                                        tc1.write("**corpus**")
                                        tc1.write(text_row.get("corpus", ""))
                                        tc2.write("**date**")
                                        tc2.write(text_row.get("date", ""))
                                        tc3.write("**findspot**")
                                        tc3.write(text_row.get("findspot", ""))
                                        tc4.write("**加权分数**")
                                        tc4.write(text_row.get("weighted_score", ""))

                                        topic_tags_zh = limit_topic_tags(text_row.get("topic_tags_zh", ""), max_tags=5)
                                        if topic_tags_zh:
                                            st.write("**中文主题标签：**", topic_tags_zh)

                                        if text_row.get("summary_zh", ""):
                                            st.info(text_row.get("summary_zh", ""))

                                        st.markdown("**原始译文：**")
                                        st.write(text_row.get("translation", ""))

                    with st.expander("查看完整图像检索结果表"):
                        st.dataframe(image_results, use_container_width=True)

                    with st.expander("查看全量候选结果"):
                        st.dataframe(full_results, use_container_width=True)

            except Exception as e:
                st.error(f"图像符号检索失败：{e}")

    # =========================
    # 文本检索模式
    # =========================
    else:
        if not query.strip():
            st.warning("请输入检索词或自然语言问题。")
        else:
            start_time = time.perf_counter()
            if search_mode == "关键词检索":
                output = keyword_search(query, top_k=top_k)
            else:
                output = semantic_search(query, top_k=top_k)

            elapsed_time = time.perf_counter() - start_time

            st.subheader("检索概览")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("检索模式", output["mode"])
            col2.metric("子模式", output.get("sub_mode", ""))
            col3.metric("返回结果数", len(output["results"]))
            col4.metric("检索耗时", f"{elapsed_time:.3f} 秒")

            st.write("**原始查询：**", output["query"])

            if output["expanded_terms"]:
                st.write("**扩展词：**", ", ".join(output["expanded_terms"]))

            if output["explanation"]:
                st.info(output["explanation"])

            if output["term_infos"]:
                with st.expander("查看命中的索引词信息"):
                    for info in output["term_infos"]:
                        if info is None:
                            continue
                        st.markdown(
                            f"""
                            - **term**: `{info['term']}`
                            - **term_id**: `{info['term_id']}`
                            - **df**: {info['df']}
                            - **total_tf**: {info['total_tf']}
                            - **fields**: {info['fields']}
                            """
                        )

            results = output["results"]
            if len(results) == 0:
                st.error("没有找到结果。")
            else:
                st.subheader("检索结果")
                for rank, (_, row) in enumerate(results.iterrows(), start=1):
                    with st.container(border=True):
                        st.markdown(f"### 结果 {rank}｜{row.get('doc_id', '')}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.write("**corpus**")
                        c1.write(row.get("corpus", ""))
                        c2.write("**date**")
                        c2.write(row.get("date", ""))
                        c3.write("**findspot**")
                        c3.write(row.get("findspot", ""))

                        if output["mode"] == "AI语义检索":
                            c4.write("**语义分数**")
                            c4.write(round(float(row.get("semantic_score", 0)), 4))
                        else:
                            c4.write("**加权分数**")
                            c4.write(row.get("weighted_score", "未计算"))

                        if output["mode"] == "AI语义检索":
                            st.write("**检索方式：** AI 语义相似度匹配")
                            st.write("**语义分数 semantic_score：**", round(float(row.get("semantic_score", 0)), 4))
                            st.info(
                                f"该结果与查询“{output['query']}”在语义上较为接近。"
                                "系统根据文本译文、古埃及转写和元数据生成语义向量，并按相似度返回相关古埃及文本证据。"
                            )
                        else:
                            matched_terms = row.get("matched_terms", row.get("matched_term", ""))
                            matched_fields = row.get("matched_fields", "")
                            st.write("**命中词：**", matched_terms)
                            st.write("**匹配字段：**", matched_fields)
                            st.write("**原始词频 total_tf：**", row.get("total_tf", ""))
                            chinese_hint = generate_chinese_hint(
                                query=output["query"],
                                matched_terms=matched_terms,
                                matched_fields=matched_fields
                            )
                            st.info(chinese_hint)

                        topic_tags_zh = limit_topic_tags(row.get("topic_tags_zh", ""), max_tags=5)
                        summary_zh = row.get("summary_zh", "")
                        matched_chinese_rules = row.get("matched_chinese_rules", "")

                        if topic_tags_zh or summary_zh:
                            st.markdown("**中文知识增强：**")
                            if topic_tags_zh:
                                st.write("**中文主题标签：**", topic_tags_zh)
                            if summary_zh:
                                st.info(summary_zh)
                            with st.expander("查看中文匹配规则"):
                                if matched_chinese_rules:
                                    st.write(matched_chinese_rules)
                                else:
                                    st.write("暂无匹配规则。")

                        st.markdown("**原始译文：**")
                        st.write(row.get("translation", ""))

                        st.markdown("**古埃及转写：**")
                        st.code(row.get("transliteration", ""), language="text")

                        with st.expander("查看 lemma / mdc / 归一化转写"):
                            st.markdown("**归一化转写：**")
                            st.code(row.get("normalized_transliteration", ""), language="text")
                            st.markdown("**lemma_forms：**")
                            st.code(row.get("lemma_forms", ""), language="text")
                            st.markdown("**mdc：**")
                            st.code(row.get("mdc", ""), language="text")


# =========================
# 10. 系统性能测评展示
# =========================
st.divider()

with st.expander("系统性能测评", expanded=False):
    st.markdown(
        """
        本模块用于展示系统批量性能测评结果，主要比较关键词检索与 AI 语义检索在响应时间、
        返回结果数量和 Top-1 文档等方面的表现。
        """
    )

    eval_df = load_evaluation_results()

    if len(eval_df) == 0:
        st.warning(
            "尚未找到性能测评结果文件。请先运行 "
            "`src/evaluate_search_performance.py` 生成 evaluation_results.csv。"
        )
    else:
        st.subheader("测评数据概览")
        c1, c2, c3 = st.columns(3)
        c1.metric("测试查询数量", len(eval_df))
        c2.metric("检索模式数量", eval_df["mode"].nunique())
        c3.metric("平均返回结果数", round(eval_df["result_count"].mean(), 2))

        st.subheader("不同检索模式平均耗时")
        avg_time_df = (
            eval_df
            .groupby("mode", as_index=False)["elapsed_time_sec"]
            .mean()
            .rename(columns={"elapsed_time_sec": "avg_elapsed_time_sec"})
        )
        avg_time_df["avg_elapsed_time_sec"] = avg_time_df["avg_elapsed_time_sec"].round(4)
        st.dataframe(avg_time_df, use_container_width=True)

        st.subheader("详细测评结果")
        display_cols = [
            "mode",
            "query",
            "top_k",
            "elapsed_time_sec",
            "result_count",
            "top_doc_id",
            "top_score",
            "top_corpus",
            "top_translation_preview"
        ]
        display_cols = [c for c in display_cols if c in eval_df.columns]
        st.dataframe(eval_df[display_cols], use_container_width=True)

        st.info(
            "说明：AI 语义检索的批量测评结果为模型预热后的热启动检索耗时；"
            "网页端首次语义检索可能包含模型加载和语义索引加载时间，因此首次耗时会更长。"
        )
