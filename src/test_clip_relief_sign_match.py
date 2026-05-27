from pathlib import Path
from io import BytesIO
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


# =========================================================
# 2. 支持中文路径读取图片
# =========================================================
def read_image_as_pil(image_path: Path):
    """
    支持 Windows 中文路径读取图片，并转换为 RGB。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = Image.open(BytesIO(img_bytes.tobytes())).convert("RGB")

    return image


# =========================================================
# 3. 加载 CLIP 模型
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


# =========================================================
# 4. 提取单张图片 CLIP embedding
# =========================================================
def encode_single_image_clip(image: Image.Image, model, processor, device):
    """
    输入单张 PIL Image，输出归一化后的 CLIP 图像向量。
    兼容不同 transformers 版本返回格式。
    """
    inputs = processor(
        images=[image],
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
# 5. 加载浮雕增强 CLIP 向量库
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
# 6. 置信度
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
# 7. CLIP 浮雕增强库匹配 + sign_id 聚合
# =========================================================
def match_clip_relief_sign_image(image_path: Path, top_k: int = 10):
    """
    输入真实浮雕图片：
    1. 提取 CLIP embedding
    2. 与 6432 个浮雕增强图 embedding 匹配
    3. 按 sign_id 聚合，每个符号保留最高分
    4. 返回 Top-K 符号
    """
    metadata, relief_embeddings = load_clip_relief_library()

    model, processor, device = load_clip_model()

    image = read_image_as_pil(image_path)
    input_embedding = encode_single_image_clip(
        image,
        model,
        processor,
        device
    )

    similarities = cosine_similarity(
        input_embedding,
        relief_embeddings
    )[0]

    scored = metadata.copy()
    scored["clip_relief_score"] = similarities

    # 同一个符号有 6 个浮雕增强版本，只保留最高分那个版本
    idx_best = scored.groupby("sign_id")["clip_relief_score"].idxmax()
    best_per_sign = scored.loc[idx_best].copy()

    best_per_sign = best_per_sign.sort_values(
        by="clip_relief_score",
        ascending=False
    ).head(top_k)

    best_per_sign["confidence_level"] = best_per_sign["clip_relief_score"].apply(
        get_confidence_level
    )

    return best_per_sign


# =========================================================
# 8. 输出结果
# =========================================================
def print_match_results(results):
    print("\nCLIP 浮雕增强匹配结果 Top-K：")
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
        print("最佳浮雕风格:", relief_variant)
        print("浮雕增强图路径:", relief_png_path)
        print("CLIP 浮雕相似度:", round(float(row.get("clip_relief_score", 0)), 4))
        print("置信度:", row.get("confidence_level", ""))
        print("-" * 120)


# =========================================================
# 9. 主程序
# =========================================================
def main():
    print("=" * 120)
    print("V2.10 CLIP 浮雕风格增强古埃及符号图像匹配测试")
    print("=" * 120)
    print("候选库：1072 个符号 × 6 种浮雕风格 = 6432 个增强样本")
    print("策略：CLIP 图像 embedding + 浮雕增强样本库 + sign_id 聚合")
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
            results = match_clip_relief_sign_image(
                image_path=image_path,
                top_k=10
            )

            print_match_results(results)

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()