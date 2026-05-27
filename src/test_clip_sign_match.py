from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

CLIP_FEATURE_DIR = SIGN_DIR / "sign_clip_features"
CLIP_METADATA_CSV = CLIP_FEATURE_DIR / "clip_sign_metadata.csv"
CLIP_EMBEDDINGS_PATH = CLIP_FEATURE_DIR / "clip_sign_embeddings.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"


# =========================
# 2. 图片读取
# =========================
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


# =========================
# 3. 加载 CLIP 模型
# =========================
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


# =========================
# 4. 提取单张图片 embedding
# =========================
def encode_single_image_clip(image: Image.Image, model, processor, device):
    """
    输入单张 PIL Image，输出归一化后的 CLIP 图像向量。
    兼容不同 transformers 版本的返回格式。
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

    # 兼容不同 transformers 版本
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


# =========================
# 5. 加载符号 embedding 库
# =========================
def load_clip_sign_library():
    if not CLIP_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到 CLIP 元数据：{CLIP_METADATA_CSV}")

    if not CLIP_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"未找到 CLIP embedding：{CLIP_EMBEDDINGS_PATH}")

    metadata = pd.read_csv(CLIP_METADATA_CSV, dtype=str).fillna("")
    embeddings = np.load(CLIP_EMBEDDINGS_PATH)

    return metadata, embeddings


# =========================
# 6. 置信度
# =========================
def get_confidence_level(score):
    score = float(score)

    if score >= 0.35:
        return "高"
    elif score >= 0.25:
        return "中"
    else:
        return "低"


# =========================
# 7. CLIP 图像匹配
# =========================
def match_clip_sign_image(image_path: Path, top_k: int = 10):
    metadata, sign_embeddings = load_clip_sign_library()

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
        sign_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()
    results["clip_score"] = similarities[top_indices]
    results["confidence_level"] = results["clip_score"].apply(get_confidence_level)

    return results


# =========================
# 8. 输出结果
# =========================
def print_match_results(results):
    print("\nCLIP 匹配结果 Top-K：")
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
        print("CLIP 相似度:", round(float(row.get("clip_score", 0)), 4))
        print("置信度:", row.get("confidence_level", ""))
        print("-" * 120)


# =========================
# 9. 主程序
# =========================
def main():
    print("=" * 120)
    print("V2.8 CLIP 深度视觉特征古埃及符号图像匹配测试")
    print("=" * 120)
    print("候选库：1072 个 Unicode 古埃及符号")
    print("策略：CLIP 图像 embedding + 余弦相似度")
    print("不包含任何单独符号加分，不包含人工注释加权。")
    print("\n示例：")
    print(r"C:\Users\GE ZITONG\Desktop\安卡1.png")
    print(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project\data_sign_demo\sign_png_full\U_13000.png")

    while True:
        user_input = input("\n请输入图片路径 / q：").strip().strip('"')

        if user_input.lower() == "q":
            print("已退出。")
            break

        image_path = Path(user_input)

        try:
            results = match_clip_sign_image(
                image_path=image_path,
                top_k=10
            )

            print_match_results(results)

        except Exception as e:
            print("匹配失败：", e)


if __name__ == "__main__":
    main()