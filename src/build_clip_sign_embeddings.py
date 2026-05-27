from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

# 使用全量合并符号库
SIGN_LIBRARY_CSV = SIGN_DIR / "sign_library_merged_processed.csv"

CLIP_FEATURE_DIR = SIGN_DIR / "sign_clip_features"
CLIP_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

CLIP_METADATA_OUT = CLIP_FEATURE_DIR / "clip_sign_metadata.csv"
CLIP_EMBEDDINGS_OUT = CLIP_FEATURE_DIR / "clip_sign_embeddings.npy"

# CLIP 模型
MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 32


# =========================
# 2. 图片读取
# =========================
def read_image_as_pil(image_path: Path):
    """
    支持 Windows 中文路径读取图片，并转换为 RGB PIL Image。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return None

    try:
        img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        image = Image.open(BytesIO(img_bytes.tobytes())).convert("RGB")
        return image
    except Exception:
        return None


# =========================
# 3. 加载 CLIP 模型
# =========================
def load_clip_model():
    """
    加载 CLIP 图像编码模型。
    第一次运行时会自动下载模型。
    """
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
# 4. 批量提取图像 embedding
# =========================
def encode_images_clip(images, model, processor, device):
    """
    输入 PIL Image 列表，输出归一化后的 CLIP 图像向量。
    兼容不同 transformers 版本的返回格式。
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

    # L2 归一化
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-12)

    return image_features.cpu().numpy()


# =========================
# 5. 主流程
# =========================
def main():
    if not SIGN_LIBRARY_CSV.exists():
        raise FileNotFoundError(f"未找到符号库文件：{SIGN_LIBRARY_CSV}")

    df = pd.read_csv(SIGN_LIBRARY_CSV, dtype=str).fillna("")

    print("=" * 80)
    print("开始构建 CLIP 古埃及符号图像向量库")
    print("=" * 80)
    print("输入符号库：", SIGN_LIBRARY_CSV)
    print("输入记录数：", len(df))
    print("输出目录：", CLIP_FEATURE_DIR)

    model, processor, device = load_clip_model()

    valid_rows = []
    all_embeddings = []

    batch_images = []
    batch_rows = []

    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        # 优先使用原始标准 PNG，因为 CLIP 更适合自然图像/完整图像，而不是纯二值 processed 图
        png_path = row.get("png_path", "")

        if not png_path:
            fail_count += 1
            continue

        image = read_image_as_pil(Path(png_path))

        if image is None:
            fail_count += 1
            continue

        batch_images.append(image)
        batch_rows.append(row)

        if len(batch_images) >= BATCH_SIZE:
            embeddings = encode_images_clip(
                batch_images,
                model,
                processor,
                device
            )

            all_embeddings.append(embeddings)
            valid_rows.extend(batch_rows)

            success_count += len(batch_images)

            print(f"已处理 {idx + 1} / {len(df)} 条，成功 {success_count} 条")

            batch_images = []
            batch_rows = []

    # 处理最后一个 batch
    if batch_images:
        embeddings = encode_images_clip(
            batch_images,
            model,
            processor,
            device
        )

        all_embeddings.append(embeddings)
        valid_rows.extend(batch_rows)

        success_count += len(batch_images)

    if not all_embeddings:
        raise ValueError("没有成功生成任何 CLIP embedding，请检查图片路径。")

    metadata = pd.DataFrame(valid_rows).reset_index(drop=True)
    embedding_matrix = np.vstack(all_embeddings)

    metadata.to_csv(CLIP_METADATA_OUT, index=False, encoding="utf-8-sig")
    np.save(CLIP_EMBEDDINGS_OUT, embedding_matrix)

    print("\n" + "=" * 80)
    print("CLIP 符号图像向量库构建完成！")
    print("=" * 80)
    print("成功处理数量：", success_count)
    print("失败数量：", fail_count)
    print("embedding_matrix shape:", embedding_matrix.shape)

    print("\n输出文件：")
    print("1.", CLIP_METADATA_OUT)
    print("2.", CLIP_EMBEDDINGS_OUT)

    print("\n人工注释数量统计：")
    if "has_manual_annotation" in metadata.columns:
        print(metadata["has_manual_annotation"].value_counts())
    else:
        print("未找到 has_manual_annotation 字段。")


if __name__ == "__main__":
    main()