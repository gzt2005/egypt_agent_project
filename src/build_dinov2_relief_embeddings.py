from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

# 输入：之前已经生成好的浮雕增强图元数据
RELIEF_METADATA_CSV = SIGN_DIR / "sign_relief_augmented_metadata.csv"

# 输出：DINOv2 浮雕增强图特征库
DINO_RELIEF_DIR = SIGN_DIR / "sign_dinov2_relief_features"
DINO_RELIEF_DIR.mkdir(parents=True, exist_ok=True)

DINO_RELIEF_METADATA_OUT = DINO_RELIEF_DIR / "dinov2_relief_metadata.csv"
DINO_RELIEF_EMBEDDINGS_OUT = DINO_RELIEF_DIR / "dinov2_relief_embeddings.npy"

# DINOv2 模型
# small 模型更轻，适合本地电脑先跑通
MODEL_NAME = "facebook/dinov2-small"

# 如果内存不够，把 16 改成 8 或 4
BATCH_SIZE = 16


# =========================================================
# 2. 支持中文路径读取图片
# =========================================================
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


# =========================================================
# 3. 加载 DINOv2 模型
# =========================================================
def load_dinov2_model():
    """
    加载 DINOv2 图像编码模型。
    第一次运行会自动下载模型。
    """
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


# =========================================================
# 4. 批量提取 DINOv2 图像向量
# =========================================================
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

    # DINOv2 的全局图像表示通常取 CLS token
    if hasattr(outputs, "last_hidden_state"):
        image_features = outputs.last_hidden_state[:, 0, :]
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        image_features = outputs.pooler_output
    else:
        raise TypeError(f"无法识别 DINOv2 输出类型：{type(outputs)}")

    # L2 归一化，便于余弦相似度计算
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-12)

    return image_features.cpu().numpy()


# =========================================================
# 5. 主流程
# =========================================================
def main():
    if not RELIEF_METADATA_CSV.exists():
        raise FileNotFoundError(f"未找到浮雕增强元数据文件：{RELIEF_METADATA_CSV}")

    df = pd.read_csv(RELIEF_METADATA_CSV, dtype=str).fillna("")

    print("=" * 90)
    print("开始构建 DINOv2 浮雕风格增强图向量库")
    print("=" * 90)
    print("输入元数据：", RELIEF_METADATA_CSV)
    print("输入图片数量：", len(df))
    print("输出目录：", DINO_RELIEF_DIR)

    model, processor, device = load_dinov2_model()

    valid_rows = []
    all_embeddings = []

    batch_images = []
    batch_rows = []

    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        relief_png_path = row.get("relief_png_path", "")

        if not relief_png_path:
            fail_count += 1
            continue

        image = read_image_as_pil(Path(relief_png_path))

        if image is None:
            fail_count += 1
            continue

        batch_images.append(image)
        batch_rows.append(row)

        if len(batch_images) >= BATCH_SIZE:
            embeddings = encode_images_dinov2(
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
        embeddings = encode_images_dinov2(
            batch_images,
            model,
            processor,
            device
        )

        all_embeddings.append(embeddings)
        valid_rows.extend(batch_rows)

        success_count += len(batch_images)

    if not all_embeddings:
        raise ValueError("没有成功生成任何 DINOv2 embedding，请检查 relief_png_path。")

    metadata = pd.DataFrame(valid_rows).reset_index(drop=True)
    embedding_matrix = np.vstack(all_embeddings)

    metadata.to_csv(DINO_RELIEF_METADATA_OUT, index=False, encoding="utf-8-sig")
    np.save(DINO_RELIEF_EMBEDDINGS_OUT, embedding_matrix)

    print("\n" + "=" * 90)
    print("DINOv2 浮雕增强图向量库构建完成！")
    print("=" * 90)
    print("成功处理数量：", success_count)
    print("失败数量：", fail_count)
    print("embedding_matrix shape:", embedding_matrix.shape)

    print("\n输出文件：")
    print("1.", DINO_RELIEF_METADATA_OUT)
    print("2.", DINO_RELIEF_EMBEDDINGS_OUT)

    print("\n浮雕风格数量统计：")
    if "relief_variant" in metadata.columns:
        print(metadata["relief_variant"].value_counts())

    print("\n人工注释数量统计：")
    if "has_manual_annotation" in metadata.columns:
        print(metadata["has_manual_annotation"].value_counts())


if __name__ == "__main__":
    main()