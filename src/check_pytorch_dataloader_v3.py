# -*- coding: utf-8 -*-
"""
check_pytorch_dataloader_v3.py

功能：
1. 读取 split_inventory_min20.csv；
2. 构建 PyTorch Dataset；
3. 构建 train / val / test DataLoader；
4. 检查图片能否正常读取；
5. 输出 batch 的 shape、label 信息；
6. 保存一张 batch 预览图；
7. 为后续 ResNet18 / MobileNetV3 训练做准备。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\check_pytorch_dataloader_v3.py
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image, ImageOps, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPLIT_CSV = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "dataset_splits"
    / "split_inventory_min20.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "dataloader_check"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_PREVIEW_PATH = OUTPUT_DIR / "train_batch_preview.png"
DATALOADER_SUMMARY_PATH = OUTPUT_DIR / "dataloader_check_summary.txt"


# =========================================================
# 2. 参数设置
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # Windows 下建议先用 0，最稳
RANDOM_SEED = 42


# =========================================================
# 3. 图像变换
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=8),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 4. Dataset 定义
# =========================================================

class HieroglyphDataset(Dataset):
    """
    古埃及象形文字符号图像数据集。
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        self.required_cols = [
            "image_path",
            "gardiner_code",
            "label_id",
            "split"
        ]

        for col in self.required_cols:
            if col not in self.df.columns:
                raise ValueError(f"数据表缺少必要字段：{col}")

        self.df["label_id"] = self.df["label_id"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image_path = row["image_path"]
        label = int(row["label_id"])
        gardiner_code = row["gardiner_code"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"图片读取失败：{image_path}，错误：{e}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, gardiner_code, image_path


# =========================================================
# 5. 数据读取与检查
# =========================================================

def load_split_dataframe() -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"未找到划分文件：{SPLIT_CSV}\n"
            "请先运行 split_hieroglyph_dataset_v3.py"
        )

    df = pd.read_csv(SPLIT_CSV, dtype=str).fillna("")

    required_cols = [
        "image_id",
        "class_name",
        "file_name",
        "image_path",
        "relative_path",
        "suffix",
        "gardiner_code",
        "label_id",
        "split"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"split_inventory_min20.csv 缺少字段：{col}")

    df["label_id"] = df["label_id"].astype(int)

    return df


def check_image_paths(df: pd.DataFrame) -> Tuple[int, int]:
    """
    检查图片路径是否存在。
    """
    total = len(df)
    exists_count = 0

    for path in df["image_path"]:
        if Path(path).exists():
            exists_count += 1

    missing_count = total - exists_count

    return exists_count, missing_count


def build_dataloaders(df: pd.DataFrame):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_dataset = HieroglyphDataset(train_df, transform=train_transform)
    val_dataset = HieroglyphDataset(val_df, transform=eval_transform)
    test_dataset = HieroglyphDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# =========================================================
# 6. 反归一化与预览图保存
# =========================================================

def denormalize_tensor(image_tensor: torch.Tensor) -> Image.Image:
    """
    将经过 Normalize 的 Tensor 转回 PIL Image，方便保存预览图。
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image_tensor = image_tensor.cpu() * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)

    image_tensor = (image_tensor * 255).byte()
    image_np = image_tensor.permute(1, 2, 0).numpy()

    return Image.fromarray(image_np)


def save_batch_preview(images, labels, codes, output_path: Path):
    """
    保存一个 batch 的预览图。
    """
    n = min(len(images), BATCH_SIZE)

    thumb_size = 128
    label_h = 28
    cols = 4
    rows = (n + cols - 1) // cols

    canvas_w = cols * thumb_size
    canvas_h = rows * (thumb_size + label_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for i in range(n):
        img = denormalize_tensor(images[i])
        img = ImageOps.contain(img, (thumb_size, thumb_size))

        x = (i % cols) * thumb_size
        y = (i // cols) * (thumb_size + label_h)

        cell = Image.new("RGB", (thumb_size, thumb_size), "white")
        paste_x = (thumb_size - img.width) // 2
        paste_y = (thumb_size - img.height) // 2
        cell.paste(img, (paste_x, paste_y))

        canvas.paste(cell, (x, y))

        text = f"{codes[i]} | {int(labels[i])}"
        draw.text((x + 4, y + thumb_size + 4), text, fill="black")

    canvas.save(output_path)


# =========================================================
# 7. 主流程
# =========================================================

def main():
    torch.manual_seed(RANDOM_SEED)

    print("=" * 90)
    print("开始检查 PyTorch Dataset / DataLoader")
    print("=" * 90)

    df = load_split_dataframe()

    print("\n数据集基本信息：")
    print(f"总图片数：{len(df)}")
    print(f"类别数量：{df['gardiner_code'].nunique()}")

    print("\nSplit 分布：")
    split_summary = (
        df.groupby("split")
        .agg(
            image_count=("image_path", "count"),
            class_count=("gardiner_code", "nunique")
        )
        .reset_index()
    )
    print(split_summary.to_string(index=False))

    print("\n正在检查图片路径是否存在...")
    exists_count, missing_count = check_image_paths(df)
    print(f"存在图片数：{exists_count}")
    print(f"缺失图片数：{missing_count}")

    if missing_count > 0:
        print("\n警告：存在缺失图片路径。请检查 split_inventory_min20.csv。")
        return

    print("\n正在构建 Dataset 和 DataLoader...")
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = build_dataloaders(df)

    print("\nDataset 大小：")
    print(f"train_dataset：{len(train_dataset)}")
    print(f"val_dataset：{len(val_dataset)}")
    print(f"test_dataset：{len(test_dataset)}")

    print("\n正在读取一个 train batch...")
    images, labels, codes, paths = next(iter(train_loader))

    print("\nBatch 信息：")
    print(f"images shape：{images.shape}")
    print(f"labels shape：{labels.shape}")
    print(f"前 10 个 labels：{labels[:10].tolist()}")
    print(f"前 10 个 codes：{list(codes[:10])}")

    print("\n正在保存 batch 预览图...")
    save_batch_preview(images, labels, codes, BATCH_PREVIEW_PATH)

    with open(DATALOADER_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("PyTorch DataLoader 检查报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总图片数：{len(df)}\n")
        f.write(f"类别数量：{df['gardiner_code'].nunique()}\n\n")
        f.write("Split 分布：\n")
        f.write(split_summary.to_string(index=False))
        f.write("\n\n")
        f.write(f"存在图片数：{exists_count}\n")
        f.write(f"缺失图片数：{missing_count}\n\n")
        f.write(f"train_dataset：{len(train_dataset)}\n")
        f.write(f"val_dataset：{len(val_dataset)}\n")
        f.write(f"test_dataset：{len(test_dataset)}\n\n")
        f.write(f"batch image shape：{tuple(images.shape)}\n")
        f.write(f"batch label shape：{tuple(labels.shape)}\n")
        f.write(f"batch preview：{BATCH_PREVIEW_PATH}\n")

    print("\n输出文件：")
    print("1.", BATCH_PREVIEW_PATH)
    print("2.", DATALOADER_SUMMARY_PATH)

    print("\nDataLoader 检查完成。下一步可以开始训练 ResNet18 baseline。")


if __name__ == "__main__":
    main()