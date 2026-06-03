# -*- coding: utf-8 -*-
"""
train_resnet18_baseline_v3.py

功能：
1. 读取 split_inventory_min20.csv；
2. 构建 PyTorch Dataset / DataLoader；
3. 使用 ResNet18 进行 88 类古埃及象形文字符号分类；
4. 输出 train / val loss 和 accuracy；
5. 保存最佳模型；
6. 在 test 集上计算 Top-1、Top-3、Top-5 Accuracy；
7. 输出训练日志和测试结果。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\train_resnet18_baseline_v3.py
"""

from pathlib import Path
from typing import Tuple, Dict, List
import time
import json

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


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
    / "resnet18_baseline"
)

MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
RESULT_DIR = OUTPUT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_resnet18_min20.pth"
LAST_MODEL_PATH = MODEL_DIR / "last_resnet18_min20.pth"

TRAIN_LOG_CSV = LOG_DIR / "train_log.csv"
TEST_RESULT_CSV = RESULT_DIR / "test_predictions.csv"
TEST_SUMMARY_TXT = RESULT_DIR / "test_summary.txt"
CLASS_MAPPING_JSON = RESULT_DIR / "class_mapping.json"


# =========================================================
# 2. 训练参数
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows 下先用 0，最稳
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 3. 图像增强与预处理
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=8),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
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
# 4. Dataset
# =========================================================

class HieroglyphDataset(Dataset):
    """
    古埃及象形文字符号图像分类数据集。
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        required_cols = [
            "image_path",
            "gardiner_code",
            "label_id",
            "split"
        ]

        for col in required_cols:
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
# 5. 数据加载
# =========================================================

def load_split_dataframe() -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"未找到划分文件：{SPLIT_CSV}\n"
            "请先运行 split_hieroglyph_dataset_v3.py"
        )

    df = pd.read_csv(SPLIT_CSV, dtype=str).fillna("")
    df["label_id"] = df["label_id"].astype(int)

    return df


def build_class_mapping(df: pd.DataFrame) -> Dict[int, str]:
    """
    建立 label_id -> Gardiner code 映射。
    """
    mapping_df = (
        df[["label_id", "gardiner_code"]]
        .drop_duplicates()
        .sort_values("label_id")
    )

    id_to_code = {
        int(row["label_id"]): row["gardiner_code"]
        for _, row in mapping_df.iterrows()
    }

    with open(CLASS_MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(id_to_code, f, ensure_ascii=False, indent=2)

    return id_to_code


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

    return train_loader, val_loader, test_loader


# =========================================================
# 6. 模型构建
# =========================================================

def build_resnet18_model(num_classes: int) -> nn.Module:
    """
    构建 ResNet18 迁移学习模型。
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# =========================================================
# 7. 指标计算
# =========================================================

def calculate_topk_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    topk: Tuple[int, ...] = (1, 3, 5)
) -> Dict[int, float]:
    """
    计算 Top-K Accuracy。
    """
    max_k = max(topk)

    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()

    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    results = {}

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[k] = correct_k.item() / labels.size(0)

    return results


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    optimizer=None,
    mode: str = "train"
):
    """
    训练或验证一个 epoch。
    """
    if mode == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    total_samples = 0

    top1_sum = 0.0
    top3_sum = 0.0
    top5_sum = 0.0

    for images, labels, codes, paths in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        batch_size = labels.size(0)

        if mode == "train":
            optimizer.zero_grad()

        with torch.set_grad_enabled(mode == "train"):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if mode == "train":
                loss.backward()
                optimizer.step()

        accs = calculate_topk_accuracy(outputs, labels, topk=(1, 3, 5))

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        top1_sum += accs[1] * batch_size
        top3_sum += accs[3] * batch_size
        top5_sum += accs[5] * batch_size

    epoch_loss = running_loss / total_samples
    top1_acc = top1_sum / total_samples
    top3_acc = top3_sum / total_samples
    top5_acc = top5_sum / total_samples

    return epoch_loss, top1_acc, top3_acc, top5_acc


# =========================================================
# 8. 测试集预测
# =========================================================

def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    id_to_code: Dict[int, str]
):
    """
    在测试集上评估并保存每张图片的预测结果。
    """
    model.eval()

    records = []

    total_samples = 0
    top1_sum = 0.0
    top3_sum = 0.0
    top5_sum = 0.0

    with torch.no_grad():
        for images, labels, codes, paths in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            accs = calculate_topk_accuracy(outputs, labels, topk=(1, 3, 5))

            batch_size = labels.size(0)
            total_samples += batch_size
            top1_sum += accs[1] * batch_size
            top3_sum += accs[3] * batch_size
            top5_sum += accs[5] * batch_size

            top5_probs, top5_indices = probs.topk(5, dim=1, largest=True, sorted=True)

            for i in range(batch_size):
                true_label = int(labels[i].cpu().item())
                true_code = codes[i]

                pred_ids = [int(x) for x in top5_indices[i].cpu().tolist()]
                pred_probs = [float(x) for x in top5_probs[i].cpu().tolist()]
                pred_codes = [id_to_code[pred_id] for pred_id in pred_ids]

                records.append({
                    "image_path": paths[i],
                    "true_label_id": true_label,
                    "true_gardiner_code": true_code,
                    "pred_top1_id": pred_ids[0],
                    "pred_top1_code": pred_codes[0],
                    "pred_top1_prob": pred_probs[0],
                    "top1_hit": pred_codes[0] == true_code,
                    "top3_hit": true_code in pred_codes[:3],
                    "top5_hit": true_code in pred_codes[:5],
                    "pred_top5_codes": "|".join(pred_codes),
                    "pred_top5_probs": "|".join([f"{p:.6f}" for p in pred_probs])
                })

    test_top1 = top1_sum / total_samples
    test_top3 = top3_sum / total_samples
    test_top5 = top5_sum / total_samples

    result_df = pd.DataFrame(records)
    result_df.to_csv(TEST_RESULT_CSV, index=False, encoding="utf-8-sig")

    with open(TEST_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("ResNet18 Baseline 测试结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试样本数：{total_samples}\n")
        f.write(f"Top-1 Accuracy：{test_top1:.4f}\n")
        f.write(f"Top-3 Accuracy：{test_top3:.4f}\n")
        f.write(f"Top-5 Accuracy：{test_top5:.4f}\n\n")
        f.write(f"预测明细：{TEST_RESULT_CSV}\n")

    return test_top1, test_top3, test_top5


# =========================================================
# 9. 主训练流程
# =========================================================

def main():
    torch.manual_seed(RANDOM_SEED)

    print("=" * 90)
    print("开始训练 ResNet18 古埃及文字图像识别 Baseline")
    print("=" * 90)
    print(f"运行设备：{DEVICE}")
    print(f"Epochs：{NUM_EPOCHS}")
    print(f"Batch size：{BATCH_SIZE}")
    print(f"Learning rate：{LEARNING_RATE}")

    df = load_split_dataframe()
    id_to_code = build_class_mapping(df)

    num_classes = df["label_id"].nunique()

    print("\n数据集信息：")
    print(f"总图片数：{len(df)}")
    print(f"类别数量：{num_classes}")

    split_summary = (
        df.groupby("split")
        .agg(
            image_count=("image_path", "count"),
            class_count=("gardiner_code", "nunique")
        )
        .reset_index()
    )

    print("\nSplit 分布：")
    print(split_summary.to_string(index=False))

    train_loader, val_loader, test_loader = build_dataloaders(df)

    model = build_resnet18_model(num_classes=num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_top1 = 0.0
    train_logs: List[Dict] = []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_top1, train_top3, train_top5 = run_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            mode="train"
        )

        val_loss, val_top1, val_top3, val_top5 = run_one_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            optimizer=None,
            mode="val"
        )

        epoch_time = time.time() - epoch_start

        log_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "train_top3": train_top3,
            "train_top5": train_top5,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top3": val_top3,
            "val_top5": val_top5,
            "epoch_time_sec": epoch_time
        }

        train_logs.append(log_record)

        print(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}] "
            f"time={epoch_time:.1f}s\n"
            f"  Train: loss={train_loss:.4f}, "
            f"Top1={train_top1:.4f}, Top3={train_top3:.4f}, Top5={train_top5:.4f}\n"
            f"  Val:   loss={val_loss:.4f}, "
            f"Top1={val_top1:.4f}, Top3={val_top3:.4f}, Top5={val_top5:.4f}"
        )

        # 保存最佳模型
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1

            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "id_to_code": id_to_code,
                "epoch": epoch,
                "val_top1": val_top1,
                "val_top3": val_top3,
                "val_top5": val_top5
            }, BEST_MODEL_PATH)

            print(f"  已保存最佳模型：{BEST_MODEL_PATH}")

        # 每轮保存日志
        pd.DataFrame(train_logs).to_csv(
            TRAIN_LOG_CSV,
            index=False,
            encoding="utf-8-sig"
        )

    # 保存最后一轮模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "id_to_code": id_to_code,
        "epoch": NUM_EPOCHS
    }, LAST_MODEL_PATH)

    total_time = time.time() - start_time

    print("\n" + "=" * 90)
    print("训练完成")
    print("=" * 90)
    print(f"总耗时：{total_time / 60:.2f} 分钟")
    print(f"最佳验证 Top1：{best_val_top1:.4f}")
    print(f"训练日志：{TRAIN_LOG_CSV}")
    print(f"最佳模型：{BEST_MODEL_PATH}")
    print(f"最后模型：{LAST_MODEL_PATH}")

    # 加载最佳模型进行测试
    print("\n正在加载最佳模型并评估 test 集...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_top1, test_top3, test_top5 = evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        id_to_code=id_to_code
    )

    print("\n测试集结果：")
    print(f"Top-1 Accuracy：{test_top1:.4f}")
    print(f"Top-3 Accuracy：{test_top3:.4f}")
    print(f"Top-5 Accuracy：{test_top5:.4f}")

    print("\n输出文件：")
    print("1.", TRAIN_LOG_CSV)
    print("2.", BEST_MODEL_PATH)
    print("3.", TEST_RESULT_CSV)
    print("4.", TEST_SUMMARY_TXT)
    print("5.", CLASS_MAPPING_JSON)

    print("\nResNet18 baseline 完成。")


if __name__ == "__main__":
    main()