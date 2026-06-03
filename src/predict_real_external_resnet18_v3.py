# -*- coding: utf-8 -*-
"""
predict_real_external_resnet18_v3.py

功能：
1. 读取真实外部测试集 real_external_test_labels.csv；
2. 加载已经训练好的 ResNet18 baseline 模型；
3. 对真实裁剪符号图像进行 Top-5 预测；
4. 输出每张图片的 Top-1 / Top-3 / Top-5 命中情况；
5. 生成预测结果 CSV 和总结报告。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\predict_real_external_resnet18_v3.py
"""

from pathlib import Path
import json

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

REAL_TEST_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "data"
    / "real_external_test"
)

LABEL_CSV = REAL_TEST_DIR / "real_external_test_labels.csv"

MODEL_PATH = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "resnet18_baseline"
    / "models"
    / "best_resnet18_min20.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "real_external_test_resnet18"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_CSV = OUTPUT_DIR / "real_external_test_predictions.csv"
SUMMARY_TXT = OUTPUT_DIR / "real_external_test_summary.txt"


# =========================================================
# 2. 参数设置
# =========================================================

IMAGE_SIZE = 224
TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 3. 图像预处理
# =========================================================

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 4. 加载标签表
# =========================================================

def load_real_external_labels() -> pd.DataFrame:
    if not LABEL_CSV.exists():
        raise FileNotFoundError(
            f"未找到真实外部测试集标签表：{LABEL_CSV}\n"
            "请先运行 create_real_external_labels_csv.py"
        )

    df = pd.read_csv(LABEL_CSV, dtype=str).fillna("")

    required_cols = [
        "image_id",
        "image_path",
        "true_gardiner_code",
        "true_name",
        "source_museum",
        "object_title",
        "object_id",
        "object_url",
        "license",
        "source_type",
        "note"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"real_external_test_labels.csv 缺少字段：{col}")

    # 拼接完整图片路径
    df["full_image_path"] = df["image_path"].apply(
        lambda x: str(REAL_TEST_DIR / x)
    )

    return df


# =========================================================
# 5. 加载模型
# =========================================================

def build_resnet18_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{MODEL_PATH}\n"
            "请先运行 train_resnet18_baseline_v3.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    num_classes = int(checkpoint["num_classes"])
    id_to_code = checkpoint["id_to_code"]

    # checkpoint 中 JSON-like dict 的 key 有可能是 int，也可能是 str
    id_to_code = {int(k): v for k, v in id_to_code.items()}

    model = build_resnet18_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, id_to_code


# =========================================================
# 6. 单图预测
# =========================================================

def predict_one_image(model, image_path: str, id_to_code: dict):
    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = probs.topk(TOP_K, dim=1)

    top_indices = top_indices[0].cpu().tolist()
    top_probs = top_probs[0].cpu().tolist()

    top_codes = [id_to_code[int(idx)] for idx in top_indices]

    return top_codes, top_probs


# =========================================================
# 7. 主流程
# =========================================================

def main():
    print("=" * 90)
    print("开始使用 ResNet18 测试真实外部古埃及符号裁剪图")
    print("=" * 90)
    print(f"运行设备：{DEVICE}")
    print(f"标签表：{LABEL_CSV}")
    print(f"模型文件：{MODEL_PATH}")

    df = load_real_external_labels()

    print("\n真实外部测试集信息：")
    print(f"图片数量：{len(df)}")
    print("类别分布：")
    print(df["true_gardiner_code"].value_counts().to_string())

    model, id_to_code = load_trained_model()

    # 当前模型覆盖的类别
    model_codes = set(id_to_code.values())

    records = []

    for _, row in df.iterrows():
        image_id = row["image_id"]
        image_path = row["full_image_path"]
        true_code = row["true_gardiner_code"]

        if true_code == "UNKNOWN":
            is_in_model_classes = False
        else:
            is_in_model_classes = true_code in model_codes

        if not Path(image_path).exists():
            records.append({
                **row.to_dict(),
                "is_in_model_classes": is_in_model_classes,
                "status": "missing_image",
                "pred_top1_code": "",
                "pred_top1_prob": "",
                "top1_hit": False,
                "top3_hit": False,
                "top5_hit": False,
                "pred_top5_codes": "",
                "pred_top5_probs": ""
            })
            continue

        try:
            top_codes, top_probs = predict_one_image(
                model=model,
                image_path=image_path,
                id_to_code=id_to_code
            )

            top1_hit = true_code == top_codes[0]
            top3_hit = true_code in top_codes[:3]
            top5_hit = true_code in top_codes[:5]

            records.append({
                **row.to_dict(),
                "is_in_model_classes": is_in_model_classes,
                "status": "success",
                "pred_top1_code": top_codes[0],
                "pred_top1_prob": f"{top_probs[0]:.6f}",
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "top5_hit": top5_hit,
                "pred_top5_codes": "|".join(top_codes),
                "pred_top5_probs": "|".join([f"{p:.6f}" for p in top_probs])
            })

        except Exception as e:
            records.append({
                **row.to_dict(),
                "is_in_model_classes": is_in_model_classes,
                "status": f"failed: {e}",
                "pred_top1_code": "",
                "pred_top1_prob": "",
                "top1_hit": False,
                "top3_hit": False,
                "top5_hit": False,
                "pred_top5_codes": "",
                "pred_top5_probs": ""
            })

    result_df = pd.DataFrame(records)

    # 只统计有明确标签且标签在模型覆盖类别中的样本
    eval_df = result_df[
        (result_df["status"] == "success")
        & (result_df["true_gardiner_code"] != "UNKNOWN")
        & (result_df["is_in_model_classes"] == True)
    ].copy()

    if len(eval_df) > 0:
        top1_acc = eval_df["top1_hit"].mean()
        top3_acc = eval_df["top3_hit"].mean()
        top5_acc = eval_df["top5_hit"].mean()
    else:
        top1_acc = 0.0
        top3_acc = 0.0
        top5_acc = 0.0

    result_df.to_csv(PREDICTION_CSV, index=False, encoding="utf-8-sig")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("真实外部古埃及符号裁剪图 ResNet18 测试报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"总样本数：{len(result_df)}\n")
        f.write(f"有效评估样本数：{len(eval_df)}\n")
        f.write(f"Top-1 Accuracy：{top1_acc:.4f}\n")
        f.write(f"Top-3 Accuracy：{top3_acc:.4f}\n")
        f.write(f"Top-5 Accuracy：{top5_acc:.4f}\n\n")
        f.write("类别分布：\n")
        f.write(result_df["true_gardiner_code"].value_counts().to_string())
        f.write("\n\n")
        f.write("逐图预测结果：\n")
        show_cols = [
            "image_id",
            "true_gardiner_code",
            "pred_top1_code",
            "pred_top1_prob",
            "top1_hit",
            "top3_hit",
            "top5_hit",
            "pred_top5_codes"
        ]
        f.write(result_df[show_cols].to_string(index=False))
        f.write("\n")

    print("\n真实外部测试结果：")
    print(f"总样本数：{len(result_df)}")
    print(f"有效评估样本数：{len(eval_df)}")
    print(f"Top-1 Accuracy：{top1_acc:.4f}")
    print(f"Top-3 Accuracy：{top3_acc:.4f}")
    print(f"Top-5 Accuracy：{top5_acc:.4f}")

    print("\n逐图预测结果：")
    show_cols = [
        "image_id",
        "true_gardiner_code",
        "pred_top1_code",
        "pred_top1_prob",
        "top1_hit",
        "top3_hit",
        "top5_hit",
        "pred_top5_codes"
    ]
    print(result_df[show_cols].to_string(index=False))

    print("\n输出文件：")
    print("1.", PREDICTION_CSV)
    print("2.", SUMMARY_TXT)

    print("\n外部真实测试完成。")


if __name__ == "__main__":
    main()