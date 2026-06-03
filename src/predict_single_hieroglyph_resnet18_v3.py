# -*- coding: utf-8 -*-
"""
predict_single_hieroglyph_resnet18_v3.py

功能：
1. 加载训练好的 ResNet18 古埃及文字识别模型；
2. 输入一张单符号图片；
3. 输出 Top-5 Gardiner 编号预测结果；
4. 保存预测结果 CSV；
5. 为后续 Streamlit 上传识别功能做准备。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"

C:\\egypt_env\\Scripts\\python.exe src\\predict_single_hieroglyph_resnet18_v3.py ^
  --image "image_recognition_v3\\data\\real_external_test\\met_neferiu_cropped_signs\\MET2026_D4_001.png"
"""

from pathlib import Path
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

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
    / "single_prediction_resnet18"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "single_prediction_result.csv"


# =========================================================
# 2. 参数设置
# =========================================================

IMAGE_SIZE = 224
TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 3. 常用 Gardiner 编号解释
# =========================================================

GARDINER_NAME_MAP = {
    "D4": "eye",
    "D21": "mouth",
    "N35": "water ripple",
    "M17": "reed leaf",
    "X1": "bread loaf",
    "Z1": "single stroke",
    "G17": "owl",
    "G43": "quail chick",
    "I9": "horned viper",
    "V30": "basket",
    "S29": "folded cloth",
    "N33": "grain / sand",
    "R8": "standard / divine emblem",
}


# =========================================================
# 4. 图像预处理
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
# 5. 模型加载
# =========================================================

def build_resnet18_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{MODEL_PATH}\n"
            "请先运行 train_resnet18_baseline_v3.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    num_classes = int(checkpoint["num_classes"])
    id_to_code = checkpoint["id_to_code"]
    id_to_code = {int(k): v for k, v in id_to_code.items()}

    model = build_resnet18_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, id_to_code


# =========================================================
# 6. 单张图片预测
# =========================================================

def predict_image(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"未找到输入图片：{image_path}")

    model, id_to_code = load_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = probs.topk(TOP_K, dim=1)

    top_indices = top_indices[0].cpu().tolist()
    top_probs = top_probs[0].cpu().tolist()

    records = []

    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        gardiner_code = id_to_code[int(idx)]
        english_name = GARDINER_NAME_MAP.get(gardiner_code, "unknown")

        records.append({
            "rank": rank,
            "gardiner_code": gardiner_code,
            "english_name": english_name,
            "probability": float(prob),
            "probability_percent": f"{prob * 100:.2f}%"
        })

    result_df = pd.DataFrame(records)

    return result_df


# =========================================================
# 7. 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-image prediction for Egyptian hieroglyph recognition."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a cropped single hieroglyph image."
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path

    print("=" * 90)
    print("ResNet18 单张古埃及符号图片预测")
    print("=" * 90)
    print(f"运行设备：{DEVICE}")
    print(f"输入图片：{image_path}")
    print(f"模型文件：{MODEL_PATH}")

    result_df = predict_image(image_path)

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nTop-5 预测结果：")
    print(result_df.to_string(index=False))

    print("\n输出文件：")
    print(OUTPUT_CSV)

    print("\n预测完成。")


if __name__ == "__main__":
    main()