# -*- coding: utf-8 -*-
"""
resnet18_hieroglyph_predictor.py

功能：
1. 封装 ResNet18 古埃及单符号识别模型；
2. 支持从本地路径或 PIL Image 输入图片；
3. 返回 Top-5 Gardiner 预测结果；
4. 供 Streamlit app.py 调用。

使用场景：
- 命令行单图预测
- Streamlit 上传图片预测
- 后续系统图像识别模块
"""

from pathlib import Path
from typing import List, Dict, Union

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# =========================================================
# 1. 基础路径配置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "resnet18_baseline"
    / "models"
    / "best_resnet18_min20.pth"
)


# =========================================================
# 2. 参数配置
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
    "O1": "house",
    "O49": "village / settlement",
    "W11": "ring stand",
    "W24": "pot",
    "AA15": "unclassified sign",
    "V31": "basket handle",
    "G36": "swallow",
    "G40": "pintail duck",
    "E34": "hare",
    "Y1": "papyrus roll",
    "Y4": "scribal palette",
    "Z4": "dual strokes",
    "D28": "ka arms",
    "H6": "feather",
    "F35": "heart and windpipe",
    "S38": "crook",
    "Q3": "stool",
    "I10": "cobra",
    "D37": "arm with bread",
    "D36": "forearm",
    "F1": "head of ox",
    "N1": "sky",
    "Y5": "game board",
    "O34": "door bolt",
    "V13": "tethering rope",
    "U7": "hoe",
    "M23": "sedge",
    "X8": "conical bread",
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
# 5. 模型构建与加载
# =========================================================

def build_resnet18_model(num_classes: int) -> nn.Module:
    """
    构建 ResNet18 模型结构。
    注意：这里 weights=None，因为我们加载的是自己训练好的权重。
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class HieroglyphResNet18Predictor:
    """
    古埃及单符号 ResNet18 识别器。
    """

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.device = DEVICE
        self.model = None
        self.id_to_code = None
        self.num_classes = None

        self.load_model()

    def load_model(self):
        """
        加载训练好的 ResNet18 模型。
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到模型文件：{self.model_path}\n"
                "请确认 best_resnet18_min20.pth 是否存在。"
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.num_classes = int(checkpoint["num_classes"])
        self.id_to_code = checkpoint["id_to_code"]
        self.id_to_code = {int(k): v for k, v in self.id_to_code.items()}

        self.model = build_resnet18_model(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        PIL Image -> 模型输入 Tensor。
        """
        image = image.convert("RGB")
        image_tensor = eval_transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def predict_pil_image(
        self,
        image: Image.Image,
        top_k: int = TOP_K
    ) -> List[Dict]:
        """
        输入 PIL Image，返回 Top-K 预测结果。
        """
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = probs.topk(top_k, dim=1)

        top_indices = top_indices[0].cpu().tolist()
        top_probs = top_probs[0].cpu().tolist()

        results = []

        for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
            gardiner_code = self.id_to_code[int(idx)]
            english_name = GARDINER_NAME_MAP.get(gardiner_code, "unknown")

            results.append({
                "rank": rank,
                "gardiner_code": gardiner_code,
                "english_name": english_name,
                "probability": float(prob),
                "probability_percent": f"{prob * 100:.2f}%"
            })

        return results

    def predict_image_path(
        self,
        image_path: Union[str, Path],
        top_k: int = TOP_K
    ) -> List[Dict]:
        """
        输入图片路径，返回 Top-K 预测结果。
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"未找到输入图片：{image_path}")

        image = Image.open(image_path).convert("RGB")

        return self.predict_pil_image(image=image, top_k=top_k)


# =========================================================
# 6. 简单测试
# =========================================================

if __name__ == "__main__":
    test_image = (
        PROJECT_DIR
        / "image_recognition_v3"
        / "data"
        / "real_external_test"
        / "met_neferiu_cropped_signs"
        / "MET2026_D4_001.png"
    )

    predictor = HieroglyphResNet18Predictor()

    results = predictor.predict_image_path(test_image)

    print("=" * 80)
    print("ResNet18 古埃及单符号识别测试")
    print("=" * 80)
    print("测试图片：", test_image)
    print("\nTop-5 预测结果：")

    for item in results:
        print(
            f"Rank {item['rank']} | "
            f"{item['gardiner_code']} | "
            f"{item['english_name']} | "
            f"{item['probability_percent']}"
        )