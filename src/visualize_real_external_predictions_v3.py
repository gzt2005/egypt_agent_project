# -*- coding: utf-8 -*-
"""
visualize_real_external_predictions_v3.py

功能：
1. 读取真实外部测试预测结果 real_external_test_predictions.csv；
2. 读取每张真实裁剪图；
3. 生成可视化网格图；
4. 每个格子显示：
   - 裁剪图
   - 真实 Gardiner 编号
   - Top-1 预测编号
   - Top-5 候选
   - 是否 Top-1 命中
5. 输出一张适合写报告/展示的 PNG 图。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\visualize_real_external_predictions_v3.py
"""

from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont


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

PREDICTION_CSV = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "real_external_test_resnet18"
    / "real_external_test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "real_external_test_resnet18"
    / "visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "real_external_predictions_grid.png"
OUTPUT_SUMMARY = OUTPUT_DIR / "visualization_summary.txt"


# =========================================================
# 2. 可视化参数
# =========================================================

THUMB_SIZE = 150
TEXT_HEIGHT = 92
COLS = 4

BG_COLOR = "white"
GOOD_COLOR = (220, 245, 220)
BAD_COLOR = (255, 225, 225)
BORDER_COLOR = (180, 180, 180)
TEXT_COLOR = "black"


# =========================================================
# 3. 数据读取
# =========================================================

def load_predictions() -> pd.DataFrame:
    if not PREDICTION_CSV.exists():
        raise FileNotFoundError(
            f"未找到预测结果文件：{PREDICTION_CSV}\n"
            "请先运行 predict_real_external_resnet18_v3.py"
        )

    df = pd.read_csv(PREDICTION_CSV, dtype=str).fillna("")

    required_cols = [
        "image_id",
        "image_path",
        "true_gardiner_code",
        "pred_top1_code",
        "pred_top1_prob",
        "top1_hit",
        "pred_top5_codes"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"预测结果 CSV 缺少字段：{col}")

    # 将字符串 True/False 转为布尔
    df["top1_hit_bool"] = (
        df["top1_hit"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )

    return df


# =========================================================
# 4. 图片处理
# =========================================================

def safe_open_image(image_path: Path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def make_thumbnail(img: Image.Image, size: int = THUMB_SIZE) -> Image.Image:
    """
    保持比例缩放，并用白底填充。
    """
    img = ImageOps.contain(img, (size, size))

    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))

    return canvas


def draw_multiline_text(draw, xy, lines, line_height=16):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=TEXT_COLOR)
        y += line_height


# =========================================================
# 5. 生成可视化网格
# =========================================================

def create_prediction_grid(df: pd.DataFrame, output_path: Path):
    n = len(df)

    rows = (n + COLS - 1) // COLS

    cell_w = THUMB_SIZE
    cell_h = THUMB_SIZE + TEXT_HEIGHT

    grid_w = COLS * cell_w
    grid_h = rows * cell_h

    canvas = Image.new("RGB", (grid_w, grid_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    for idx, (_, row) in enumerate(df.iterrows()):
        col = idx % COLS
        row_idx = idx // COLS

        x0 = col * cell_w
        y0 = row_idx * cell_h

        hit = bool(row["top1_hit_bool"])
        cell_bg = GOOD_COLOR if hit else BAD_COLOR

        # cell 背景
        draw.rectangle(
            [x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
            fill=cell_bg,
            outline=BORDER_COLOR
        )

        # 打开图片
        image_path = REAL_TEST_DIR / row["image_path"]
        img = safe_open_image(image_path)

        if img is not None:
            thumb = make_thumbnail(img)
            canvas.paste(thumb, (x0, y0))
        else:
            draw.text((x0 + 8, y0 + 60), "Image missing", fill="red")

        # 文本
        true_code = row["true_gardiner_code"]
        pred_code = row["pred_top1_code"]
        prob = row["pred_top1_prob"]
        top5 = row["pred_top5_codes"]

        # Top5 太长，截短显示
        top5_short = top5
        if len(top5_short) > 22:
            top5_short = top5_short[:22] + "..."

        mark = "OK" if hit else "ERR"

        text_lines = [
            f"{mark}  T:{true_code} P:{pred_code}",
            f"p={prob}",
            f"Top5:{top5_short}",
            f"{row['image_id'][:18]}"
        ]

        draw_multiline_text(
            draw=draw,
            xy=(x0 + 4, y0 + THUMB_SIZE + 4),
            lines=text_lines,
            line_height=18
        )

    canvas.save(output_path)


# =========================================================
# 6. 主流程
# =========================================================

def main():
    print("=" * 90)
    print("开始生成真实外部测试结果可视化图")
    print("=" * 90)

    df = load_predictions()

    print(f"预测结果数量：{len(df)}")

    top1_acc = df["top1_hit_bool"].mean()

    print(f"Top-1 Accuracy：{top1_acc:.4f}")

    print("\n类别分布：")
    print(df["true_gardiner_code"].value_counts().to_string())

    print("\n正在生成可视化图...")
    create_prediction_grid(df, OUTPUT_IMAGE)

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("真实外部测试结果可视化报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"样本数量：{len(df)}\n")
        f.write(f"Top-1 Accuracy：{top1_acc:.4f}\n")
        f.write(f"输出图片：{OUTPUT_IMAGE}\n\n")
        f.write("类别分布：\n")
        f.write(df["true_gardiner_code"].value_counts().to_string())
        f.write("\n")

    print("\n输出文件：")
    print("1.", OUTPUT_IMAGE)
    print("2.", OUTPUT_SUMMARY)

    print("\n可视化完成。")


if __name__ == "__main__":
    main()