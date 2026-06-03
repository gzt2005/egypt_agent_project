# -*- coding: utf-8 -*-
"""
prepare_hieroglyph_dataset_v3.py

功能：
1. 读取 EgyptianHieroglyphicText 数据集扫描结果；
2. 清洗类别名，将 n35 / m17 等转为 N35 / M17；
3. 统计类别样本分布；
4. 根据最小样本数阈值筛选可训练类别；
5. 输出 filtered_inventory_minXX.csv；
6. 生成类别样本预览图，检查图片质量。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\prepare_hieroglyph_dataset_v3.py
"""

from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "dataset_reports"
)

INVENTORY_CSV = REPORT_DIR / "dataset_inventory.csv"
CLASS_DIST_CSV = REPORT_DIR / "class_distribution.csv"

OUTPUT_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "dataset_preparation"
)

PREVIEW_DIR = OUTPUT_DIR / "preview_grids"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 参数设置
# =========================================================

# 第一版建议重点看这些阈值
MIN_SAMPLE_THRESHOLDS = [1, 5, 10, 20, 30, 50, 100]

# 默认筛选阈值：建议先用 20
DEFAULT_MIN_SAMPLES = 20

# 预览图参数
PREVIEW_CLASSES_TOP_N = 20
PREVIEW_IMAGES_PER_CLASS = 8
THUMB_SIZE = 96


# =========================================================
# 3. 工具函数
# =========================================================

def normalize_gardiner_code(class_name: str) -> str:
    """
    将类别名统一成大写 Gardiner 编号。
    例如：
    n35 -> N35
    m17 -> M17
    x1  -> X1
    """
    if pd.isna(class_name):
        return "UNKNOWN"

    text = str(class_name).strip()

    if not text:
        return "UNKNOWN"

    return text.upper()


def load_inventory() -> pd.DataFrame:
    """
    读取数据清单。
    """
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(
            f"未找到数据清单：{INVENTORY_CSV}\n"
            "请先运行 inspect_hieroglyph_dataset.py"
        )

    df = pd.read_csv(INVENTORY_CSV, dtype=str).fillna("")

    required_cols = ["image_id", "class_name", "file_name", "image_path", "relative_path", "suffix"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"dataset_inventory.csv 缺少字段：{col}")

    df["gardiner_code"] = df["class_name"].apply(normalize_gardiner_code)

    return df


def build_class_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    重新统计清洗后的类别分布。
    """
    stats = (
        df.groupby("gardiner_code")
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
        .reset_index(drop=True)
    )

    return stats


def summarize_thresholds(class_stats: pd.DataFrame) -> pd.DataFrame:
    """
    统计不同最小样本数阈值下，可用类别数量和图片数量。
    """
    records = []

    for threshold in MIN_SAMPLE_THRESHOLDS:
        valid_classes = class_stats[class_stats["image_count"] >= threshold]["gardiner_code"].tolist()

        valid_image_count = class_stats[class_stats["image_count"] >= threshold]["image_count"].sum()

        records.append({
            "min_samples_per_class": threshold,
            "valid_class_count": len(valid_classes),
            "valid_image_count": int(valid_image_count)
        })

    return pd.DataFrame(records)


def create_filtered_inventory(
    inventory_df: pd.DataFrame,
    class_stats: pd.DataFrame,
    min_samples: int
) -> pd.DataFrame:
    """
    根据最小样本数阈值筛选可训练数据。
    """
    valid_classes = set(
        class_stats[class_stats["image_count"] >= min_samples]["gardiner_code"].tolist()
    )

    filtered_df = inventory_df[inventory_df["gardiner_code"].isin(valid_classes)].copy()

    filtered_df = filtered_df.sort_values(
        by=["gardiner_code", "image_id"],
        ascending=[True, True]
    ).reset_index(drop=True)

    filtered_df["label_id"] = filtered_df["gardiner_code"].astype("category").cat.codes

    return filtered_df


def safe_open_image(path: str) -> Image.Image | None:
    """
    安全打开图片，失败则返回 None。
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


def make_thumbnail(img: Image.Image, size: int = THUMB_SIZE) -> Image.Image:
    """
    将图片转成固定尺寸缩略图，保持比例，白底填充。
    """
    img = ImageOps.contain(img, (size, size))

    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))

    return canvas


def make_preview_grid(
    inventory_df: pd.DataFrame,
    class_stats: pd.DataFrame,
    output_path: Path,
    top_n: int = PREVIEW_CLASSES_TOP_N,
    images_per_class: int = PREVIEW_IMAGES_PER_CLASS
) -> None:
    """
    为样本最多的 Top-N 类别生成预览图。
    """
    top_classes = class_stats.head(top_n)["gardiner_code"].tolist()

    cell_w = THUMB_SIZE
    cell_h = THUMB_SIZE + 24

    grid_w = images_per_class * cell_w
    grid_h = top_n * cell_h

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    for row_idx, code in enumerate(top_classes):
        class_df = inventory_df[inventory_df["gardiner_code"] == code].copy()

        if class_df.empty:
            continue

        sample_df = class_df.sample(
            n=min(images_per_class, len(class_df)),
            random_state=42
        )

        y0 = row_idx * cell_h

        # 写类别名
        draw.text((4, y0 + 2), code, fill="black")

        for col_idx, (_, row) in enumerate(sample_df.iterrows()):
            img = safe_open_image(row["image_path"])

            if img is None:
                continue

            thumb = make_thumbnail(img, THUMB_SIZE)

            x = col_idx * cell_w
            y = y0 + 24

            grid.paste(thumb, (x, y))

    grid.save(output_path)


# =========================================================
# 4. 主流程
# =========================================================

def main():
    print("=" * 90)
    print("开始准备古埃及文字图像识别数据集 V3")
    print("=" * 90)

    inventory_df = load_inventory()
    class_stats = build_class_stats(inventory_df)

    print("\n原始数据：")
    print(f"图片总数：{len(inventory_df)}")
    print(f"类别数量：{class_stats['gardiner_code'].nunique()}")

    print("\n样本最多类别 Top 10：")
    print(class_stats.head(10).to_string(index=False))

    print("\n样本最少类别 Top 10：")
    print(class_stats.tail(10).to_string(index=False))

    # 输出清洗后的类别分布
    clean_class_stats_out = OUTPUT_DIR / "class_distribution_clean.csv"
    class_stats.to_csv(clean_class_stats_out, index=False, encoding="utf-8-sig")

    # 阈值统计
    threshold_summary = summarize_thresholds(class_stats)
    threshold_summary_out = OUTPUT_DIR / "threshold_summary.csv"
    threshold_summary.to_csv(threshold_summary_out, index=False, encoding="utf-8-sig")

    print("\n不同样本阈值下的数据可用情况：")
    print(threshold_summary.to_string(index=False))

    # 默认筛选数据集
    filtered_df = create_filtered_inventory(
        inventory_df=inventory_df,
        class_stats=class_stats,
        min_samples=DEFAULT_MIN_SAMPLES
    )

    filtered_out = OUTPUT_DIR / f"filtered_inventory_min{DEFAULT_MIN_SAMPLES}.csv"
    filtered_df.to_csv(filtered_out, index=False, encoding="utf-8-sig")

    filtered_class_stats = build_class_stats(filtered_df)
    filtered_class_stats_out = OUTPUT_DIR / f"filtered_class_distribution_min{DEFAULT_MIN_SAMPLES}.csv"
    filtered_class_stats.to_csv(filtered_class_stats_out, index=False, encoding="utf-8-sig")

    print("\n默认筛选结果：")
    print(f"最小样本数阈值：{DEFAULT_MIN_SAMPLES}")
    print(f"筛选后图片数：{len(filtered_df)}")
    print(f"筛选后类别数：{filtered_df['gardiner_code'].nunique()}")

    print("\n筛选后类别 Top 10：")
    print(filtered_class_stats.head(10).to_string(index=False))

    # 生成预览图
    preview_out = PREVIEW_DIR / f"top{PREVIEW_CLASSES_TOP_N}_classes_preview.png"

    print("\n正在生成样本预览图...")
    make_preview_grid(
        inventory_df=inventory_df,
        class_stats=class_stats,
        output_path=preview_out,
        top_n=PREVIEW_CLASSES_TOP_N,
        images_per_class=PREVIEW_IMAGES_PER_CLASS
    )

    # 生成说明文件
    summary_txt = OUTPUT_DIR / "dataset_preparation_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("古埃及文字图像识别数据集准备报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"原始图片总数：{len(inventory_df)}\n")
        f.write(f"原始类别数量：{class_stats['gardiner_code'].nunique()}\n\n")
        f.write("不同样本阈值下的数据可用情况：\n")
        f.write(threshold_summary.to_string(index=False))
        f.write("\n\n")
        f.write(f"默认筛选阈值：每类至少 {DEFAULT_MIN_SAMPLES} 张\n")
        f.write(f"筛选后图片数：{len(filtered_df)}\n")
        f.write(f"筛选后类别数：{filtered_df['gardiner_code'].nunique()}\n\n")
        f.write("输出文件：\n")
        f.write(f"1. {clean_class_stats_out}\n")
        f.write(f"2. {threshold_summary_out}\n")
        f.write(f"3. {filtered_out}\n")
        f.write(f"4. {filtered_class_stats_out}\n")
        f.write(f"5. {preview_out}\n")

    print("\n输出文件：")
    print("1.", clean_class_stats_out)
    print("2.", threshold_summary_out)
    print("3.", filtered_out)
    print("4.", filtered_class_stats_out)
    print("5.", preview_out)
    print("6.", summary_txt)

    print("\n下一步建议：")
    print("1. 打开 threshold_summary.csv，决定第一版模型使用多少类别；")
    print("2. 打开 top20_classes_preview.png，检查图片质量；")
    print("3. 如果图片质量可用，下一步进行 train/val/test 划分；")
    print("4. 第一版建议先训练样本数 >= 20 的类别，或者人工选 50 个核心类别。")

    print("\n数据集准备完成。")


if __name__ == "__main__":
    main()