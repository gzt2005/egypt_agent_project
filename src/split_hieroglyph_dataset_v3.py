# -*- coding: utf-8 -*-
"""
split_hieroglyph_dataset_v3.py

功能：
1. 读取 filtered_inventory_min20.csv；
2. 对每个 Gardiner 类别进行分层划分；
3. 生成 train / val / test 数据集；
4. 输出 split_inventory_min20.csv；
5. 输出各 split 的类别分布统计；
6. 为后续 ResNet / MobileNet / DINOv2 baseline 训练和评估做准备。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\split_hieroglyph_dataset_v3.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "dataset_preparation"
    / "filtered_inventory_min20.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "dataset_splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SPLIT_CSV = OUTPUT_DIR / "split_inventory_min20.csv"
OUTPUT_SPLIT_STATS_CSV = OUTPUT_DIR / "split_class_distribution_min20.csv"
OUTPUT_SUMMARY_TXT = OUTPUT_DIR / "split_summary_min20.txt"


# =========================================================
# 2. 划分参数
# =========================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


# =========================================================
# 3. 数据读取
# =========================================================

def load_filtered_inventory() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{INPUT_CSV}\n"
            "请先运行 prepare_hieroglyph_dataset_v3.py"
        )

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    required_cols = [
        "image_id",
        "class_name",
        "file_name",
        "image_path",
        "relative_path",
        "suffix",
        "gardiner_code",
        "label_id"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"filtered_inventory_min20.csv 缺少字段：{col}")

    df["label_id"] = df["label_id"].astype(int)

    return df


# =========================================================
# 4. 分层划分
# =========================================================

def split_one_class(class_df: pd.DataFrame) -> pd.DataFrame:
    """
    对单个类别进行 train / val / test 划分。
    每类至少 20 张，所以可以稳定划分。
    """
    n = len(class_df)

    if n < 3:
        raise ValueError(
            f"类别 {class_df['gardiner_code'].iloc[0]} 样本数过少，无法划分。"
        )

    # 第一次：划分 train 和 temp
    train_df, temp_df = train_test_split(
        class_df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # 第二次：temp 再均分为 val 和 test
    val_relative_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_ratio,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def stratified_split_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 gardiner_code 分组，每类内部单独划分。
    """
    split_parts = []

    for code, class_df in df.groupby("gardiner_code"):
        split_df = split_one_class(class_df)
        split_parts.append(split_df)

    result_df = pd.concat(split_parts, ignore_index=True)

    result_df = result_df.sort_values(
        by=["split", "gardiner_code", "image_id"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return result_df


# =========================================================
# 5. 统计输出
# =========================================================

def build_split_stats(split_df: pd.DataFrame) -> pd.DataFrame:
    """
    统计每个 split 下每个类别的图片数量。
    """
    stats = (
        split_df
        .groupby(["split", "gardiner_code"])
        .size()
        .reset_index(name="image_count")
        .sort_values(["split", "gardiner_code"])
        .reset_index(drop=True)
    )

    return stats


def build_overall_stats(split_df: pd.DataFrame) -> pd.DataFrame:
    """
    统计 train / val / test 总体图片数和类别数。
    """
    records = []

    for split_name, group_df in split_df.groupby("split"):
        records.append({
            "split": split_name,
            "image_count": len(group_df),
            "class_count": group_df["gardiner_code"].nunique()
        })

    overall_df = pd.DataFrame(records)

    split_order = {"train": 0, "val": 1, "test": 2}
    overall_df["split_order"] = overall_df["split"].map(split_order)
    overall_df = overall_df.sort_values("split_order").drop(columns=["split_order"])

    return overall_df


def write_summary(
    split_df: pd.DataFrame,
    split_stats: pd.DataFrame,
    overall_stats: pd.DataFrame
) -> None:
    with open(OUTPUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("古埃及文字图像识别数据集划分报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("划分比例：\n")
        f.write(f"train: {TRAIN_RATIO}\n")
        f.write(f"val: {VAL_RATIO}\n")
        f.write(f"test: {TEST_RATIO}\n\n")

        f.write("总体统计：\n")
        f.write(overall_stats.to_string(index=False))
        f.write("\n\n")

        f.write("数据集总图片数：")
        f.write(str(len(split_df)))
        f.write("\n")

        f.write("类别总数：")
        f.write(str(split_df["gardiner_code"].nunique()))
        f.write("\n\n")

        f.write("每类划分结果前 30 行：\n")
        f.write(split_stats.head(30).to_string(index=False))
        f.write("\n")


# =========================================================
# 6. 主流程
# =========================================================

def main():
    print("=" * 90)
    print("开始划分古埃及文字图像识别数据集 V3")
    print("=" * 90)

    df = load_filtered_inventory()

    print("\n输入数据：")
    print(f"图片总数：{len(df)}")
    print(f"类别数量：{df['gardiner_code'].nunique()}")

    print("\n开始按类别分层划分 train / val / test ...")
    split_df = stratified_split_by_class(df)

    split_stats = build_split_stats(split_df)
    overall_stats = build_overall_stats(split_df)

    split_df.to_csv(OUTPUT_SPLIT_CSV, index=False, encoding="utf-8-sig")
    split_stats.to_csv(OUTPUT_SPLIT_STATS_CSV, index=False, encoding="utf-8-sig")
    write_summary(split_df, split_stats, overall_stats)

    print("\n划分完成。")

    print("\n总体统计：")
    print(overall_stats.to_string(index=False))

    print("\n每个 split 的类别数量：")
    print(split_df.groupby("split")["gardiner_code"].nunique())

    print("\n输出文件：")
    print("1.", OUTPUT_SPLIT_CSV)
    print("2.", OUTPUT_SPLIT_STATS_CSV)
    print("3.", OUTPUT_SUMMARY_TXT)

    print("\n下一步建议：")
    print("1. 检查 split_inventory_min20.csv 是否包含 train/val/test；")
    print("2. 确认每个 split 都包含 88 个类别；")
    print("3. 下一步开始构建 PyTorch Dataset 和 DataLoader；")
    print("4. 第一版模型建议训练 ResNet18 或 MobileNetV3。")


if __name__ == "__main__":
    main()