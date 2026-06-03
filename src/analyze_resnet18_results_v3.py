# -*- coding: utf-8 -*-
"""
analyze_resnet18_results_v3.py

功能：
1. 读取 ResNet18 测试集预测结果；
2. 统计整体 Top-1 / Top-3 / Top-5 Accuracy；
3. 统计每个 Gardiner 类别的准确率；
4. 找出最容易识别错的类别；
5. 统计最常见混淆对；
6. 输出错误案例表；
7. 生成 Top-1 错误案例预览图。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\analyze_resnet18_results_v3.py
"""

from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

PRED_CSV = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "resnet18_baseline"
    / "results"
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "resnet18_baseline"
    / "analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_METRICS_CSV = OUTPUT_DIR / "overall_metrics.csv"
PER_CLASS_METRICS_CSV = OUTPUT_DIR / "per_class_metrics.csv"
CONFUSION_PAIRS_CSV = OUTPUT_DIR / "confusion_pairs_top30.csv"
ERROR_CASES_CSV = OUTPUT_DIR / "top1_error_cases.csv"
ERROR_PREVIEW_PNG = OUTPUT_DIR / "top1_error_cases_preview.png"
ANALYSIS_SUMMARY_TXT = OUTPUT_DIR / "analysis_summary.txt"


# =========================================================
# 2. 参数设置
# =========================================================

PREVIEW_CASES = 24
THUMB_SIZE = 128


# =========================================================
# 3. 读取数据
# =========================================================

def load_predictions() -> pd.DataFrame:
    if not PRED_CSV.exists():
        raise FileNotFoundError(
            f"未找到预测结果文件：{PRED_CSV}\n"
            "请先运行 train_resnet18_baseline_v3.py"
        )

    df = pd.read_csv(PRED_CSV, dtype=str).fillna("")

    bool_cols = ["top1_hit", "top3_hit", "top5_hit"]
    for col in bool_cols:
        df[col] = df[col].astype(str).str.lower().map({
            "true": True,
            "false": False
        })

    df["pred_top1_prob"] = df["pred_top1_prob"].astype(float)

    return df


# =========================================================
# 4. 指标统计
# =========================================================

def compute_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "test_samples": len(df),
        "top1_accuracy": df["top1_hit"].mean(),
        "top3_accuracy": df["top3_hit"].mean(),
        "top5_accuracy": df["top5_hit"].mean(),
        "top1_error_count": int((~df["top1_hit"]).sum()),
        "top3_error_count": int((~df["top3_hit"]).sum()),
        "top5_error_count": int((~df["top5_hit"]).sum())
    }

    return pd.DataFrame([metrics])


def compute_per_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for code, group in df.groupby("true_gardiner_code"):
        records.append({
            "gardiner_code": code,
            "test_count": len(group),
            "top1_accuracy": group["top1_hit"].mean(),
            "top3_accuracy": group["top3_hit"].mean(),
            "top5_accuracy": group["top5_hit"].mean(),
            "top1_error_count": int((~group["top1_hit"]).sum()),
            "avg_top1_prob": group["pred_top1_prob"].mean()
        })

    result = pd.DataFrame(records)
    result = result.sort_values(
        by=["top1_accuracy", "test_count"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return result


def compute_confusion_pairs(df: pd.DataFrame) -> pd.DataFrame:
    error_df = df[df["top1_hit"] == False].copy()

    if error_df.empty:
        return pd.DataFrame(columns=[
            "true_gardiner_code",
            "pred_top1_code",
            "count"
        ])

    confusion = (
        error_df
        .groupby(["true_gardiner_code", "pred_top1_code"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return confusion.head(30)


# =========================================================
# 5. 错误案例预览图
# =========================================================

def safe_open_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def make_error_preview(error_df: pd.DataFrame, output_path: Path):
    if error_df.empty:
        return

    sample_df = error_df.head(PREVIEW_CASES).copy()

    cols = 4
    rows = (len(sample_df) + cols - 1) // cols

    cell_w = THUMB_SIZE
    cell_h = THUMB_SIZE + 48

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img = safe_open_image(row["image_path"])
        if img is None:
            continue

        img = ImageOps.contain(img, (THUMB_SIZE, THUMB_SIZE))

        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h

        cell = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), "white")
        paste_x = (THUMB_SIZE - img.width) // 2
        paste_y = (THUMB_SIZE - img.height) // 2
        cell.paste(img, (paste_x, paste_y))

        canvas.paste(cell, (x, y))

        text1 = f"T: {row['true_gardiner_code']}"
        text2 = f"P: {row['pred_top1_code']}"
        text3 = f"p={float(row['pred_top1_prob']):.2f}"

        draw.text((x + 4, y + THUMB_SIZE + 2), text1, fill="black")
        draw.text((x + 4, y + THUMB_SIZE + 18), text2, fill="black")
        draw.text((x + 4, y + THUMB_SIZE + 34), text3, fill="black")

    canvas.save(output_path)


# =========================================================
# 6. 主流程
# =========================================================

def main():
    print("=" * 90)
    print("开始分析 ResNet18 古埃及文字识别结果")
    print("=" * 90)

    df = load_predictions()

    overall_df = compute_overall_metrics(df)
    per_class_df = compute_per_class_metrics(df)
    confusion_df = compute_confusion_pairs(df)

    error_df = df[df["top1_hit"] == False].copy()
    error_df = error_df.sort_values(
        by=["true_gardiner_code", "pred_top1_prob"],
        ascending=[True, False]
    ).reset_index(drop=True)

    overall_df.to_csv(OVERALL_METRICS_CSV, index=False, encoding="utf-8-sig")
    per_class_df.to_csv(PER_CLASS_METRICS_CSV, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(CONFUSION_PAIRS_CSV, index=False, encoding="utf-8-sig")
    error_df.to_csv(ERROR_CASES_CSV, index=False, encoding="utf-8-sig")

    make_error_preview(error_df, ERROR_PREVIEW_PNG)

    print("\n整体指标：")
    print(overall_df.to_string(index=False))

    print("\nTop-1 准确率最低的类别 Top 10：")
    print(per_class_df.head(10).to_string(index=False))

    print("\n最常见混淆对 Top 10：")
    if not confusion_df.empty:
        print(confusion_df.head(10).to_string(index=False))
    else:
        print("没有 Top-1 错误。")

    with open(ANALYSIS_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("ResNet18 古埃及文字识别结果分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write("整体指标：\n")
        f.write(overall_df.to_string(index=False))
        f.write("\n\n")
        f.write("Top-1 准确率最低的类别 Top 10：\n")
        f.write(per_class_df.head(10).to_string(index=False))
        f.write("\n\n")
        f.write("最常见混淆对 Top 10：\n")
        if not confusion_df.empty:
            f.write(confusion_df.head(10).to_string(index=False))
        else:
            f.write("没有 Top-1 错误。")
        f.write("\n\n")
        f.write("输出文件：\n")
        f.write(f"1. {OVERALL_METRICS_CSV}\n")
        f.write(f"2. {PER_CLASS_METRICS_CSV}\n")
        f.write(f"3. {CONFUSION_PAIRS_CSV}\n")
        f.write(f"4. {ERROR_CASES_CSV}\n")
        f.write(f"5. {ERROR_PREVIEW_PNG}\n")

    print("\n输出文件：")
    print("1.", OVERALL_METRICS_CSV)
    print("2.", PER_CLASS_METRICS_CSV)
    print("3.", CONFUSION_PAIRS_CSV)
    print("4.", ERROR_CASES_CSV)
    print("5.", ERROR_PREVIEW_PNG)
    print("6.", ANALYSIS_SUMMARY_TXT)

    print("\n分析完成。")


if __name__ == "__main__":
    main()