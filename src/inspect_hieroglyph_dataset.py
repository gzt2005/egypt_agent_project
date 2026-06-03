# -*- coding: utf-8 -*-
"""
inspect_hieroglyph_dataset.py

功能：
1. 自动寻找 EgyptianHieroglyphicText 数据集目录；
2. 扫描数据集中所有图片文件；
3. 自动判断可能的图片根目录；
4. 统计图片总数、类别数量、每类样本数量；
5. 生成数据清单 CSV 和类别分布 CSV；
6. 为后续古埃及文字图像识别训练/评估做数据准备。

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\inspect_hieroglyph_dataset.py
"""

from pathlib import Path
import pandas as pd


# =========================================================
# 1. 项目路径设置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "outputs"
    / "dataset_reports"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# =========================================================
# 2. 候选数据集路径
# =========================================================
# 这里把可能的数据集位置都列出来。
# 程序会自动选择第一个存在的路径。

CANDIDATE_DATASET_ROOTS = [
    # 方案 1：如果你把数据集复制到了项目目录下
    PROJECT_DIR
    / "image_recognition_v3"
    / "data"
    / "external_datasets"
    / "EgyptianHieroglyphicText",

    # 方案 2：如果你是浏览器 ZIP 解压的，可能叫这个名字
    PROJECT_DIR
    / "image_recognition_v3"
    / "data"
    / "external_datasets"
    / "EgyptianHieroglyphicText-main",

    # 方案 3：GitHub Desktop 默认 clone 位置，你当前就是这个
    Path(r"C:\Users\GE ZITONG\Documents\GitHub\EgyptianHieroglyphicText"),

    # 方案 4：如果你手动下载到了用户目录
    Path(r"C:\Users\GE ZITONG\EgyptianHieroglyphicText"),
]


# =========================================================
# 3. 基础工具函数
# =========================================================

def find_existing_dataset_root() -> Path | None:
    """
    从候选路径中寻找第一个存在的数据集目录。
    """
    for root in CANDIDATE_DATASET_ROOTS:
        if root.exists() and root.is_dir():
            return root
    return None


def is_image_file(file_path: Path) -> bool:
    """
    判断是否是图片文件。
    """
    return file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS


def count_images_in_folder(folder: Path) -> int:
    """
    统计某个文件夹及其子文件夹下的图片数量。
    """
    count = 0
    for file in folder.rglob("*"):
        if is_image_file(file):
            count += 1
    return count


# =========================================================
# 4. 自动寻找可能的图片根目录
# =========================================================

def find_candidate_image_roots(dataset_root: Path) -> pd.DataFrame:
    """
    扫描数据集仓库下所有包含图片的目录，并统计图片数量。

    为什么要这样做？
    因为 GitHub 数据集的目录结构可能不固定，有的叫 Dataset，
    有的叫 Glyph2025，有的图片藏在多级目录里。
    """
    records = []

    for folder in dataset_root.rglob("*"):
        if not folder.is_dir():
            continue

        image_count = count_images_in_folder(folder)

        if image_count > 0:
            records.append({
                "folder": str(folder),
                "image_count": image_count,
                "folder_name": folder.name,
                "relative_folder": str(folder.relative_to(dataset_root))
            })

    if not records:
        return pd.DataFrame(columns=[
            "folder", "image_count", "folder_name", "relative_folder"
        ])

    df = pd.DataFrame(records)
    df = df.sort_values("image_count", ascending=False).reset_index(drop=True)
    return df


# =========================================================
# 5. 构建图片清单
# =========================================================

def guess_class_name(image_root: Path, image_path: Path) -> str:
    """
    根据图片相对路径推测类别名。

    默认规则：
    - 如果 image_root 下一级文件夹是类别，则取第一级文件夹名；
    - 如果图片直接放在 image_root 下，则类别记为 unknown。
    """
    relative_path = image_path.relative_to(image_root)
    parts = relative_path.parts

    if len(parts) >= 2:
        return parts[0]

    return "unknown"


def build_inventory(image_root: Path) -> pd.DataFrame:
    """
    生成完整图片清单。

    输出字段：
    - image_id
    - class_name
    - file_name
    - image_path
    - relative_path
    - suffix
    """
    records = []

    image_files = [
        file for file in image_root.rglob("*")
        if is_image_file(file)
    ]

    for idx, file in enumerate(image_files, start=1):
        class_name = guess_class_name(image_root, file)

        records.append({
            "image_id": f"IMG_{idx:06d}",
            "class_name": class_name,
            "file_name": file.name,
            "image_path": str(file),
            "relative_path": str(file.relative_to(image_root)),
            "suffix": file.suffix.lower()
        })

    return pd.DataFrame(records)


# =========================================================
# 6. 数据集统计
# =========================================================

def build_class_distribution(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    统计每个类别下的图片数量。
    """
    if inventory_df.empty:
        return pd.DataFrame(columns=["class_name", "image_count"])

    class_stats = (
        inventory_df
        .groupby("class_name")
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
        .reset_index(drop=True)
    )

    return class_stats


def print_dataset_summary(
    dataset_root: Path,
    image_root: Path,
    inventory_df: pd.DataFrame,
    class_stats: pd.DataFrame
) -> None:
    """
    在终端打印数据集概览。
    """
    print("\n" + "=" * 90)
    print("数据集检查结果")
    print("=" * 90)

    print(f"数据集根目录：{dataset_root}")
    print(f"图片根目录：{image_root}")
    print(f"图片总数：{len(inventory_df)}")
    print(f"类别数量：{inventory_df['class_name'].nunique()}")

    print("\n最多样本类别 Top 10：")
    if not class_stats.empty:
        print(class_stats.head(10).to_string(index=False))
    else:
        print("无类别统计结果。")

    print("\n最少样本类别 Top 10：")
    if not class_stats.empty:
        print(class_stats.tail(10).to_string(index=False))
    else:
        print("无类别统计结果。")


# =========================================================
# 7. 主流程
# =========================================================

def main():
    print("=" * 90)
    print("开始检查 EgyptianHieroglyphicText 数据集")
    print("=" * 90)

    print("\n候选数据集路径：")
    for root in CANDIDATE_DATASET_ROOTS:
        print(f"- {root}")

    dataset_root = find_existing_dataset_root()

    if dataset_root is None:
        print("\n错误：没有找到 EgyptianHieroglyphicText 数据集目录。")
        print("请确认你已经通过 GitHub Desktop 克隆成功，或把数据集复制到：")
        print(
            PROJECT_DIR
            / "image_recognition_v3"
            / "data"
            / "external_datasets"
            / "EgyptianHieroglyphicText"
        )
        return

    print(f"\n已找到数据集目录：{dataset_root}")

    print("\n正在扫描包含图片的目录，请稍候...")
    candidate_roots_df = find_candidate_image_roots(dataset_root)

    candidate_roots_out = OUTPUT_DIR / "candidate_image_roots.csv"
    candidate_roots_df.to_csv(
        candidate_roots_out,
        index=False,
        encoding="utf-8-sig"
    )

    if candidate_roots_df.empty:
        print("\n没有在数据集目录下找到图片文件。")
        print("可能原因：")
        print("1. 仓库中只有代码，没有直接包含图片；")
        print("2. 图片需要通过仓库脚本额外下载；")
        print("3. Git LFS 文件没有拉取完整；")
        print("4. 数据集目录结构与预期不同。")
        print(f"\n候选目录扫描结果已保存：{candidate_roots_out}")
        return

    print("\n候选图片目录 Top 10：")
    print(candidate_roots_df.head(10).to_string(index=False))
    print(f"\n候选目录报告已保存：{candidate_roots_out}")

    # 默认选择图片最多的目录作为图片根目录
    image_root = Path(candidate_roots_df.iloc[0]["folder"])

    print("\n默认选择图片最多的目录作为 image_root：")
    print(image_root)

    print("\n正在生成图片清单...")
    inventory_df = build_inventory(image_root)

    if inventory_df.empty:
        print("\n图片清单为空。")
        print("请检查图片根目录是否正确。")
        return

    inventory_out = OUTPUT_DIR / "dataset_inventory.csv"
    inventory_df.to_csv(
        inventory_out,
        index=False,
        encoding="utf-8-sig"
    )

    class_stats = build_class_distribution(inventory_df)

    class_stats_out = OUTPUT_DIR / "class_distribution.csv"
    class_stats.to_csv(
        class_stats_out,
        index=False,
        encoding="utf-8-sig"
    )

    print_dataset_summary(
        dataset_root=dataset_root,
        image_root=image_root,
        inventory_df=inventory_df,
        class_stats=class_stats
    )

    print("\n输出文件：")
    print(f"1. 候选图片目录：{candidate_roots_out}")
    print(f"2. 图片清单：{inventory_out}")
    print(f"3. 类别分布：{class_stats_out}")

    print("\n下一步建议：")
    print("1. 打开 class_distribution.csv，查看类别数量和每类样本数；")
    print("2. 打开 dataset_inventory.csv，确认 image_path 是否能正常指向图片；")
    print("3. 如果图片总数接近 13,729、类别数接近 310，说明数据集较完整；")
    print("4. 如果数量很少，说明仓库可能没有下载完整或图片需要额外获取。")

    print("\n检查完成。")


if __name__ == "__main__":
    main()