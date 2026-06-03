# -*- coding: utf-8 -*-
"""
create_real_external_labels_csv.py

功能：
1. 递归扫描 real_external_test 文件夹下所有裁剪图片；
2. 自动跳过 original 原图文件夹；
3. 根据文件名解析来源前缀和 Gardiner 编号；
4. 自动生成 real_external_test_labels.csv；
5. 支持多个来源文件夹混合存放。

命名规则：
SOURCE_GARDINER_INDEX.png

示例：
MET2026_D4_001.png
MET98468_N35_001.png
MET543863_N35_001.png
DP226728_G17_001.png
DP322047_G43_001.png
MET19944964_G17_001.png
REAL_N35_001.png

运行方式：
cd "C:\\Users\\GE ZITONG\\Desktop\\egypt_agent_project"
C:\\egypt_env\\Scripts\\python.exe src\\create_real_external_labels_csv.py
"""

from pathlib import Path
import re
import pandas as pd


# =========================================================
# 1. 路径设置
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

REAL_TEST_DIR = (
    PROJECT_DIR
    / "image_recognition_v3"
    / "data"
    / "real_external_test"
)

OUTPUT_CSV = REAL_TEST_DIR / "real_external_test_labels.csv"

REAL_TEST_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 支持图片格式
# =========================================================

IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}


# =========================================================
# 3. Gardiner 编号到英文名映射
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
    "UNKNOWN": "unknown"
}


# =========================================================
# 4. 来源前缀映射
# =========================================================

SOURCE_INFO_MAP = {
    "MET2026": {
        "source_museum": "The Met",
        "object_title": "Stela",
        "object_id": "20.2.6",
        "object_url": "https://www.metmuseum.org/art/collection/search/545752",
        "license": "Public Domain",
        "source_type": "relief",
        "note": "manually cropped from The Met stela 20.2.6"
    },
    "MET98468": {
        "source_museum": "The Met",
        "object_title": "Stela",
        "object_id": "98.4.68",
        "object_url": "https://www.metmuseum.org/art/collection/search/571335",
        "license": "Public Domain",
        "source_type": "relief",
        "note": "manually cropped from The Met stela 98.4.68"
    },
    "MET543863": {
        "source_museum": "The Met",
        "object_title": "False Door of the Royal Sealer Neferiu",
        "object_id": "543863",
        "object_url": "https://www.metmuseum.org/art/collection/search/543863",
        "license": "Public Domain",
        "source_type": "false door relief",
        "note": "manually cropped from The Met false door of Neferiu"
    },
    "MET19944964": {
        "source_museum": "The Met",
        "object_title": "Relief of hieroglyphic inscription",
        "object_id": "1994.496.4",
        "object_url": "https://www.metmuseum.org/art/collection/search/555650",
        "license": "Public Domain",
        "source_type": "relief",
        "note": "manually cropped from The Met relief of hieroglyphic inscription"
    },
    "DP226728": {
        "source_museum": "Unknown / downloaded image",
        "object_title": "Unknown",
        "object_id": "DP226728",
        "object_url": "Unknown",
        "license": "Unknown",
        "source_type": "relief",
        "note": "manually cropped external image; source metadata to be checked"
    },
    "DP322047": {
        "source_museum": "Unknown / downloaded image",
        "object_title": "Unknown",
        "object_id": "DP322047",
        "object_url": "Unknown",
        "license": "Unknown",
        "source_type": "relief",
        "note": "manually cropped external image; source metadata to be checked"
    },
    "DP261896": {
        "source_museum": "Unknown / downloaded image",
        "object_title": "Unknown",
        "object_id": "DP261896",
        "object_url": "Unknown",
        "license": "Unknown",
        "source_type": "relief",
        "note": "manually cropped external image; source metadata to be checked"
    },
    "REAL": {
        "source_museum": "Unknown",
        "object_title": "Unknown",
        "object_id": "Unknown",
        "object_url": "Unknown",
        "license": "Unknown",
        "source_type": "relief",
        "note": "manually cropped external real image"
    }
}


# =========================================================
# 5. 工具函数
# =========================================================

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def should_skip_image(path: Path) -> bool:
    """
    跳过原图文件夹中的大图，只保留裁剪图。
    """
    parts_lower = [p.lower() for p in path.parts]

    skip_keywords = [
        "original",
        "原图",
        "met_neferiu_original",
        "met_548125_original",
        "met_20_2_6_original",
        "met_98_4_68_original"
    ]

    return any(keyword in parts_lower for keyword in skip_keywords)


def parse_filename(file_path: Path):
    """
    从文件名中解析：
    image_id, source_prefix, gardiner_code

    支持：
    MET2026_D4_001.png
    DP226728_G17_001.png
    MET19944964_G17_001.png
    REAL_N35_001.png
    """

    stem = file_path.stem.strip()
    parts = stem.split("_")

    if len(parts) < 3:
        return {
            "image_id": stem,
            "source_prefix": "UNKNOWN",
            "gardiner_code": "UNKNOWN"
        }

    source_prefix = parts[0].upper()
    gardiner_code = parts[1].upper()

    # Gardiner 格式：D4 / N35 / AA15 / S14A
    if not re.match(r"^[A-Z]{1,2}\d+[A-Z]?$", gardiner_code):
        gardiner_code = "UNKNOWN"

    return {
        "image_id": stem,
        "source_prefix": source_prefix,
        "gardiner_code": gardiner_code
    }


def get_source_info(source_prefix: str):
    return SOURCE_INFO_MAP.get(source_prefix, {
        "source_museum": "Unknown",
        "object_title": "Unknown",
        "object_id": source_prefix,
        "object_url": "Unknown",
        "license": "Unknown",
        "source_type": "unknown",
        "note": "source prefix not recognized; metadata should be checked"
    })


def find_all_cropped_images():
    """
    递归扫描 real_external_test 下所有图片。
    """
    image_files = []

    for file in REAL_TEST_DIR.rglob("*"):
        if not is_image_file(file):
            continue

        if should_skip_image(file):
            continue

        # 避免把输出图、预览图之类误扫进去
        if file.name.lower().startswith("preview"):
            continue

        image_files.append(file)

    return sorted(image_files)


def build_label_records():
    image_files = find_all_cropped_images()

    records = []

    for file_path in image_files:
        parsed = parse_filename(file_path)

        image_id = parsed["image_id"]
        source_prefix = parsed["source_prefix"]
        gardiner_code = parsed["gardiner_code"]

        source_info = get_source_info(source_prefix)
        true_name = GARDINER_NAME_MAP.get(gardiner_code, "unknown")

        relative_path = file_path.relative_to(REAL_TEST_DIR)

        record = {
            "image_id": image_id,
            "image_path": str(relative_path).replace("\\", "/"),
            "true_gardiner_code": gardiner_code,
            "true_name": true_name,
            "source_museum": source_info["source_museum"],
            "object_title": source_info["object_title"],
            "object_id": source_info["object_id"],
            "object_url": source_info["object_url"],
            "license": source_info["license"],
            "source_type": source_info["source_type"],
            "note": source_info["note"]
        }

        records.append(record)

    return records


# =========================================================
# 6. 主流程
# =========================================================

def main():
    print("=" * 90)
    print("开始创建真实外部测试集标签 CSV")
    print("=" * 90)

    print(f"真实外部测试集目录：{REAL_TEST_DIR}")
    print(f"输出 CSV：{OUTPUT_CSV}")

    image_files = find_all_cropped_images()

    print(f"\n扫描到图片数量：{len(image_files)}")

    if image_files:
        print("\n扫描到的图片前 30 个：")
        for file in image_files[:30]:
            print("-", file.relative_to(REAL_TEST_DIR))

    records = build_label_records()

    if not records:
        print("\n没有找到任何裁剪图片。")
        print("请确认图片是否放在 real_external_test 的子文件夹中。")
        return

    df = pd.DataFrame(records)

    df = df.sort_values(
        by=["true_gardiner_code", "image_id"],
        ascending=[True, True]
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n标签 CSV 创建完成。")
    print(f"图片数量：{len(df)}")
    print(f"输出文件：{OUTPUT_CSV}")

    print("\n类别分布：")
    class_counts = (
        df["true_gardiner_code"]
        .value_counts()
        .reset_index()
    )
    class_counts.columns = ["true_gardiner_code", "count"]
    print(class_counts.to_string(index=False))

    print("\n来源分布：")
    source_counts = (
        df["object_id"]
        .value_counts()
        .reset_index()
    )
    source_counts.columns = ["object_id", "count"]
    print(source_counts.to_string(index=False))

    print("\n前 30 行预览：")
    print(df.head(30).to_string(index=False))

    print("\n下一步：")
    print("1. 打开 real_external_test_labels.csv 检查 UNKNOWN 是否合理；")
    print("2. 如果某些来源是 Unknown，后续补充 SOURCE_INFO_MAP；")
    print("3. 确认无误后，运行 predict_real_external_resnet18_v3.py。")


if __name__ == "__main__":
    main()