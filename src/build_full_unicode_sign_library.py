from pathlib import Path
import unicodedata
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"
FONT_DIR = SIGN_DIR / "fonts"

FULL_SIGN_PNG_DIR = SIGN_DIR / "sign_png_full"
FULL_SIGN_PNG_DIR.mkdir(parents=True, exist_ok=True)

FULL_UNICODE_SIGNS_CSV = SIGN_DIR / "full_unicode_signs.csv"
MANUAL_ANNOTATION_CSV = SIGN_DIR / "sign_annotations_manual.csv"
MERGED_LIBRARY_CSV = SIGN_DIR / "sign_library_merged.csv"

OLD_SIGN_CSV = SIGN_DIR / "hieroglyph_signs.csv"

FONT_PATH = FONT_DIR / "NotoSansEgyptianHieroglyphs-Regular.ttf"

IMG_SIZE = 256
FONT_SIZE = 160


# =========================================================
# 2. Unicode 古埃及文字区段设置
# =========================================================
# 基础 Egyptian Hieroglyphs: U+13000–U+1342F
# Format Controls: U+13430–U+1345F，属于格式控制符，不作为图像符号生成
# Extended-A: U+13460–U+143FF，取决于字体是否支持
UNICODE_RANGES = [
    {
        "block_name": "Egyptian Hieroglyphs",
        "start": 0x13000,
        "end": 0x1342F,
        "include": True
    },
    {
        "block_name": "Egyptian Hieroglyph Format Controls",
        "start": 0x13430,
        "end": 0x1345F,
        "include": False
    },
    {
        "block_name": "Egyptian Hieroglyphs Extended-A",
        "start": 0x13460,
        "end": 0x143FF,
        "include": True
    }
]


# =========================================================
# 3. 字体支持字符检查
# =========================================================
def load_supported_codepoints(font_path: Path):
    """
    使用 fontTools 读取字体 cmap，获取该字体实际支持的 Unicode codepoints。
    这样可以避免把字体不能显示的字符也加入符号库。
    """
    if not font_path.exists():
        raise FileNotFoundError(
            f"未找到字体文件：{font_path}\n"
            "请确认 NotoSansEgyptianHieroglyphs-Regular.ttf 已放入 data_sign_demo/fonts/"
        )

    font = TTFont(str(font_path))
    supported = set()

    for table in font["cmap"].tables:
        supported.update(table.cmap.keys())

    return supported


def get_unicode_name(codepoint: int):
    """
    获取 Unicode 字符名。
    如果当前 Python Unicode 数据库不认识该字符，则返回空字符串。
    """
    char = chr(codepoint)

    try:
        return unicodedata.name(char)
    except ValueError:
        return ""


def get_auto_label(codepoint: int, unicode_name: str):
    """
    自动生成标签。
    基础区段通常有 EGYPTIAN HIEROGLYPH A001 这样的名称。
    如果 Python 无法识别名称，就用 U+xxxxx 作为标签。
    """
    if unicode_name:
        prefix = "EGYPTIAN HIEROGLYPH "
        if unicode_name.startswith(prefix):
            return unicode_name.replace(prefix, "").strip()

        return unicode_name.replace("EGYPTIAN HIEROGLYPH", "").strip()

    return f"U+{codepoint:05X}"


def get_block_name(codepoint: int):
    for item in UNICODE_RANGES:
        if item["start"] <= codepoint <= item["end"]:
            return item["block_name"]

    return "Unknown"


# =========================================================
# 4. 渲染图片
# =========================================================
def render_sign_to_png(char: str, font: ImageFont.FreeTypeFont, save_path: Path):
    """
    将单个 Unicode 古埃及字符渲染为 PNG。
    """
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (IMG_SIZE - text_w) // 2 - bbox[0]
    y = (IMG_SIZE - text_h) // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)

    # 检查是否真的有黑色像素，避免空白图
    extrema = img.getextrema()
    if extrema == (255, 255):
        return False

    img.save(save_path)
    return True


# =========================================================
# 5. 从旧的 34 符号库生成人工注释表
# =========================================================
def build_manual_annotation_seed():
    """
    如果已有 hieroglyph_signs.csv，就将其中的中文名、英文名、related_terms
    转换成 sign_annotations_manual.csv，作为人工增强层的初始版本。
    """
    if not OLD_SIGN_CSV.exists():
        print("未找到旧版 hieroglyph_signs.csv，跳过人工注释种子生成。")
        return pd.DataFrame()

    old_df = pd.read_csv(OLD_SIGN_CSV, dtype=str).fillna("")

    records = []

    for _, row in old_df.iterrows():
        unicode_char = row.get("unicode_char", "")
        if not unicode_char:
            continue

        codepoint = ord(unicode_char)
        unicode_codepoint = f"U+{codepoint:05X}"

        records.append({
            "unicode_codepoint": unicode_codepoint,
            "unicode_char": unicode_char,
            "gardiner_code": row.get("gardiner_code", ""),
            "zh_name": row.get("zh_name", ""),
            "en_name": row.get("en_name", ""),
            "phonetic_value": "",
            "meaning_zh": row.get("zh_name", ""),
            "meaning_en": row.get("en_name", ""),
            "related_terms": row.get("related_terms", ""),
            "text_search_terms": row.get("related_terms", ""),
            "category": "",
            "annotation_source": "manual_seed_from_existing_library"
        })

    manual_df = pd.DataFrame(records)

    if len(manual_df) > 0:
        manual_df = manual_df.drop_duplicates(subset=["unicode_codepoint"])
        manual_df.to_csv(MANUAL_ANNOTATION_CSV, index=False, encoding="utf-8-sig")
        print(f"人工注释种子表已生成：{MANUAL_ANNOTATION_CSV}")
        print(f"人工注释记录数：{len(manual_df)}")

    return manual_df


# =========================================================
# 6. 构建全量 Unicode 符号库
# =========================================================
def build_full_unicode_sign_library():
    supported_codepoints = load_supported_codepoints(FONT_PATH)
    pil_font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)

    records = []

    print("=" * 80)
    print("开始构建全量 Unicode 古埃及符号库")
    print("=" * 80)
    print("字体文件：", FONT_PATH)
    print("字体支持 codepoint 数量：", len(supported_codepoints))

    for range_item in UNICODE_RANGES:
        block_name = range_item["block_name"]
        start = range_item["start"]
        end = range_item["end"]
        include = range_item["include"]

        print("\n" + "-" * 80)
        print(f"扫描区段：{block_name} U+{start:05X}–U+{end:05X}")

        if not include:
            print("该区段为格式控制符，跳过图像生成。")
            continue

        block_count = 0

        for codepoint in range(start, end + 1):
            if codepoint not in supported_codepoints:
                continue

            char = chr(codepoint)
            unicode_name = get_unicode_name(codepoint)

            # 基础区段一般有 Unicode name；Extended-A 在旧版 Python 里可能没有 name
            auto_label = get_auto_label(codepoint, unicode_name)
            unicode_codepoint = f"U+{codepoint:05X}"

            sign_id = unicode_codepoint.replace("+", "_")
            file_name = f"{sign_id}.png"
            png_path = FULL_SIGN_PNG_DIR / file_name

            rendered = render_sign_to_png(char, pil_font, png_path)

            if not rendered:
                continue

            records.append({
                "sign_id": sign_id,
                "unicode_codepoint": unicode_codepoint,
                "codepoint_decimal": codepoint,
                "unicode_char": char,
                "unicode_name": unicode_name,
                "unicode_block": block_name,
                "auto_label": auto_label,
                "png_path": str(png_path),
                "has_manual_annotation": False
            })

            block_count += 1

            if len(records) % 200 == 0:
                print(f"已生成 {len(records)} 个符号 PNG...")

        print(f"{block_name} 生成数量：{block_count}")

    full_df = pd.DataFrame(records)

    full_df.to_csv(FULL_UNICODE_SIGNS_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("全量 Unicode 古埃及符号库生成完成！")
    print("=" * 80)
    print("符号总数：", len(full_df))
    print("输出 CSV：", FULL_UNICODE_SIGNS_CSV)
    print("PNG 文件夹：", FULL_SIGN_PNG_DIR)

    print("\n各区段数量：")
    print(full_df["unicode_block"].value_counts())

    print("\n前 10 条预览：")
    print(full_df.head(10))

    return full_df


# =========================================================
# 7. 合并自动层与人工注释层
# =========================================================
def build_merged_library(full_df: pd.DataFrame, manual_df: pd.DataFrame):
    """
    合并 full_unicode_signs.csv 和 sign_annotations_manual.csv。
    自动层负责全量字符，人工层负责中文名、Gardiner 编号、检索词等。
    """
    if manual_df is None or len(manual_df) == 0:
        full_df.to_csv(MERGED_LIBRARY_CSV, index=False, encoding="utf-8-sig")
        print("无人工注释表，已直接复制 full_unicode_signs 为 sign_library_merged.csv")
        return full_df

    merged = full_df.merge(
        manual_df,
        on=["unicode_codepoint", "unicode_char"],
        how="left"
    )

    manual_cols = [
        "gardiner_code",
        "zh_name",
        "en_name",
        "phonetic_value",
        "meaning_zh",
        "meaning_en",
        "related_terms",
        "text_search_terms",
        "category",
        "annotation_source"
    ]

    for col in manual_cols:
        if col not in merged.columns:
            merged[col] = ""

    merged[manual_cols] = merged[manual_cols].fillna("")

    merged["has_manual_annotation"] = merged["gardiner_code"].apply(
        lambda x: True if isinstance(x, str) and x.strip() else False
    )

    merged.to_csv(MERGED_LIBRARY_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("自动符号库与人工注释层合并完成！")
    print("=" * 80)
    print("合并后文件：", MERGED_LIBRARY_CSV)
    print("合并后记录数：", len(merged))
    print("已有人工注释数量：", merged["has_manual_annotation"].sum())

    return merged


# =========================================================
# 8. 主程序
# =========================================================
def main():
    full_df = build_full_unicode_sign_library()
    manual_df = build_manual_annotation_seed()
    build_merged_library(full_df, manual_df)


if __name__ == "__main__":
    main()