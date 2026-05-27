from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"
FONT_DIR = SIGN_DIR / "fonts"
SIGN_PNG_DIR = SIGN_DIR / "sign_png"
SIGN_PROCESSED_DIR = SIGN_DIR / "sign_processed"
CSV_PATH = SIGN_DIR / "hieroglyph_signs.csv"

FONT_PATH = FONT_DIR / "NotoSansEgyptianHieroglyphs-Regular.ttf"

IMG_SIZE = 256
FONT_SIZE = 160


# =========================================================
# 2. 标准符号库：基础 20 个 + 第一批扩展 14 个 = 34 个
# =========================================================
SIGNS = [
    # -------------------------
    # 原始 20 个基础符号
    # -------------------------
    {
        "gardiner_code": "N5",
        "unicode_char": "𓇳",
        "zh_name": "太阳",
        "en_name": "sun",
        "related_terms": "sun,solar,ra,re,raw"
    },
    {
        "gardiner_code": "S34",
        "unicode_char": "𓋹",
        "zh_name": "生命符号",
        "en_name": "ankh",
        "related_terms": "life,ankh,anh"
    },
    {
        "gardiner_code": "G17",
        "unicode_char": "𓅓",
        "zh_name": "猫头鹰",
        "en_name": "owl",
        "related_terms": "owl,m"
    },
    {
        "gardiner_code": "D21",
        "unicode_char": "𓂋",
        "zh_name": "口",
        "en_name": "mouth",
        "related_terms": "mouth,r"
    },
    {
        "gardiner_code": "M17",
        "unicode_char": "𓇋",
        "zh_name": "芦苇叶",
        "en_name": "reed leaf",
        "related_terms": "reed,i"
    },
    {
        "gardiner_code": "X1",
        "unicode_char": "𓏏",
        "zh_name": "面包",
        "en_name": "bread loaf",
        "related_terms": "bread,t"
    },
    {
        "gardiner_code": "O1",
        "unicode_char": "𓉐",
        "zh_name": "房屋",
        "en_name": "house",
        "related_terms": "house,pr"
    },
    {
        "gardiner_code": "O6",
        "unicode_char": "𓉗",
        "zh_name": "神庙围墙",
        "en_name": "temple enclosure",
        "related_terms": "temple,hwt-ntr"
    },
    {
        "gardiner_code": "A1",
        "unicode_char": "𓀀",
        "zh_name": "坐着的人",
        "en_name": "seated man",
        "related_terms": "man,person"
    },
    {
        "gardiner_code": "B1",
        "unicode_char": "𓁐",
        "zh_name": "女人",
        "en_name": "woman",
        "related_terms": "woman,female"
    },
    {
        "gardiner_code": "G1",
        "unicode_char": "𓄿",
        "zh_name": "秃鹫",
        "en_name": "vulture",
        "related_terms": "vulture,a"
    },
    {
        "gardiner_code": "I10",
        "unicode_char": "𓆓",
        "zh_name": "眼镜蛇",
        "en_name": "cobra",
        "related_terms": "cobra,dj"
    },
    {
        "gardiner_code": "R4",
        "unicode_char": "𓊵",
        "zh_name": "供品台",
        "en_name": "offering table",
        "related_terms": "offering,htp"
    },
    {
        "gardiner_code": "V28",
        "unicode_char": "𓎛",
        "zh_name": "绳圈",
        "en_name": "wick",
        "related_terms": "wick,h"
    },
    {
        "gardiner_code": "D36",
        "unicode_char": "𓂝",
        "zh_name": "手臂",
        "en_name": "arm",
        "related_terms": "arm,a"
    },
    {
        "gardiner_code": "F35",
        "unicode_char": "𓄤",
        "zh_name": "心与气管",
        "en_name": "heart and windpipe",
        "related_terms": "heart,nfr,good,beautiful"
    },
    {
        "gardiner_code": "Y5",
        "unicode_char": "𓏞",
        "zh_name": "书写板",
        "en_name": "writing board",
        "related_terms": "scribe,writing"
    },
    {
        "gardiner_code": "M23",
        "unicode_char": "𓇓",
        "zh_name": "莎草植物",
        "en_name": "sedge",
        "related_terms": "sedge,king,nswt"
    },
    {
        "gardiner_code": "N35",
        "unicode_char": "𓈖",
        "zh_name": "水波",
        "en_name": "water",
        "related_terms": "water,n"
    },
    {
        "gardiner_code": "Z1",
        "unicode_char": "𓏤",
        "zh_name": "单笔画",
        "en_name": "stroke",
        "related_terms": "stroke,one"
    },

    # -------------------------
    # 第一批扩展符号：新增 14 个
    # -------------------------
    {
        "gardiner_code": "D4",
        "unicode_char": "𓁹",
        "zh_name": "眼睛",
        "en_name": "eye",
        "related_terms": "eye,seeing,vision,watch"
    },
    {
        "gardiner_code": "D10",
        "unicode_char": "𓂀",
        "zh_name": "荷鲁斯之眼",
        "en_name": "wedjat eye",
        "related_terms": "wedjat,horus,eye,protection,healing"
    },
    {
        "gardiner_code": "G43",
        "unicode_char": "𓅱",
        "zh_name": "鹌鹑雏鸟",
        "en_name": "quail chick",
        "related_terms": "quail,bird,w,chick"
    },
    {
        "gardiner_code": "I9",
        "unicode_char": "𓆑",
        "zh_name": "角蝰",
        "en_name": "horned viper",
        "related_terms": "viper,snake,f,danger"
    },
    {
        "gardiner_code": "N1",
        "unicode_char": "𓇯",
        "zh_name": "天空",
        "en_name": "sky",
        "related_terms": "sky,heaven,firmament"
    },
    {
        "gardiner_code": "N14",
        "unicode_char": "𓇼",
        "zh_name": "星星",
        "en_name": "star",
        "related_terms": "star,night,sky,astral"
    },
    {
        "gardiner_code": "N16",
        "unicode_char": "𓇾",
        "zh_name": "土地",
        "en_name": "land",
        "related_terms": "land,earth,ground,territory"
    },
    {
        "gardiner_code": "N25",
        "unicode_char": "𓈉",
        "zh_name": "山地",
        "en_name": "hill country",
        "related_terms": "hill,mountain,foreign land,desert"
    },
    {
        "gardiner_code": "N36",
        "unicode_char": "𓈘",
        "zh_name": "水渠",
        "en_name": "water channel",
        "related_terms": "canal,water,channel,irrigation"
    },
    {
        "gardiner_code": "R8",
        "unicode_char": "𓊹",
        "zh_name": "神旗",
        "en_name": "god standard",
        "related_terms": "god,deity,divine,standard,temple"
    },
    {
        "gardiner_code": "S29",
        "unicode_char": "𓋴",
        "zh_name": "折布",
        "en_name": "folded cloth",
        "related_terms": "cloth,fabric,s"
    },
    {
        "gardiner_code": "V31",
        "unicode_char": "𓎡",
        "zh_name": "篮子",
        "en_name": "basket",
        "related_terms": "basket,container,k"
    },
    {
        "gardiner_code": "Y1",
        "unicode_char": "𓏛",
        "zh_name": "书卷",
        "en_name": "scroll",
        "related_terms": "scroll,writing,document,text"
    },
    {
        "gardiner_code": "Z2",
        "unicode_char": "𓏥",
        "zh_name": "双短划",
        "en_name": "double strokes",
        "related_terms": "two,dual,plural,stroke"
    },
]


# =========================================================
# 3. 创建文件夹
# =========================================================
SIGN_DIR.mkdir(parents=True, exist_ok=True)
FONT_DIR.mkdir(parents=True, exist_ok=True)
SIGN_PNG_DIR.mkdir(parents=True, exist_ok=True)
SIGN_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 4. 检查符号编号是否重复
# =========================================================
def check_duplicate_gardiner_codes(signs):
    codes = [item["gardiner_code"] for item in signs]
    duplicated = sorted(set([code for code in codes if codes.count(code) > 1]))

    if duplicated:
        raise ValueError(f"发现重复 Gardiner 编号：{duplicated}")

    print("Gardiner 编号检查通过：无重复。")


# =========================================================
# 5. 检查字体
# =========================================================
if not FONT_PATH.exists():
    raise FileNotFoundError(
        f"未找到字体文件：{FONT_PATH}\n"
        f"请确认字体文件名是否为 NotoSansEgyptianHieroglyphs-Regular.ttf"
    )

font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)


# =========================================================
# 6. 渲染单个符号图像
# =========================================================
def render_sign(char: str, save_path: Path):
    """
    将单个古埃及 Unicode 字符渲染为 PNG 图片。
    """
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (IMG_SIZE - text_w) // 2 - bbox[0]
    y = (IMG_SIZE - text_h) // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)
    img.save(save_path)


# =========================================================
# 7. 生成图像库和元数据表
# =========================================================
def build_sign_library():
    check_duplicate_gardiner_codes(SIGNS)

    records = []

    for sign in SIGNS:
        gardiner_code = sign["gardiner_code"]
        unicode_char = sign["unicode_char"]
        zh_name = sign["zh_name"]
        en_name = sign["en_name"]
        related_terms = sign["related_terms"]

        file_name = f"{gardiner_code}.png"
        img_path = SIGN_PNG_DIR / file_name

        render_sign(unicode_char, img_path)

        records.append({
            "gardiner_code": gardiner_code,
            "unicode_char": unicode_char,
            "zh_name": zh_name,
            "en_name": en_name,
            "related_terms": related_terms,
            "png_path": str(img_path)
        })

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("古埃及标准符号库生成完成！")
    print("=" * 60)
    print(f"字体文件：{FONT_PATH}")
    print(f"符号数量：{len(df)}")
    print(f"元数据表：{CSV_PATH}")
    print(f"PNG 输出文件夹：{SIGN_PNG_DIR}")

    print("\n前 15 条预览：")
    print(df.head(15))


# =========================================================
# 8. 主程序
# =========================================================
if __name__ == "__main__":
    build_sign_library()