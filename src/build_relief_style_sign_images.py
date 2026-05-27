from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image


# =========================================================
# 1. 路径设置
# =========================================================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

# 输入：全量符号库
FULL_SIGN_CSV = SIGN_DIR / "sign_library_merged_processed.csv"

# 输出：浮雕风格增强图像
RELIEF_DIR = SIGN_DIR / "sign_relief_augmented"
RELIEF_DIR.mkdir(parents=True, exist_ok=True)

# 输出：浮雕增强元数据
RELIEF_METADATA_CSV = SIGN_DIR / "sign_relief_augmented_metadata.csv"

# 图像尺寸
OUTPUT_SIZE = 224

# 每个符号生成多少种浮雕风格
RELIEF_VARIANTS = [
    "relief_light_left",
    "relief_light_right",
    "relief_light_top",
    "relief_soft_shadow",
    "relief_strong_shadow",
    "relief_noise",
]


# =========================================================
# 2. 支持中文路径读取图片
# =========================================================
def read_image_gray(image_path: Path):
    """
    支持 Windows 中文路径读取灰度图。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return None

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    return img


def save_image_rgb(image: np.ndarray, save_path: Path):
    """
    支持中文路径保存 RGB 图像。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded_img = cv2.imencode(".png", image_rgb)

    if success:
        encoded_img.tofile(str(save_path))
        return True

    return False


# =========================================================
# 3. 基础图像标准化
# =========================================================
def ensure_size(img: np.ndarray, size: int = 224):
    """
    将输入图像统一调整到 size × size。
    """
    if img is None:
        return None

    if img.shape != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    return img


def prepare_symbol_mask(img_gray: np.ndarray, size: int = 224):
    """
    将标准符号图转换为符号 mask：
    - 输入通常是白底黑字或黑底白字
    - 输出为 0-1 浮点 mask，符号区域为 1，背景为 0
    """
    img_gray = ensure_size(img_gray, size)

    # 判断背景亮度，自动决定是否反色
    mean_val = img_gray.mean()

    if mean_val > 127:
        # 白底黑字：反色后符号为白
        _, mask = cv2.threshold(
            img_gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        # 黑底白字：直接阈值
        _, mask = cv2.threshold(
            img_gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # 轻微闭运算，让断裂线条连接
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = mask.astype("float32") / 255.0

    return mask


# =========================================================
# 4. 生成石壁背景
# =========================================================
def generate_stone_background(size: int = 224, variant: str = "default", seed: int = 0):
    """
    生成米黄色石壁背景。
    """
    rng = np.random.default_rng(seed)

    # 基础石壁颜色，RGB
    base_color = np.array([185, 155, 115], dtype=np.float32)

    if "strong" in variant:
        base_color = np.array([170, 135, 95], dtype=np.float32)

    if "soft" in variant:
        base_color = np.array([198, 170, 130], dtype=np.float32)

    bg = np.ones((size, size, 3), dtype=np.float32) * base_color

    # 大尺度纹理
    noise = rng.normal(0, 18, (size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=8, sigmaY=8)

    # 小尺度颗粒
    fine_noise = rng.normal(0, 6, (size, size)).astype(np.float32)
    fine_noise = cv2.GaussianBlur(fine_noise, (3, 3), 0)

    texture = noise + fine_noise

    for c in range(3):
        bg[:, :, c] += texture

    # 加一些轻微横向/斜向划痕
    scratch_layer = np.zeros((size, size), dtype=np.float32)

    scratch_count = 18 if "noise" in variant else 10

    for _ in range(scratch_count):
        x1 = int(rng.integers(0, size))
        y1 = int(rng.integers(0, size))
        length = int(rng.integers(size // 8, size // 3))
        angle = rng.uniform(-0.8, 0.8)

        x2 = int(np.clip(x1 + length * np.cos(angle), 0, size - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, size - 1))

        color = float(rng.uniform(8, 22))
        thickness = int(rng.integers(1, 2))

        cv2.line(
            scratch_layer,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    scratch_layer = cv2.GaussianBlur(scratch_layer, (3, 3), 0)

    for c in range(3):
        bg[:, :, c] -= scratch_layer

    bg = np.clip(bg, 0, 255).astype(np.uint8)

    return bg


# =========================================================
# 5. 浮雕光照与阴影
# =========================================================
def create_shifted_mask(mask: np.ndarray, dx: int, dy: int):
    """
    将 mask 平移，用于生成高光和阴影。
    """
    h, w = mask.shape

    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    shifted = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return shifted


def get_light_shadow_params(variant: str):
    """
    根据不同浮雕风格设置高光和阴影方向。
    """
    if variant == "relief_light_left":
        return {
            "highlight_shift": (-3, -3),
            "shadow_shift": (4, 4),
            "highlight_strength": 55,
            "shadow_strength": 55
        }

    if variant == "relief_light_right":
        return {
            "highlight_shift": (3, -3),
            "shadow_shift": (-4, 4),
            "highlight_strength": 55,
            "shadow_strength": 55
        }

    if variant == "relief_light_top":
        return {
            "highlight_shift": (0, -4),
            "shadow_shift": (0, 5),
            "highlight_strength": 60,
            "shadow_strength": 50
        }

    if variant == "relief_soft_shadow":
        return {
            "highlight_shift": (-2, -2),
            "shadow_shift": (3, 3),
            "highlight_strength": 35,
            "shadow_strength": 35
        }

    if variant == "relief_strong_shadow":
        return {
            "highlight_shift": (-4, -4),
            "shadow_shift": (5, 5),
            "highlight_strength": 75,
            "shadow_strength": 80
        }

    if variant == "relief_noise":
        return {
            "highlight_shift": (-3, -2),
            "shadow_shift": (4, 3),
            "highlight_strength": 50,
            "shadow_strength": 60
        }

    return {
        "highlight_shift": (-3, -3),
        "shadow_shift": (4, 4),
        "highlight_strength": 50,
        "shadow_strength": 50
    }


# =========================================================
# 6. 生成浮雕风格图
# =========================================================
def generate_relief_image(mask: np.ndarray, variant: str, seed: int = 0):
    """
    将符号 mask 渲染为浮雕风格图像。
    """
    size = mask.shape[0]

    bg = generate_stone_background(size=size, variant=variant, seed=seed).astype(np.float32)

    params = get_light_shadow_params(variant)

    hdx, hdy = params["highlight_shift"]
    sdx, sdy = params["shadow_shift"]

    highlight_mask = create_shifted_mask(mask, hdx, hdy)
    shadow_mask = create_shifted_mask(mask, sdx, sdy)

    # 原始符号主体稍微压暗，模拟刻痕/凹凸边缘
    body_mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # 边缘区域
    edge = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    edge = cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)
    edge = edge.astype("float32") / 255.0
    edge = cv2.GaussianBlur(edge, (3, 3), 0)

    image = bg.copy()

    # 阴影：偏暗
    shadow_strength = params["shadow_strength"]
    for c in range(3):
        image[:, :, c] -= shadow_mask * shadow_strength

    # 高光：偏亮
    highlight_strength = params["highlight_strength"]
    for c in range(3):
        image[:, :, c] += highlight_mask * highlight_strength

    # 主体边缘微暗
    for c in range(3):
        image[:, :, c] -= body_mask * 18

    # 边缘强化
    for c in range(3):
        image[:, :, c] -= edge * 20

    # 轻微整体模糊，模拟照片
    if variant in ["relief_soft_shadow", "relief_noise"]:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # 添加轻微噪声
    rng = np.random.default_rng(seed + 1000)
    photo_noise = rng.normal(0, 4, image.shape).astype(np.float32)

    if variant == "relief_noise":
        photo_noise = rng.normal(0, 8, image.shape).astype(np.float32)

    image += photo_noise

    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# =========================================================
# 7. 主流程
# =========================================================
def main():
    if not FULL_SIGN_CSV.exists():
        raise FileNotFoundError(f"未找到全量符号库文件：{FULL_SIGN_CSV}")

    df = pd.read_csv(FULL_SIGN_CSV, dtype=str).fillna("")

    print("=" * 90)
    print("开始生成全量古埃及符号浮雕风格增强图")
    print("=" * 90)
    print("输入符号库：", FULL_SIGN_CSV)
    print("输入符号数量：", len(df))
    print("输出文件夹：", RELIEF_DIR)
    print("每个符号生成风格数量：", len(RELIEF_VARIANTS))
    print("=" * 90)

    records = []
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        sign_id = row.get("sign_id", "")
        unicode_codepoint = row.get("unicode_codepoint", "")
        png_path = row.get("png_path", "")

        if not sign_id or not png_path:
            fail_count += 1
            continue

        img_gray = read_image_gray(Path(png_path))

        if img_gray is None:
            fail_count += 1
            continue

        mask = prepare_symbol_mask(img_gray, OUTPUT_SIZE)

        for variant_idx, variant in enumerate(RELIEF_VARIANTS):
            seed = idx * 100 + variant_idx

            relief_img = generate_relief_image(
                mask=mask,
                variant=variant,
                seed=seed
            )

            output_name = f"{sign_id}_{variant}.png"
            output_path = RELIEF_DIR / output_name

            ok = save_image_rgb(relief_img, output_path)

            if not ok:
                fail_count += 1
                continue

            new_row = row.to_dict()
            new_row["relief_variant"] = variant
            new_row["relief_png_path"] = str(output_path)
            new_row["relief_generation_method"] = "synthetic_stone_relief_v1"

            records.append(new_row)

        success_count += 1

        if (idx + 1) % 100 == 0:
            print(
                f"已处理 {idx + 1} / {len(df)} 个符号，"
                f"已生成浮雕图 {len(records)} 张"
            )

    relief_df = pd.DataFrame(records)
    relief_df.to_csv(RELIEF_METADATA_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("全量浮雕风格增强图生成完成！")
    print("=" * 90)
    print("成功处理符号数量：", success_count)
    print("失败数量：", fail_count)
    print("浮雕增强图片数量：", len(relief_df))
    print("输出元数据：", RELIEF_METADATA_CSV)
    print("输出文件夹：", RELIEF_DIR)

    print("\n浮雕风格统计：")
    if len(relief_df) > 0:
        print(relief_df["relief_variant"].value_counts())

    print("\n前 10 条预览：")
    print(relief_df.head(10))


if __name__ == "__main__":
    main()