from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"
SIGN_PNG_DIR = SIGN_DIR / "sign_png"
SIGN_PROCESSED_DIR = SIGN_DIR / "sign_processed"

CSV_PATH = SIGN_DIR / "hieroglyph_signs.csv"
OUTPUT_CSV_PATH = SIGN_DIR / "hieroglyph_signs_processed.csv"

SIGN_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SIZE = 128


# =========================
# 2. 图像预处理函数
# =========================
def preprocess_sign_image(image_path: Path, output_size: int = 128):
    """
    对单个古埃及符号图像进行标准化处理：
    1. 灰度读取
    2. 二值化
    3. 找到符号区域
    4. 裁剪并居中
    5. 统一缩放到 output_size × output_size
    """

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    # 反色：符号区域变白，背景变黑，方便找轮廓
    _, binary_inv = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 找非零区域
    coords = cv2.findNonZero(binary_inv)

    if coords is None:
        # 如果没有找到符号，返回空白图
        return np.zeros((output_size, output_size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)

    cropped = binary_inv[y:y + h, x:x + w]

    # 加一点边距
    pad = 20
    cropped = cv2.copyMakeBorder(
        cropped,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # 保持比例缩放
    h2, w2 = cropped.shape
    scale = (output_size - 20) / max(h2, w2)
    new_w = max(1, int(w2 * scale))
    new_h = max(1, int(h2 * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # 居中到 output_size × output_size
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    start_x = (output_size - new_w) // 2
    start_y = (output_size - new_h) // 2

    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return canvas


# =========================
# 3. 批量处理符号库
# =========================
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"未找到符号元数据表：{CSV_PATH}")

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    processed_paths = []

    print("=" * 60)
    print("开始处理标准古埃及符号图像库")
    print("=" * 60)

    for idx, row in df.iterrows():
        gardiner_code = row["gardiner_code"]
        png_path = Path(row["png_path"])

        if not png_path.exists():
            print(f"跳过，图片不存在：{png_path}")
            processed_paths.append("")
            continue

        processed_img = preprocess_sign_image(png_path, OUTPUT_SIZE)

        output_path = SIGN_PROCESSED_DIR / f"{gardiner_code}_processed.png"
        cv2.imwrite(str(output_path), processed_img)

        processed_paths.append(str(output_path))

        print(f"{idx + 1:02d}. {gardiner_code} 处理完成：{output_path.name}")

    df["processed_png_path"] = processed_paths
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("标准符号图像预处理完成！")
    print("=" * 60)
    print("处理后符号数量：", len(df))
    print("输出文件夹：", SIGN_PROCESSED_DIR)
    print("处理后元数据表：", OUTPUT_CSV_PATH)

    print("\n前 10 条预览：")
    print(df.head(10))


if __name__ == "__main__":
    main()