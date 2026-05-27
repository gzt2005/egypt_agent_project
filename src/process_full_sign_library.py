from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# =========================
# 1. 路径设置
# =========================
PROJECT_DIR = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project")

SIGN_DIR = PROJECT_DIR / "data_sign_demo"

MERGED_LIBRARY_CSV = SIGN_DIR / "sign_library_merged.csv"
OUTPUT_CSV = SIGN_DIR / "sign_library_merged_processed.csv"

SIGN_PROCESSED_FULL_DIR = SIGN_DIR / "sign_processed_full"
SIGN_PROCESSED_FULL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SIZE = 128


# =========================
# 2. 图片读取函数
# =========================
def read_image_gray(image_path: Path):
    """
    支持 Windows 中文路径的图片读取。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return None

    img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    return img


# =========================
# 3. 图像预处理函数
# =========================
def preprocess_sign_image_from_array(img, output_size: int = 128):
    """
    对全量标准符号图片进行统一标准化处理：
    1. 灰度图输入
    2. 二值化
    3. 提取非背景区域
    4. 裁剪
    5. 加边距
    6. 等比例缩放
    7. 居中到 128×128
    """
    if img is None:
        return None

    _, binary_inv = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary_inv)

    if coords is None:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary_inv[y:y + h, x:x + w]

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

    h2, w2 = cropped.shape
    scale = (output_size - 20) / max(h2, w2)

    new_w = max(1, int(w2 * scale))
    new_h = max(1, int(h2 * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    start_x = (output_size - new_w) // 2
    start_y = (output_size - new_h) // 2

    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return canvas


# =========================
# 4. 主流程
# =========================
def main():
    if not MERGED_LIBRARY_CSV.exists():
        raise FileNotFoundError(f"未找到合并符号库：{MERGED_LIBRARY_CSV}")

    df = pd.read_csv(MERGED_LIBRARY_CSV, dtype=str).fillna("")

    print("=" * 80)
    print("开始预处理全量 Unicode 古埃及符号库")
    print("=" * 80)
    print("输入文件：", MERGED_LIBRARY_CSV)
    print("输入记录数：", len(df))
    print("输出文件夹：", SIGN_PROCESSED_FULL_DIR)

    processed_paths = []
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        sign_id = row.get("sign_id", "")
        png_path = row.get("png_path", "")

        if not sign_id or not png_path:
            processed_paths.append("")
            fail_count += 1
            continue

        img = read_image_gray(Path(png_path))

        if img is None:
            processed_paths.append("")
            fail_count += 1
            continue

        processed_img = preprocess_sign_image_from_array(img, OUTPUT_SIZE)

        if processed_img is None:
            processed_paths.append("")
            fail_count += 1
            continue

        output_path = SIGN_PROCESSED_FULL_DIR / f"{sign_id}_processed.png"

        ok = cv2.imwrite(str(output_path), processed_img)

        if ok:
            processed_paths.append(str(output_path))
            success_count += 1
        else:
            processed_paths.append("")
            fail_count += 1

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1} / {len(df)} 条，成功 {success_count} 条")

    df["processed_png_path"] = processed_paths

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("全量符号库预处理完成！")
    print("=" * 80)
    print("总记录数：", len(df))
    print("成功处理：", success_count)
    print("失败数量：", fail_count)
    print("输出 CSV：", OUTPUT_CSV)
    print("输出文件夹：", SIGN_PROCESSED_FULL_DIR)

    print("\n前 10 条预览：")
    print(df.head(10))


if __name__ == "__main__":
    main()