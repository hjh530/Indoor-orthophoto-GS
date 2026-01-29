"""
stitch_5x5_grid_feature_matching.py

功能：
 1. 解析 virtual_..._rX_cY.png 文件名。
 2. 第一阶段：按行 (Row) 拼接，生成 5 个横向长条。
 3. 第二阶段：将 5 个长条旋转 90 度后拼接，再转回来 (解决垂直拼接问题)。
 4. 使用 OpenCV SIFT/ORB 特征匹配，无需坐标文件。
"""

import cv2
import os
import re
import numpy as np

# ================= 配置 =================
# 输入图片文件夹
image_folder = r"C:\Users\86181\Desktop\classroom\todm"

# 输出路径
output_path = r"C:\Users\86181\Desktop\classroom\final_feature_stitched.jpg"

# 缩放因子 (0.5 表示缩小一半处理，防止 25张图 内存爆炸)
# 建议先用 0.5 测试，成功后再改回 1.0
SCALE_FACTOR = 1.0 
# =======================================

def parse_grid_info(filename):
    """ 解析文件名获取行列号 """
    match = re.search(r'_r(\d+)_c(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def stitch_images(img_list, mode="horizontal"):
    """
    调用 OpenCV 拼接器拼接一组图片
    """
    if len(img_list) < 2:
        print("  [跳过] 图片少于2张，无需拼接")
        return img_list[0]

    # 初始化拼接器 (SCANS 模式适合正射/平面)
    stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_SCANS)
    # 稍微降低置信度阈值，防止因为纹理少而拼接失败
    stitcher.setPanoConfidenceThresh(0.99) 

    status, pano = stitcher.stitch(img_list)

    if status != cv2.Stitcher_OK:
        print(f"  [失败] 拼接错误代码: {status}")
        return None
    return pano

def rotate_image(image, angle):
    """ 旋转图片 (90度倍数) """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def main():
    print(f"正在读取图片: {image_folder} ...")
    
    # 1. 加载并整理图片到 5x5 网格结构
    # grid_map[row_index] = { col_index: image_data }
    grid_map = {}
    
    files = os.listdir(image_folder)
    for f in files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        
        row, col = parse_grid_info(f)
        if row is None: continue
        
        path = os.path.join(image_folder, f)
        img = cv2.imread(path)
        if img is None: continue

        # 缩放加速
        if SCALE_FACTOR != 1.0:
            img = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        
        if row not in grid_map: grid_map[row] = {}
        grid_map[row][col] = img

    # 检查行数
    sorted_rows = sorted(grid_map.keys())
    print(f"检测到 {len(sorted_rows)} 行数据: {sorted_rows}")

    # 2. 第一阶段：逐行拼接 (生成长条)
    row_strips = [] # 存储拼好的每一行
    
    print("\n=== 第一阶段：行拼接 (Horizontal) ===")
    for r in sorted_rows:
        print(f"正在拼接第 {r} 行...")
        cols = sorted(grid_map[r].keys())
        
        # 获取这一行的所有图片 (按列排序)
        row_imgs = [grid_map[r][c] for c in cols]
        
        # 执行拼接
        strip = stitch_images(row_imgs)
        if strip is not None:
            row_strips.append(strip)
            # 可选：保存中间结果看看
            # cv2.imwrite(f"debug_row_{r}.jpg", strip)
        else:
            print(f"警告：第 {r} 行拼接失败，最终结果可能会缺失这一行！")

    if not row_strips:
        print("错误：所有行都拼接失败。")
        return

    # 3. 第二阶段：纵向拼接 (Vertical)
    # 技巧：OpenCV 拼接器擅长拼左右结构的图。
    # 我们把所有长条顺时针旋转 90 度，这就变成了"左右拼接"。
    # 拼完后再逆时针转回来。
    
    print("\n=== 第二阶段：列拼接 (Vertical) ===")
    print("正在旋转长条以适配算法...")
    
    # 注意：我们之前生成的顺序是 Row 1 (Top) -> Row 5 (Bottom)
    # 旋转 90 度后，Top 在右边，Bottom 在左边。
    # 为了保证拼接顺序，我们需要把列表反过来，或者旋转后调整顺序。
    # 简单做法：全部旋转 90 度，然后直接拼。
    
    rotated_strips = [rotate_image(s, 90) for s in row_strips]
    
    # 再次调用拼接器
    print("正在合并所有行...")
    final_vertical_pano = stitch_images(rotated_strips)
    
    if final_vertical_pano is not None:
        # 拼好后是横着的，需要转回 -90 度
        print("正在恢复方向...")
        final_result = rotate_image(final_vertical_pano, -90)
        
        cv2.imwrite(output_path, final_result)
        print(f"\n[成功] 最终正射影像已保存: {output_path}")
    else:
        print("\n[失败] 无法合并行长条。可能是重叠不够或特征点不足。")

if __name__ == "__main__":
    main()




