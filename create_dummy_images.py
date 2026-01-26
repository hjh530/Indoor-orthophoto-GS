# create_dummy_images_from_intrinsics.py
import os
from PIL import Image

# ---------- 配置路径 (请根据实际情况修改) ----------
# 1. sparse 文件夹路径 (包含 images.txt 和 cameras.txt)
# 这里对应你之前生成的 output_camerassparse_dir
sparse_dir = r"C:\\Users\\86181\\Desktop\\classroom\\sparse\\1"

# 2. 图片保存路径 (存放真实 jpg 的地方)
images_dir = r"C:\\Users\\86181\\Desktop\\classroom\\images" 


# ------------------------------------------------

def read_camera_dimensions(cameras_path):
    """
    读取 cameras.txt，返回字典: {camera_id: (width, height)}
    """
    print(f"读取内参: {cameras_path}")
    dims = {}
    if not os.path.exists(cameras_path):
        print(f"错误: 找不到文件 {cameras_path}")
        return dims

    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line: continue
            
            parts = line.split()
            # 格式: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            try:
                cam_id = int(parts[0])
                width = int(parts[2])
                height = int(parts[3])
                dims[cam_id] = (width, height)
            except Exception as e:
                print(f"解析相机行出错: {line} -> {e}")
    return dims

def main():
    images_txt_path = os.path.join(sparse_dir, "images.txt")
    cameras_txt_path = os.path.join(sparse_dir, "cameras.txt")
    
    # 1. 先获取所有相机的分辨率
    cam_dims = read_camera_dimensions(cameras_txt_path)
    print(f"已加载 {len(cam_dims)} 个相机内参。")

    print(f"正在处理图像列表: {images_txt_path}")
    if not os.path.exists(images_txt_path):
        print(f"错误: 找不到文件 {images_txt_path}")
        return

    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    count = 0
    # 2. 遍历 images.txt
    for i in range(len(lines)):
        line = lines[i].strip()
        # 跳过注释和空行
        if line.startswith("#") or len(line) < 2: continue
        
        parts = line.split()
        
        # image行通常包含: IMAGE_ID, Q..., T..., CAMERA_ID, NAME
        if len(parts) >= 9:
            try:
                cam_id = int(parts[8])
                image_name = parts[9]
                
                # [关键修改] 改为 startswith("virtual")
                # 这样可以匹配 virtual_..., virtual1_..., virtual2_..., virtual3_...
                if image_name.startswith("virtual"):
                    target_path = os.path.join(images_dir, image_name)
                    
                    if not os.path.exists(target_path):
                        # 从内参字典中获取正确的分辨率
                        if cam_id in cam_dims:
                            width, height = cam_dims[cam_id]
                            
                            # 生成全黑图片
                            img = Image.new('RGB', (width, height), (0, 0, 0)) 
                            img.save(target_path)
                            
                            count += 1
                            if count % 100 == 0:
                                print(f"已生成 {count} 张... (最新: {image_name} - {width}x{height})")
                        else:
                            # 如果找不到对应相机的内参，这通常不应该发生
                            # 我们可以做一个兜底，默认给个 1024x768，防止报错
                            print(f"[Warning] 找不到 Camera ID {cam_id} 的内参，使用默认尺寸生成: {image_name}")
                            img = Image.new('RGB', (1024, 768), (0, 0, 0))
                            img.save(target_path)
                            count += 1
                            
            except ValueError:
                continue # 可能是点数据行，跳过

    print(f"完成。共生成了 {count} 张符合内参分辨率的占位图片。")

if __name__ == "__main__":
    main()

