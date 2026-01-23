"""
align_room_10x10_final.py

逻辑确认：
 1. Z轴向上 (Up)。
 2. 相机垂直向下看 (Look Down, Quaternion=[0,1,0,0])。
 3. 相机高度 = 地板 + (天花板-地板)*0.8。
 4. 生成顺序：从左上(MinX, MaxY) 到 右下(MaxX, MinY)。
"""

import numpy as np
import os
import random

# ---------- 配置 ----------
sparse_dir = r"C:\\Users\\86181\\Desktop\\classroom\\sparse\\0"
output_camerassparse_dir = r"C:\\Users\\86181\\Desktop\\classroom\\sparse\\1"

images_txt = os.path.join(sparse_dir, "images.txt")
points_txt = os.path.join(sparse_dir, "points3D.txt")
cameras_txt = os.path.join(sparse_dir, "cameras.txt")

final_output_images = os.path.join(output_camerassparse_dir, "images.txt")
final_output_points = os.path.join(output_camerassparse_dir, "points3D.txt")
final_output_cameras = os.path.join(output_camerassparse_dir, "cameras.txt")

# RANSAC 参数
RANSAC_ITER = 2000
DIST_THRESHOLD = 0.02
GROUND_POINT_ID = None

# ---------- 基础数学工具 ----------
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)]
    ])

def rotmat2qvec(R):
    qw = np.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    if qw == 0: return np.array([1,0,0,0])
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return np.array([qw, qx, qy, qz])

def normalize(v):
    n = np.linalg.norm(v)
    if n == 0: return v
    return v / n

def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# ---------- 读写函数 ----------
def read_cameras_txt(path):
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip()=="": continue
            line = line.replace(",", " ")
            e = line.split()
            cams[int(e[0])] = {"model":e[1], "width":int(e[2]), "height":int(e[3]), "params":np.array(list(map(float, e[4:])))}
    return cams

def write_cameras_txt(path, cams):
    with open(path, 'w') as f:
        f.write("# Camera list (ID, model, width, height, params)\n")
        for cid in sorted(cams.keys()):
            cam = cams[cid]
            f.write(f"{cid} {cam['model']} {cam['width']} {cam['height']} " + " ".join(map(str, cam['params'])) + "\n")

def read_images_txt(path):
    images, point_tracks = {}, {}
    with open(path, 'r') as f: lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or line == "": i += 1; continue
        line = line.replace(",", " ")
        e = line.split()
        try:
            img_id = int(e[0])
            qvec = np.array(list(map(float, e[1:5])))
            tvec = np.array(list(map(float, e[5:8])))
            cam_id = int(e[8])
            name = e[9]
        except ValueError:
            i+=1; continue
        i += 1
        pts2d = []
        if i < len(lines):
            l2 = lines[i].strip()
            if l2 != "":
                l2 = l2.replace(",", " ")
                e2 = l2.split()
                for idx in range(0, len(e2), 3):
                    pts2d.append((float(e2[idx]), float(e2[idx+1]), int(e2[idx+2])))
                    if int(e2[idx+2]) != -1: point_tracks.setdefault(int(e2[idx+2]), []).append((img_id, idx//3))
        images[img_id] = {"qvec":qvec, "tvec":tvec, "camera_id":cam_id, "name":name, "points2D":pts2d}
        i += 1
    return images, point_tracks

def write_images_txt(path, images):
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id in sorted(images.keys()):
            img = images[img_id]
            q = " ".join(map(str, img["qvec"]))
            t = " ".join(map(str, img["tvec"]))
            f.write(f"{img_id} {q} {t} {img['camera_id']} {img['name']}\n")
            if img["points2D"]:
                f.write(" ".join(f"{x} {y} {pid}" for x,y,pid in img["points2D"]) + "\n")
            else:
                f.write("\n")

def read_points3D_txt(path):
    pts = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip()=="": continue
            line = line.replace(",", " ")
            e = line.split()
            pts[int(e[0])] = {"xyz":np.array(list(map(float, e[1:4]))), "rgb":(int(e[4]), int(e[5]), int(e[6])), "error":float(e[7])}
    return pts

def write_points3D_txt(path, points3D, point_tracks):
    with open(path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3D)}\n")
        for pid in sorted(points3D.keys()):
            pt = points3D[pid]
            xyz = " ".join(map(str, pt["xyz"]))
            rgb = " ".join(map(str, pt["rgb"]))
            err = pt["error"]
            line = f"{pid} {xyz} {rgb} {err}"
            if pid in point_tracks:
                line += " " + " ".join(f"{img_id} {idx}" for img_id, idx in point_tracks[pid])
            line += "\n"
            f.write(line)

def fit_plane_from_points(pts):
    centroid = pts.mean(axis=0); U, S, Vt = np.linalg.svd(pts - centroid)
    return Vt[-1, :] / np.linalg.norm(Vt[-1, :]), centroid

def ransac_plane(pts, iters=2000, thresh=0.02):
    best_inliers, best_normal, best_p0 = [], None, None
    N = pts.shape[0]
    for _ in range(iters):
        ids = random.sample(range(N), 3)
        p1 = pts[ids[0]]; v1 = pts[ids[1]] - p1; v2 = pts[ids[2]] - p1
        n = np.cross(v1, v2); norm = np.linalg.norm(n)
        if norm < 1e-6: continue
        n /= norm
        dists = np.abs((pts - p1).dot(n))
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers): best_inliers, best_normal, best_p0 = inliers, n, p1
    if best_normal is None: best_normal, best_p0 = fit_plane_from_points(pts)
    return best_normal, best_p0, best_inliers

def rotation_from_vector_to_z(n):
    n = normalize(n); target = np.array([0., 0., 1.])
    if np.allclose(n, target): return np.eye(3)
    if np.allclose(n, -target): return np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    v = np.cross(n, target); s = np.linalg.norm(v); c = np.dot(n, target)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s**2))

# ---------- 算法模块 ----------

def align_xy_axes(points):
    """ 计算最佳水平旋转角度，使包围盒最小（摆正房间） """
    if len(points) == 0: return np.eye(3)
    pts_xy = points[:, :2]
    best_angle = 0.0
    min_area = float('inf')
    
    # 搜索 0~90度
    search_angles = np.linspace(0, 90, 90) 
    for angle_deg in search_angles:
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pts_rotated = pts_xy.dot(R.T)
        
        # 使用 5% - 95% 抗噪计算面积
        min_x = np.percentile(pts_rotated[:, 0], 5)
        max_x = np.percentile(pts_rotated[:, 0], 95)
        min_y = np.percentile(pts_rotated[:, 1], 5)
        max_y = np.percentile(pts_rotated[:, 1], 95)
        area = (max_x - min_x) * (max_y - min_y)
        
        if area < min_area:
            min_area = area
            best_angle = theta
            
    print(f"Manhattan Alignment: Rotated {np.degrees(best_angle):.2f} deg")
    return rotation_matrix_z(best_angle)

def get_robust_bbox(points_array, lower_percent=2.0, upper_percent=98.0):
    if len(points_array) == 0:
        return 0, 0, 0, 0, 0, 0
    # 获取鲁棒的物理边界
    min_x = np.percentile(points_array[:, 0], lower_percent)
    max_x = np.percentile(points_array[:, 0], upper_percent)
    min_y = np.percentile(points_array[:, 1], lower_percent)
    max_y = np.percentile(points_array[:, 1], upper_percent)
    min_z = np.percentile(points_array[:, 2], lower_percent)
    max_z = np.percentile(points_array[:, 2], upper_percent)
    return min_x, max_x, min_y, max_y, min_z, max_z

# ---------- [核心修改] 10x10 Grid (正确顺序 & 高度计算) ----------
def generate_render_views_10x10(images, bbox, target_height):
    merged_images = dict(images)
    existing_ids = sorted(images.keys())
    next_id = existing_ids[-1] + 1 if len(existing_ids) > 0 else 1

    min_x, max_x, min_y, max_y, _, _ = bbox
    
    ref_camera_id = None
    for img in images.values():
        if not img["name"].startswith("Terr_"):
            ref_camera_id = img["camera_id"]
            break
    if ref_camera_id is None: return images

    print(f"--- 场景范围 (XY已校正) ---")
    print(f"X: {min_x:.2f} ~ {max_x:.2f}")
    print(f"Y: {min_y:.2f} ~ {max_y:.2f}")
    print(f"设定高度: {target_height:.2f}")
    
    # 相机朝向：绕X轴旋转180度 -> Z轴变-Z(下), Y轴变-Y(下)
    # 这样相机坐标系的 +Z (前方) 指向世界 -Z (地下) -> 符合俯视
    # 相机坐标系的 +Y (图像下方) 指向世界 -Y (南) -> 符合地图习惯
    q_ortho = np.array([0.0, 1.0, 0.0, 0.0])
    R_ortho = qvec2rotmat(q_ortho)

    # 10x10 采样点比例
    ratios = [(i + 0.5) / 10.0 for i in range(10)]
    
    # [核心顺序修正]
    # 行遍历 (Y): 从 MaxY (北/上) -> MinY (南/下)
    y_ratios = list(reversed(ratios)) 
    # 列遍历 (X): 从 MinX (西/左) -> MaxX (东/右)
    x_ratios = ratios             

    print(f"生成 10x10=100 张正射位姿...")

    count = 1
    # 外层循环: Row 1 -> Row 10 (从上到下)
    for row_idx, fy in enumerate(y_ratios, start=1):
        # 内层循环: Col 1 -> Col 10 (从左到右)
        for col_idx, fx in enumerate(x_ratios, start=1):
            
            # 计算绝对坐标
            cx = min_x + fx * (max_x - min_x)
            cy = min_y + fy * (max_y - min_y)
            cz = target_height
            
            Cr = np.array([cx, cy, cz])
            
            # 计算平移向量 t = -R * C
            t_r = -R_ortho.dot(Cr)
            
            virtual_name = f"virtual_{count}_r{row_idx}_c{col_idx}.png"
            
            merged_images[next_id] = {
                "qvec": q_ortho, "tvec": t_r,
                "camera_id": ref_camera_id, "name": virtual_name, "points2D": []
            }
            next_id += 1
            count += 1

    return merged_images

# ---------- 主流程 ----------
def main():
    if not os.path.exists(output_camerassparse_dir): os.makedirs(output_camerassparse_dir)

    print("读取 txt ...")
    try:
        images, point_tracks = read_images_txt(images_txt)
        points3D = read_points3D_txt(points_txt)
        cameras = read_cameras_txt(cameras_txt)
    except Exception as e:
        print(f"读取失败: {e}"); return

    all_ids = sorted(points3D.keys())
    pts = np.vstack([points3D[pid]["xyz"] for pid in all_ids])
    
    # 1. 地面 RANSAC
    print(f"RANSAC 地面拟合...")
    normal, p0, inliers = ransac_plane(pts, iters=RANSAC_ITER, thresh=DIST_THRESHOLD)
    
    # [Z轴方向修正] 确保 Z 指向天花板
    cam_centers_raw = []
    for img in images.values():
        if img["name"].startswith("Terr_"): continue
        R_raw = qvec2rotmat(img["qvec"])
        C_raw = -R_raw.T.dot(img["tvec"])
        cam_centers_raw.append(C_raw)
    
    if len(cam_centers_raw) > 0:
        avg_cam_raw = np.mean(cam_centers_raw, axis=0)
        # 如果相机在法向量反方向，说明法向量指反了
        if np.dot(avg_cam_raw - p0, normal) < 0:
            print("[INFO] Z 轴指向地下，正在翻转...")
            normal = -normal
        else:
            print("[INFO] Z 轴方向正确。")
    else:
        if normal[2] < 0: normal = -normal

    A1 = rotation_from_vector_to_z(normal)
    
    # 计算中间平移量
    if GROUND_POINT_ID is not None and GROUND_POINT_ID in points3D:
        P_ground = points3D[GROUND_POINT_ID]["xyz"].copy()
    else:
        inlier_pts = pts[inliers]; centroid = inlier_pts.mean(axis=0)
        P_ground = centroid - np.dot(centroid - p0, normal) * normal

    # 2. XY 对齐
    pts_aligned_z = (pts - P_ground).dot(A1.T)
    print("正在计算 XY 轴对齐 (摆正房间)...")
    A2 = align_xy_axes(pts_aligned_z) 
    
    A_final = A2.dot(A1)
    
    # 应用变换
    new_points = {}
    all_aligned_pts_list = []
    
    for pid, pt in points3D.items():
        new_xyz = A_final.dot(pt["xyz"] - P_ground)
        new_points[pid] = {"xyz": new_xyz, "rgb": pt["rgb"], "error": pt["error"]}
        all_aligned_pts_list.append(new_xyz)
    
    all_aligned_pts = np.array(all_aligned_pts_list)
        
    new_images = {}
    for img_id, img in images.items():
        qvec = img["qvec"]; tvec = img["tvec"]
        R_i = qvec2rotmat(qvec)
        C_old = -R_i.T.dot(tvec)
        
        # 变换 R 和 C
        R_i_new = R_i.dot(A_final.T)
        C_new = A_final.dot(C_old - P_ground)
        t_i_new = -R_i_new.dot(C_new)
        
        new_images[img_id] = {
            "qvec": rotmat2qvec(R_i_new), "tvec": t_i_new, 
            "camera_id": img["camera_id"], "name": img["name"], "points2D": img["points2D"]
        }
    
    # 3. 计算 BBox 和高度 (0.8)
    if len(all_aligned_pts) > 100:
        robust_bbox = get_robust_bbox(all_aligned_pts, lower_percent=2.0, upper_percent=98.0)
        min_x, max_x, min_y, max_y, min_z, max_z = robust_bbox
        
        mask = (all_aligned_pts[:, 0] > min_x) & (all_aligned_pts[:, 0] < max_x) & \
               (all_aligned_pts[:, 1] > min_y) & (all_aligned_pts[:, 1] < max_y)
        clean_points = all_aligned_pts[mask]
        
        # [核心修改]：使用几何高度的 80%
        # 高度 = MinZ + (MaxZ - MinZ) * 0.8
        height_range = max_z - min_z
        target_height = min_z + height_range * 0.5
        
        print(f"高度统计: MinZ={min_z:.2f}, MaxZ={max_z:.2f}, Range={height_range:.2f}")
        print(f"目标高度 (80%): {target_height:.2f}")

    else:
        robust_bbox = (0,0,0,0,0,0)
        target_height = 2.5

    print("生成并合并渲染视角 (10x10) ...")
    merged_all_images = generate_render_views_10x10(new_images, robust_bbox, target_height)
    
    print(f"写入最终文件到 {output_camerassparse_dir} ...")
    write_images_txt(final_output_images, merged_all_images)
    write_points3D_txt(final_output_points, new_points, point_tracks)
    write_cameras_txt(final_output_cameras, cameras)
    print("完成。")

if __name__ == "__main__":
    main()




