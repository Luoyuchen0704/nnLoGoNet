import os
import re
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import multiprocessing
from PIL import Image, ImageSequence
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt, label, binary_erosion
import cv2

def rename_fives_inplace(target_dir):
    """
    原地重命名 FIVES 数据集文件
    规则: 1_A.png -> FIVES_001_0000.png
    逻辑: 提取文件名开头的数字进行排序，然后按顺序编号
    
    参数:
        target_dir: 需要处理的目标文件夹路径
    """
    
    # 1. 检查路径
    if not os.path.exists(target_dir):
        print(f"❌ 错误: 路径不存在 {target_dir}")
        return

    # 2. 获取所有 png 文件
    # 过滤掉已经重命名过的文件 (以 CHASE-DB1_ 开头)，防止重复执行导致混乱
    all_files = [f for f in os.listdir(target_dir) if f.endswith('.png')]
    raw_files = [f for f in all_files ]
    
    if not raw_files:
        print("⚠️ 未找到需要重命名的原始文件 (或文件已全部重命名)")
        return

    # 3. 自然排序 (关键)
    # 提取文件名开头的数字: '1_A.png' -> 1, '10_A.png' -> 10
    try:
        raw_files.sort()
    except Exception as e:
        print(f"❌ 排序失败，文件名格式可能不统一: {e}")
        return

    print(f"找到 {len(raw_files)} 个原始文件，准备原地重命名...")
    print(f"  首个文件: {raw_files[0]}")
    print(f"  末尾文件: {raw_files[-1]}")

    count = 0
    # 4. 遍历并执行重命名
    for i, filename in enumerate(raw_files, start=1):
        
        # 限制只处理前 20 个 (如果需要)
        if i > 20:
            break

        # 构建新文件名: CHASE-DB1_001.jpg
        new_filename = f"CHASE-DB1_{i:03d}.png"
        
        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, new_filename)
        
        # 安全检查：防止覆盖已存在的文件
        if os.path.exists(new_path):
            print(f"⚠️ 跳过: {new_filename} 已存在，防止覆盖")
            continue

        try:
            # --- 核心操作: 重命名 ---
            os.rename(old_path, new_path)
            count += 1
            
            # 可选: 打印进度
            # print(f"Renamed: {filename} -> {new_filename}")
            
        except Exception as e:
            print(f"❌ 重命名失败 {filename}: {e}")

    print(f"\n✅ 原地重命名完成！共修改了 {count} 个文件。")
    print(f"目标路径: {target_dir}")
    
def check_label_values(input_dir):
    """
    检查指定目录下所有 .png 图片的像素值分布
    用于确认标签是否为二值 (0/1) 或者 (0/255)
    """
    
    if not os.path.exists(input_dir):
        print(f"❌ 错误: 路径不存在 {input_dir}")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    files.sort()
    
    if not files:
        print("⚠️ 目录下没有找到 .png 文件")
        return

    print(f"正在检查 {len(files)} 个标签文件中的像素值...")
    
    # 用于统计所有出现过的唯一值组合
    # 格式: { (0, 255): 100张, (0, 1): 50张 }
    value_patterns = {}
    
    # 用于记录有问题的具体文件名（最多记录5个）
    problematic_files = []

    for filename in tqdm(files, desc="Checking"):
        file_path = os.path.join(input_dir, filename)
        
        try:
            # 读取图片 (转换为数组)
            img = Image.open(file_path)
            arr = np.array(img)
            
            # 如果是 RGB (H, W, 3)，只取一个通道，或者检查是否有杂色
            if arr.ndim == 3:
                # 简单检查：是否三个通道值一样 (灰度)
                # 这里为了速度，只取第一个通道进行统计
                arr = arr[:, :, 0] 
            
            # 获取该图片所有的唯一像素值
            unique_vals = np.unique(arr)
            
            # 将 numpy array 转为 tuple 以便作为字典的 key
            vals_tuple = tuple(unique_vals)
            
            if vals_tuple in value_patterns:
                value_patterns[vals_tuple] += 1
            else:
                value_patterns[vals_tuple] = 1
            
            # 检查是否有异常值 (例如不是纯粹的 0,1 或 0,255)
            # 只要唯一值数量超过 2 个 (针对二分类)，或者包含非 0/1/255 的值，就可能是插值伪影
            if len(unique_vals) > 2:
                if len(problematic_files) < 5:
                    problematic_files.append((filename, unique_vals))

        except Exception as e:
            print(f"❌ 读取错误 {filename}: {e}")

    # --- 输出报告 ---
    print("\n" + "="*40)
    print(" 📊 标签数值检查报告")
    print("="*40)
    
    for vals, count in value_patterns.items():
        status = ""
        if vals == (0, 1):
            status = "✅ 完美 (适合 nnU-Net 直接训练)"
        elif vals == (0, 255):
            status = "⚠️ 需要转换 (0保持, 255->1)"
        elif vals == (0,):
            status = "⚠️ 全黑图像 (没有前景目标)"
        else:
            status = "❌ 异常 (可能是插值导致的边缘噪声)"
            
        print(f"数值组合 {vals}: {count} 张图片 -> {status}")

    if problematic_files:
        print("\n🔍 发现异常文件示例 (可能存在边缘模糊/插值问题):")
        for fname, vals in problematic_files:
            # 如果值太多，只打印前10个
            val_str = str(vals) if len(vals) < 10 else str(vals[:10]) + "..."
            print(f"  - {fname}: {val_str}")

def convert_255_to_1_inplace(target_dir):
    """
    原地读取文件夹内的 png 标签，将像素值 255 转换为 1。
    同时使用阈值法 (>128) 处理，以消除可能的边缘插值噪声。
    """
    
    if not os.path.exists(target_dir):
        print(f"❌ 错误: 路径不存在 {target_dir}")
        return

    # 获取所有 png 文件
    files = [f for f in os.listdir(target_dir) if f.endswith('.png')]
    
    if not files:
        print("⚠️ 未找到 png 文件")
        return

    print(f"正在处理 {len(files)} 个文件 (目标: 0/255 -> 0/1)...")
    print("⚠️ 注意: 转换后图片预览将变为全黑，这是正常的！")

    count_converted = 0
    count_skipped = 0

    for filename in tqdm(files, desc="Converting"):
        file_path = os.path.join(target_dir, filename)
        
        try:
            # 1. 读取图片
            img = Image.open(file_path)
            arr = np.array(img)
            
            # 2. 检查是否有 255 (或者 >1 的值)
            # 如果最大值已经是 1，说明可能处理过了，跳过以节省时间
            if arr.max() <= 1:
                count_skipped += 1
                continue

            # 3. 维度处理 (如果是 RGB/RGBA，转为单通道)
            if arr.ndim == 3:
                # 假设三个通道一致，取第一个；或者取最大值通道
                arr = arr[:, :, 0]

            # 4. --- 核心转换逻辑 (阈值法) ---
            # 大于 128 的变为 1，小于等于 128 的变为 0
            # 使用 astype(np.uint8) 确保格式正确
            new_arr = np.where(arr > 128, 1, 0).astype(np.uint8)

            # 5. 保存回原路径 (覆盖)
            # 必须保存为 'L' (8-bit 灰度) 或 '1' (1-bit 二值)
            # nnU-Net 推荐 'L' 模式 (uint8)，兼容性最好
            new_img = Image.fromarray(new_arr, mode='L')
            new_img.save(file_path)
            
            count_converted += 1

        except Exception as e:
            print(f"❌ 处理失败 {filename}: {e}")

    print("\n" + "="*30)
    print("✅ 转换完成")
    print(f"  - 转换了: {count_converted} 张")
    print(f"  - 跳过了: {count_skipped} 张 (已经是 0/1)")
    print(f"  - 目标路径: {target_dir}")
    print("="*30)
    print("💡 提示: 现在可以用 inspect_results.py 再次检查，应该只看到 (0, 1)。")

def convert_images_to_grayscale_inplace(target_dir, method='green'):
    """
    遍历文件夹下的 .png 和 .bmp 图片，将其转换为单通道灰度图并覆盖保存。
    
    参数:
        target_dir: 图片文件夹路径 (imagesTr)
        method: 'standard' (标准灰度) 或 'green' (提取绿色通道 - 推荐用于眼底图)
    """
    
    if not os.path.exists(target_dir):
        print(f"❌ 错误: 路径不存在 {target_dir}")
        return

    # 修改点 1: 支持 .png 和 .bmp，并忽略大小写
    valid_extensions = ('.png', '.bmp', '.jpg')
    files = [f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)]
    files.sort()
    
    if not files:
        print("⚠️ 未找到 png 或 bmp 文件")
        return

    print(f"正在处理 {len(files)} 张图片...")
    print(f"转换模式: {'提取绿色通道 (推荐)' if method == 'green' else '标准灰度转换'}")
    print(f"目标路径: {target_dir}")

    count = 0
    
    for filename in tqdm(files, desc="Converting"):
        file_path = os.path.join(target_dir, filename)
        
        try:
            # 1. 读取图片
            img = Image.open(file_path)
            
            # 2. 检查是否已经是单通道 (L模式)
            if img.mode == 'L':
                continue # 已经是灰度，跳过
            
            # 3. 转换逻辑
            if method == 'green':
                # --- 方案 A: 提取绿色通道 (眼底血管分割首选) ---
                # 将图片转为 RGB (防止是 RGBA 或 P 模式)
                img = img.convert('RGB')
                r, g, b = img.split()
                new_img = g # 只取绿色通道
            else:
                # --- 方案 B: 标准灰度转换 (0.299R + 0.587G + 0.114B) ---
                new_img = img.convert('L')
            
            # 4. 覆盖保存
            # 注意: BMP 会保存为灰度 BMP，PNG 会保存为灰度 PNG，保持原格式后缀
            new_img.save(file_path)
            count += 1
            
        except Exception as e:
            print(f"❌ 处理失败 {filename}: {e}")

    print("\n✅ 转换完成！")
    print(f"共处理了 {count} 张 RGB 图片，现在它们都是单通道了。")
    print("请重新运行 nnUNetv2_plan_and_preprocess。")

def batch_convert_png_to_jpg(folder_path):
    """
    遍历指定路径，将所有 PNG 转换为 JPG 保存在同目录下。
    自动将透明背景填充为白色。
    """
    if not os.path.exists(folder_path):
        print("路径不存在")
        return

    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            # 生成同名 .jpg 路径
            new_file_path = os.path.splitext(file_path)[0] + '.jpg'
            
            try:
                with Image.open(file_path) as img:
                    # 处理透明通道：如果有透明度，创建一个白底图层合并
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        img = img.convert('RGBA')
                        bg = Image.new('RGB', img.size, (255, 255, 255)) # 白色背景
                        bg.paste(img, mask=img.split()[3])
                        img = bg
                    else:
                        img = img.convert('RGB')
                    
                    # 保存 (quality=95 保证画质)
                    img.save(new_file_path, 'JPEG', quality=95)
                    count += 1
                    print(f"转换成功: {filename} -> .jpg")
            except Exception as e:
                print(f"文件出错 {filename}: {e}")

    print(f"\n处理完成，共转换 {count} 张图片。")

def batch_jpg_to_png(folder_path):
    """
    遍历指定文件夹，将所有的 .jpg/.jpeg 图片转换为 .png 格式。
    生成的 .png 会保存在同一目录下。
    """
    # 1. 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误：路径不存在 -> {folder_path}")
        return

    count = 0
    # 2. 遍历文件夹
    for filename in os.listdir(folder_path):
        # 忽略大小写检查后缀
        if filename.lower().endswith(('.jpg', '.jpeg')):
            full_src_path = os.path.join(folder_path, filename)
            
            # 3. 生成新的文件名 (把后缀换成 .png)
            name_without_ext = os.path.splitext(filename)[0]
            new_filename = name_without_ext + ".png"
            full_dst_path = os.path.join(folder_path, new_filename)
            
            try:
                with Image.open(full_src_path) as img:
                    # JPG 只有 RGB，转 PNG 直接保存即可
                    # 默认使用最大压缩等级 (compress_level=9) 以节省空间，无损压缩
                    img.save(full_dst_path, 'PNG')
                    count += 1
                    print(f"转换成功: {filename} -> {new_filename}")
            except Exception as e:
                print(f"转换失败 {filename}: {e}")

    print(f"\n全部完成！共转换 {count} 张图片。")

def evaluate_final(gt_dir, pred_dir, mask_dir):
    img_exts = ('.tif', '.png', '.gif', '.jpg')
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(img_exts)])
    
    metrics = {"Acc": [], "AUC": [], "Dice": [], "MCC": [], "HD95": []}
    
    print(f"🚀 开始评估 (自动降维 + 自动缩放 + .npz读取)...")
    
    count = 0
    has_prob_flag = False

    for gt_name in gt_files:
        stem = os.path.splitext(gt_name)[0] # DRIVE_001
        
        gt_path = os.path.join(gt_dir, gt_name)
        
        # 1. 匹配 Mask
        mask_candidates = [f for f in os.listdir(mask_dir) if stem in f]
        if not mask_candidates:
             import re
             m = re.search(r'(\d+)', stem)
             if m:
                 mask_candidates = [f for f in os.listdir(mask_dir) if m.group(1) in f]
        
        if not mask_candidates:
            print(f"⚠️ 跳过: {gt_name} 找不到 Mask")
            continue
        mask_path = os.path.join(mask_dir, mask_candidates[0])
        
        # 2. 匹配 Prediction
        pred_npz = os.path.join(pred_dir, stem + ".npz")
        # 模糊匹配 .npz
        if not os.path.exists(pred_npz):
            candidates = [f for f in os.listdir(pred_dir) if stem in f and f.endswith('.npz')]
            if candidates: pred_npz = os.path.join(pred_dir, candidates[0])

        pred_img_candidates = [f for f in os.listdir(pred_dir) if f.startswith(stem) and f.endswith(img_exts)]
        pred_img_path = os.path.join(pred_dir, pred_img_candidates[0]) if pred_img_candidates else None
        
        try:
            # 读取 GT 和 Mask
            gt_img = np.array(Image.open(gt_path).convert('L'))
            mask_img = np.array(Image.open(mask_path).convert('L'))
            roi_mask = mask_img > 0
            
            # 读取 Pred
            prob_map = None
            pred_binary = None

            if os.path.exists(pred_npz):
                # 读取 .npz
                data = np.load(pred_npz)['probabilities'] 
                # data shape 可能是 (2, 1, 584, 565) 或 (2, 584, 565)
                
                # 取前景类 (索引1)
                prob_raw = data[1] 
                
                # --- 核心修复：降维 ---
                # 如果是 (1, 584, 565) -> 变成 (584, 565)
                prob_map = np.squeeze(prob_raw)
                
                has_prob_flag = True
                pred_binary = (prob_map > 0.5).astype(np.uint8)
                
            elif pred_img_path and os.path.exists(pred_img_path):
                pred_arr = np.array(Image.open(pred_img_path).convert('L'))
                if pred_arr.max() <= 1:
                    pred_binary = pred_arr.astype(np.uint8)
                    prob_map = pred_arr.astype(np.float32)
                else:
                    pred_binary = (pred_arr > 127).astype(np.uint8)
                    prob_map = pred_arr.astype(np.float32) / 255.0
            else:
                print(f"❌ 找不到预测文件: {stem}")
                continue

            # --- 核心修复: 尺寸对齐 ---
            # 如果 npz 出来的尺寸和 GT 不一样 (包括 512 vs 584 的情况)
            if gt_img.shape != pred_binary.shape:
                # print(f"⚠️ 尺寸调整: {stem} {pred_binary.shape} -> {gt_img.shape}")
                pred_binary = resize_pred_to_gt(pred_binary, gt_img.shape, is_prob=False)
                if prob_map is not None:
                    prob_map = resize_pred_to_gt(prob_map, gt_img.shape, is_prob=True)

            # 后处理
            pred_binary = remove_small_objects(pred_binary, min_size=30)

            # 准备数据
            gt_binary = (gt_img > 127).astype(np.uint8) if gt_img.max() > 1 else gt_img.astype(np.uint8)
            
            gt_flat = gt_binary[roi_mask]
            pred_flat = pred_binary[roi_mask]
            prob_flat = prob_map[roi_mask]
            
            gt_flat = np.where(gt_flat > 0, 1, 0)
            pred_flat = np.where(pred_flat > 0, 1, 0)
            
            if len(gt_flat) == 0: continue

            # 计算
            acc = accuracy_score(gt_flat, pred_flat)
            dice = f1_score(gt_flat, pred_flat, zero_division=1)
            auc = roc_auc_score(gt_flat, prob_flat) if len(np.unique(gt_flat)) > 1 else 0.5
            mcc = matthews_corrcoef(gt_flat, pred_flat) if len(np.unique(gt_flat)) > 1 else 0.0
            hd95 = compute_hd95(gt_binary & roi_mask, pred_binary & roi_mask)

            metrics["Acc"].append(acc)
            metrics["AUC"].append(auc)
            metrics["Dice"].append(dice)
            metrics["MCC"].append(mcc)
            metrics["HD95"].append(hd95)
            
            count += 1
            
        except Exception as e:
            print(f"❌ Error {stem}: {e}")

    # 输出
    print("-" * 60)
    print(f"✅ 成功评估 {count} 个样本 " + ("(检测到 .npz 真实概率)" if has_prob_flag else "(仅使用伪概率)"))
    if count > 0:
        for key in ["Acc", "AUC", "Dice", "MCC", "HD95"]:
            vals = np.array(metrics[key])
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                mean = np.mean(vals)
                std = np.std(vals)
                unit = "(px)" if key == "HD95" else "%"
                fac = 1.0 if key == "HD95" else 100.0
                print(f"{key:<10} | {mean*fac:.4f}{unit} ± {std*fac:.4f}{unit}")
    else:
        print("❌ 依然没有成功样本，请检查上方的错误提示。")
    print("-" * 60)

def resize_pred_to_gt(pred_arr, target_shape, is_prob=False):
    """
    将预测图缩放到 GT 的尺寸 (处理 H, W 不一致的情况)
    """
    if pred_arr.shape == target_shape:
        return pred_arr
        
    # PIL resize 需要 (Width, Height)
    target_size_pil = (target_shape[1], target_shape[0]) 
    
    img = Image.fromarray(pred_arr)
    # 概率图用双线性插值，二值图用最近邻插值
    method = Image.BILINEAR if is_prob else Image.NEAREST
    
    img_resized = img.resize(target_size_pil, resample=method)
    return np.array(img_resized)

def compute_hd95(gt, pred, spacing=1.0):
    """
    计算标准 Surface HD95
    """
    # 1. 确保输入是布尔类型
    gt = gt > 0
    pred = pred > 0

    # 处理空掩码情况
    if not np.any(gt) and not np.any(pred): return 0.0
    if not np.any(gt) or not np.any(pred): return np.inf

    # 2. 【核心修改】提取轮廓 (Boundary extraction)
    # 轮廓 = 原始掩码 XOR 腐蚀后的掩码
    gt_border = np.logical_xor(gt, binary_erosion(gt))
    pred_border = np.logical_xor(pred, binary_erosion(pred))

    # 3. 计算距离变换 (针对轮廓计算)
    # 注意：我们要算的是“到 pred 轮廓的距离”，所以输入是 ~pred_border
    pred_border_dist_map = distance_transform_edt(np.logical_not(pred_border), sampling=spacing)
    gt_border_dist_map = distance_transform_edt(np.logical_not(gt_border), sampling=spacing)

    # 4. 提取表面距离
    # 在 GT 轮廓上采样：GT 轮廓点到 Pred 轮廓的最近距离
    dist_gt_to_pred = pred_border_dist_map[gt_border]
    
    # 在 Pred 轮廓上采样：Pred 轮廓点到 GT 轮廓的最近距离
    dist_pred_to_gt = gt_border_dist_map[pred_border]

    # 5. 合并并计算 95% 分位数
    all_distances = np.concatenate((dist_gt_to_pred, dist_pred_to_gt))
    
    if len(all_distances) == 0: return 0.0
    
    return np.percentile(all_distances, 95)

def remove_small_objects(pred_binary, min_size=20):
    """
    后处理：移除小于 min_size 像素的连通区域 (噪点)
    这对降低 HD95 非常有效
    """
    labeled, num_features = label(pred_binary)
    if num_features == 0: return pred_binary
    
    # 计算每个区域的大小
    component_sizes = np.bincount(labeled.ravel())
    
    # 创建一个掩码，只保留大于阈值的区域 (注意忽略背景0)
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled]
    
    pred_cleaned = pred_binary.copy()
    pred_cleaned[too_small_mask] = 0
    return pred_cleaned

def generate_perfect_masks(image_dir, output_dir, threshold=15):
    """
    使用【最小外接圆拟合】生成完美的圆形 Mask
    解决边缘锯齿问题
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_exts = ( '.jpeg', '.png', '.tif', '.bmp')
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
    
    print(f"找到 {len(files)} 张原图，开始生成完美圆形 Mask...")

    for filename in files:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        
        # 1. 转单通道
        if len(img.shape) == 3:
            # 取蓝色通道通常眼底轮廓最清晰，或者直接转灰度
            gray = img[:, :, 0] 
        else:
            gray = img

        # 2. 基础阈值分割 (得到粗糙的 Mask)
        # 稍微提高一点阈值，避开极暗的噪声
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # 3. 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"⚠️ 警告: {filename} 未检测到轮廓")
            continue

        # 4. 找到最大的轮廓 (即眼球区域)
        max_contour = max(contours, key=cv2.contourArea)

        # 5. 【核心步骤】拟合最小外接圆
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)

        # 6. 在黑底上画出这个完美的白圆
        # 半径稍微收缩一点点 (radius - 5)，去除边缘极其模糊的区域，减少假阳性
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(x), int(y)), int(radius - 2), (255), -1)

        # 7. 保存
        save_name = filename 
        # 如果需要强制存为 .gif 或其他格式，在这里修改
        # save_name = os.path.splitext(filename)[0] + ".gif"
        
        output_path = os.path.join(output_dir, save_name)
        cv2.imwrite(output_path, mask)
        
        print(f"✅ 生成完美 Mask: {filename} (R={int(radius)})")

    print("处理完成。")
def convert_labels(input_dir, output_dir):
    """
    遍历目录，将 .tif 和 .png 图片中的标签值 1 修改为 255
    """
    # 1. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 已创建输出目录: {output_dir}")

    # 支持的后缀
    valid_exts = ('.tif', '.tiff', '.png')
    
    # 获取文件列表
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not files:
        print(f"⚠️ 在 {input_dir} 中未找到 .tif 或 .png 文件")
        return

    print(f"🚀 开始处理 {len(files)} 张图片...")

    count = 0
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        save_path = os.path.join(output_dir, filename)

        try:
            # 2. 读取图片
            # 使用 PIL 读取，保持原始模式（如 'L' 模式）
            img = Image.open(file_path)
            img_array = np.array(img)

            # 3. 核心逻辑：将值为 1 的像素改为 255
            # 注意：这里只改 1 -> 255，其他值（如 0 或已经是 255 的）保持不变
            # 如果你的标签是 0 和 1，这会变成 0 和 255
            mask = (img_array == 1)
            
            if np.any(mask):
                img_array[mask] = 255
                # print(f"  - {filename}: 修改了像素值")
            
            # 4. 保存图片
            # 将数组转回 Image 对象
            # 必须确保数据类型是 uint8，否则保存可能会报错或变成全黑
            new_img = Image.fromarray(img_array.astype(np.uint8))
            
            # 如果原图是二值图但被读成了其他模式，这里强制转为 'L' (灰度) 可以减小体积
            if new_img.mode != 'L':
                new_img = new_img.convert('L')
                
            new_img.save(save_path)
            count += 1
            
        except Exception as e:
            print(f"❌ 处理 {filename} 失败: {e}")

    print(f"✅ 处理完成！成功转换并保存了 {count} 张图片到 '{output_dir}'")


if __name__ == '__main__':
    # ================= 配置路径 =================
    imagesTs = "./nnUNet_raw/Dataset018_CHASE-DB1/imagesTs"
    labelsTs = "./nnUNet_raw/Dataset018_CHASE-DB1/labelsTs"

    predictions = "./nnUNet_raw/Dataset018_CHASE-DB1/predictions/nnWNet"
    masksTs = "./nnUNet_raw/Dataset018_CHASE-DB1/masksTs"

    result_images = "./result_images/CHASE-DB1/nnWNet"

    convert_labels(predictions, result_images)
    
    # generate_perfect_masks(imagesTs, masksTs)
    # evaluate_final(labelsTs, predictions, masksTs)
    
    
    
    # 运行命令
    # python ./nnunetv2/dataset_conversion/Dataset018_CHASE-DB1.py 
    # 
    