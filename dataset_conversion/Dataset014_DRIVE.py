import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import multiprocessing
from tqdm import tqdm
import os
from PIL import Image, ImageSequence
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt, label, binary_erosion
import re

def binary_label_mapping(directory_path):
    """
    遍历指定目录，将所有图片标签二值化（0和1），并修复RGB维度问题。
    处理结果将直接覆盖原文件。
    """
    # 支持的文件扩展名，可根据需要添加
    valid_extensions = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.nii.gz', '.nii')
    
    # 获取目录下所有文件
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print(f"在目录 '{directory_path}' 中未找到支持的图像文件。")
        return

    print(f"找到 {len(files)} 个文件，开始处理...")
    success_count = 0

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        
        try:
            # 1. 读取图像
            img = sitk.ReadImage(file_path)
            img_npy = sitk.GetArrayFromImage(img)
            
            # 2. 修复 RGB 格式导致的维度问题
            # 如果是 2D 图像但数组是 3 维 (H, W, C)，说明是多通道（如 RGB）
            # 注意：对于 3D 医疗影像 (D, H, W)，SimpleITK 读取后 img.GetDimension() 为 3，
            # 这里我们主要针对 2D 标签误存为 RGB 的情况进行处理。
            if img.GetDimension() == 2 and img_npy.ndim == 3:
                # 取第一个通道 (R通道)，通常标签图三个通道值相同
                img_npy = img_npy[:, :, 0]
            
            # 3. 二值化映射
            # 创建一个新的 uint8 数组，全为 0
            seg_new = np.zeros_like(img_npy, dtype=np.uint8)
            # 将大于等于 128 的位置设为 1
            seg_new[img_npy >= 128] = 1
            
            # 4. 转回 SimpleITK 对象
            img_corr = sitk.GetImageFromArray(seg_new)
            
            # 5. 复制元数据 (非常重要，保持与原图一致的空间信息)
            img_corr.CopyInformation(img)
            # 或者手动设置：
            # img_corr.SetSpacing(img.GetSpacing())
            # img_corr.SetOrigin(img.GetOrigin())
            # img_corr.SetDirection(img.GetDirection())
            
            # 6. 覆盖保存
            sitk.WriteImage(img_corr, file_path)
            print(f"[成功] {filename}")
            success_count += 1
            
        except Exception as e:
            print(f"[失败] {filename}: {e}")

    print("-" * 30)
    print(f"处理完成。共处理: {success_count}/{len(files)}")

def process_all_labels(label_dir_in, label_dir_out, n_processes=12):
    """
    批量处理所有标签文件
    """
    os.makedirs(label_dir_out, exist_ok=True)
    label_files = subfiles(label_dir_in, suffix='.tif', join=False)
    
    print(f"Found {len(label_files)} label files to process...")
    
    # 准备参数列表
    args_list = [
        (join(label_dir_in, file), join(label_dir_out, file))
        for file in label_files
    ]
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        jobs = [pool.starmap_async(binary_label_mapping, [args]) for args in args_list]
        # 添加 tqdm 进度条并获取结果，如果有异常会在这里抛出
        _ = [job.get() for job in tqdm(jobs, desc="Processing labels")]
        
    print(f"Processed {len(label_files)} files. Output saved to: {label_dir_out}")

def convert_to_single_channel(input_folder):
    """
    将指定文件夹下的所有 RGB 图像转换为单通道图像。
    对于眼底图像（DRIVE），通常提取绿色通道（Green Channel）效果最好。
    如果你不确定，也可以改为简单的灰度转换 .convert('L')。
    """
    files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.tif', '.jpg', '.jpeg'))]
    
    if not files:
        print(f"在 {input_folder} 中没有找到图片文件。")
        return

    print(f"正在处理 {len(files)} 张图片...")
    
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        
        try:
            # 打开图片
            img = Image.open(file_path)
            
            # 检查通道数
            if img.mode == 'RGB':
                # 方案 A: 提取绿色通道 (推荐用于眼底血管分割)
                r, g, b = img.split()
                new_img = g
                
                # 方案 B: 如果你想用标准灰度，取消下面这行的注释，并注释掉上面的方案 A
                # new_img = img.convert('L')
                
                # 保存覆盖原文件 (或者你可以保存到新路径)
                new_img.save(file_path)
                print(f"[已修复] {filename}: RGB -> 单通道 (Green)")
            
            elif img.mode == 'L':
                print(f"[跳过] {filename} 已经是单通道")
            
            elif img.mode == 'RGBA':
                # 处理带透明通道的图
                r, g, b, a = img.split()
                new_img = g
                new_img.save(file_path)
                print(f"[已修复] {filename}: RGBA -> 单通道 (Green)")
                
        except Exception as e:
            print(f"[错误] 处理 {filename} 失败: {e}")
    
# 保存为 check_labels.py 然后运行 python check_labels.py
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib
from multiprocessing import Pool



def check_case(file_path):
    if file_path.endswith(".seg.npy"):
        data = np.load(file_path)
        # 假设 DRIVE 只有 0 和 1 两个类别
        if np.max(data) > 1: 
            return f"Error: {file_path} contains label {np.max(data)}"
        if np.min(data) < 0:
            return f"Error: {file_path} contains label {np.min(data)}"
    return None


import os

def rename_drive_files_auto(target_dir):
    """
    根据目录名称自动重命名 DRIVE 数据集文件
    
    逻辑判断:
    1. 若路径含 "images": 01_test.tif -> DRIVE_001_0000.tif
    2. 若路径含 "labels" 或 "mask": 01_manual1.gif -> DRIVE_001.gif
    
    参数:
        target_dir: 目标文件夹路径
    """
    
    # 1. 检查路径是否存在
    if not os.path.exists(target_dir):
        print(f"❌ 错误: 路径不存在 {target_dir}")
        return

    # 2. 判定处理模式 (Images vs Labels)
    dir_name_lower = target_dir.lower()
    
    if "images" in dir_name_lower:
        mode = "images"
        valid_exts = ('.tif', '.png', '.jpg') # 允许的原始图片格式
        out_ext = ".tif"
        print(f"📂 检测到 'images' 目录，将在重命名时添加 '_0000' 后缀并转换为 .tif")
        
    elif "labels" in dir_name_lower or "mask" in dir_name_lower:
        mode = "labels"
        valid_exts = ('.gif', '.png', '.tif') # 允许的原始标签格式
        out_ext = ".gif"
        print(f"📂 检测到 'labels' 目录，将在重命名时转换为 .gif")
        
    else:
        print(f"⚠️ 警告: 路径中未包含 'images' 或 'labels'，无法确定重命名规则。")
        print(f"当前路径: {target_dir}")
        return

    # 3. 获取文件并过滤
    # 过滤掉已经是目标格式的文件 (以 DRIVE_ 开头)，防止重复操作
    all_files = os.listdir(target_dir)
    raw_files = [f for f in all_files if f.lower().endswith(valid_exts) and not f.startswith("DRIVE_")]
    
    if not raw_files:
        print("⚠️ 未找到需要重命名的原始文件 (或文件已全部重命名完毕)")
        return

    # 4. 排序
    # DRIVE 数据通常是 01_test.tif, 02_test.tif，直接字符串排序即可
    try:
        raw_files.sort()
    except Exception as e:
        print(f"❌ 排序失败: {e}")
        return

    print(f"找到 {len(raw_files)} 个待处理文件。")
    
    count = 0
    
    # 5. 遍历并执行重命名
    for i, filename in enumerate(raw_files, start=1):
        
        # 获取文件扩展名 (如果需要保留原扩展名可使用，但这里需求是强制转格式)
        # _, ext = os.path.splitext(filename)
        
        # --- 根据模式构建新文件名 ---
        if mode == "images":
            # 格式: DRIVE_001_0000.tif
            new_filename = f"DRIVE_{i:03d}_0000{out_ext}"
        else:
            # 格式: DRIVE_001.gif
            new_filename = f"DRIVE_{i:03d}{out_ext}"
        
        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, new_filename)
        
        # 防止覆盖
        if os.path.exists(new_path):
            print(f"⚠️ 跳过: {new_filename} 已存在")
            continue

        try:
            os.rename(old_path, new_path)
            count += 1
            print(f"[OK] {filename} -> {new_filename}")
        except Exception as e:
            print(f"❌ 重命名失败 {filename}: {e}")

    print("-" * 40)
    print(f"✅ 处理完成！模式: {mode} | 共修改: {count} 个文件")

def batch_convert_gif_to_tif(directory_path):
    """
    将指定目录下的所有 .gif 文件转换为 .tif 文件。
    保留多帧 GIF 的所有帧。
    """
    # 检查路径是否存在
    if not os.path.isdir(directory_path):
        print(f"错误: 路径 '{directory_path}' 不存在。")
        return

    # 获取目录下所有文件
    files = os.listdir(directory_path)
    gif_files = [f for f in files if f.lower().endswith('.gif')]

    if not gif_files:
        print(f"在 '{directory_path}' 中未找到 .gif 文件。")
        return

    print(f"找到 {len(gif_files)} 个 GIF 文件，开始转换...")

    success_count = 0

    for filename in gif_files:
        gif_path = os.path.join(directory_path, filename)
        # 构建输出文件名 (将 .gif 替换为 .tif)
        tif_filename = os.path.splitext(filename)[0] + ".tif"
        tif_path = os.path.join(directory_path, tif_filename)

        try:
            with Image.open(gif_path) as img:
                # 处理多帧 GIF (动图)
                frames = []
                for frame in ImageSequence.Iterator(img):
                    # 复制帧并转换为 RGB (防止索引颜色模式导致的兼容性问题)
                    # 如果需要保留透明度，可以使用 'RGBA'
                    frames.append(frame.copy().convert('RGB'))
                
                # 保存为 TIF
                # save_all=True 确保保存所有帧
                # append_images 包含除第一帧以外的所有帧
                # compression 可以设置为 'tiff_lzw', 'tiff_deflate' 或 'raw' (无压缩)
                if len(frames) > 1:
                    frames[0].save(
                        tif_path, 
                        save_all=True, 
                        append_images=frames[1:], 
                        compression="tiff_lzw"
                    )
                else:
                    # 单帧 GIF
                    frames[0].save(tif_path, compression="tiff_lzw")

            print(f"[成功] {filename} -> {tif_filename}")
            success_count += 1

        except Exception as e:
            print(f"[失败] {filename}: {e}")

    print("-" * 30)
    print(f"处理完成。成功转换: {success_count}/{len(gif_files)}")



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

def check_npz_dimensions(pred_dir, gt_dir=None):
    """
    检查 .npz 文件的维度，并可选地与 GT 图片对比
    """
    print(f"🔍 正在检查目录: {pred_dir}")
    
    files = [f for f in os.listdir(pred_dir) if f.endswith('.npz')]
    if not files:
        print("❌ 未找到 .npz 文件！请检查路径或确认是否使用了 --save_probabilities")
        return

    files.sort()
    print(f"找到 {len(files)} 个 .npz 文件。显示前 10 个检查结果：\n")
    print(f"{'文件名':<20} | {'NPZ Shape (Cls, H, W)':<25} | {'GT Shape (H, W)':<20} | {'状态'}")
    print("-" * 85)

    for filename in files[:10]: # 只检查前10个，避免刷屏
        npz_path = os.path.join(pred_dir, filename)
        
        try:
            # 1. 读取 NPZ
            # nnU-Net 的 key 通常是 'probabilities'
            data = np.load(npz_path)
            if 'probabilities' in data:
                prob = data['probabilities']
                npz_shape = prob.shape # 通常是 (2, H, W)
            else:
                # 备用：打印所有 keys
                print(f"{filename:<20} | Key Error: {list(data.keys())}")
                continue

            # 2. 读取 GT (如果提供了目录)
            gt_shape_str = "N/A"
            status = "NPZ OK"
            
            if gt_dir:
                # 尝试匹配 GT 文件名 (假设主文件名一致)
                stem = filename.replace('.npz', '')
                # 尝试常见图片后缀
                gt_candidates = [f for f in os.listdir(gt_dir) if f.startswith(stem) and f.endswith(('.png', '.tif', '.gif'))]
                
                if gt_candidates:
                    gt_path = os.path.join(gt_dir, gt_candidates[0])
                    gt_img = Image.open(gt_path)
                    gt_shape = (gt_img.height, gt_img.width) # PIL size is (W, H), so shape is (H, W)
                    gt_shape_str = str(gt_shape)
                    
                    # 3. 核心对比逻辑
                    # npz_shape 通常是 (NumClasses, H, W)，所以比较后两位
                    # 注意：nnU-Net 输出可能是 (2, 512, 512) 而 GT 是 (584, 565)
                    if npz_shape[1:] == gt_shape:
                        status = "✅ 匹配"
                    else:
                        status = "❌ 尺寸不匹"
                else:
                    status = "GT 缺失"

            # 打印结果
            print(f"{filename:<20} | {str(npz_shape):<25} | {gt_shape_str:<20} | {status}")

        except Exception as e:
            print(f"{filename:<20} | 读取错误: {e}")

    print("-" * 85)
    if gt_dir:
        print("💡 提示：如果状态显示 '❌ 尺寸不匹'，说明 nnU-Net 对图像进行了重采样。")
        print("   在评估代码中必须使用 resize_pred_to_gt 函数强制对齐。")


    print(f"✅ 处理完成！成功转换并保存了 {count} 张图片到 '{output_dir}'")



# 使用示例
if __name__ == "__main__":
    
    imagesTs = "./nnUNet_raw/Dataset014_DRIVE/imagesTs"
    labelsTs = "./nnUNet_raw/Dataset014_DRIVE/labelsTs"

    predictions = "./nnUNet_raw/Dataset014_DRIVE/predictions/nnLoGoNet_SRL"
    masksTs = "./nnUNet_raw/Dataset014_DRIVE/masksTs"
    
    result_images = "./result_images/DRIVE/nnLoGoNet_SRL"
    
    convert_labels(predictions, result_images)
    
    # evaluate_final(labelsTs, predictions, masksTs)
    
    
    
    # 运行命令
    # python ./nnunetv2/dataset_conversion/Dataset014_DRIVE.py 
    # 