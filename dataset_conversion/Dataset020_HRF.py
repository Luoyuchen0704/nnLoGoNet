import os
import re
import numpy as np
from tqdm import tqdm
import random
import shutil
import json

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
    all_files = [f for f in os.listdir(target_dir) if f.endswith('.tif')]
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
        if i > 100:
            break

        # 构建新文件名: CHASE-DB1_001.jpg
        if 'images' in target_dir:
            new_filename = f"HRF_{i:03d}_0000.tif"
        elif 'labels' in target_dir:
            new_filename = f"HRF_{i:03d}.tif"
        elif 'masks' in target_dir:
            new_filename = f"HRF_{i:03d}_mask.tif"

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

    files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    files.sort()
    
    if not files:
        print("⚠️ 目录下没有找到 .tif 文件")
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
    files = [f for f in os.listdir(target_dir) if f.endswith('.tif')]
    
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
    valid_extensions = ('.png', '.bmp', '.jpg', '.tif')
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

def batch_jpg_to_tif(folder_path, use_compression=True):
    """
    遍历指定文件夹，将所有的 .jpg/.JPG 图片转换为 .tif 格式。
    
    Args:
        folder_path (str): 文件夹路径
        use_compression (bool): 是否使用 LZW 无损压缩 (建议开启，否则文件会巨大)
    """
    if not os.path.exists(folder_path):
        print(f"路径不存在: {folder_path}")
        return

    count = 0
    # 支持的源文件后缀
    valid_exts = ('.jpg', '.jpeg') 

    for filename in os.listdir(folder_path):
        # lower() 确保同时匹配 .jpg, .JPG, .jpeg
        if filename.lower().endswith(valid_exts):
            src_path = os.path.join(folder_path, filename)
            
            # 生成 .tif 文件名
            name_pure = os.path.splitext(filename)[0]
            dst_path = os.path.join(folder_path, name_pure + ".tif")
            
            try:
                with Image.open(src_path) as img:
                    # 转换参数设置
                    save_kwargs = {'format': 'TIFF'}
                    
                    if use_compression:
                        # tiff_lzw 是通用的无损压缩算法
                        save_kwargs['compression'] = 'tiff_lzw'
                    
                    # 保存
                    img.save(dst_path, **save_kwargs)
                    count += 1
                    print(f"转换成功: {filename} -> .tif")
                    
            except Exception as e:
                print(f"转换失败 {filename}: {e}")

    print(f"\n处理完成，共转换 {count} 张图片。")

def delete_jpg_files(directory_path):
    """
    删除指定目录下的所有 .jpg/.JPG 文件 (不区分大小写)。
    
    参数:
    directory_path (str): 目标文件夹路径
    """
    # 1. 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"❌ 错误: 路径 '{directory_path}' 不存在。")
        return

    # 2. 获取目录下所有文件
    files = os.listdir(directory_path)
    deleted_count = 0
    
    print(f"📂 正在扫描目录: {directory_path} ...")

    for filename in files:
        # 3. 检查后缀名 (关键点：转为小写对比)
        # 如果你还想删除 .jpeg，可以写成: .endswith(('.jpg', '.jpeg'))
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # 4. 执行删除
                os.remove(file_path)
                print(f"🗑️ [已删除] {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ [删除失败] {filename}: {e}")

    print("-" * 30)
    if deleted_count == 0:
        print("✅ 未找到 .jpg 文件。")
    else:
        print(f"✅ 操作完成。共清理了 {deleted_count} 个文件。")

def create_fixed_split(preprocessed_dir, val_ratio=0.2):
    """
    创建固定的训练/验证划分
    preprocessed_dir: 预处理后的数据目录
    val_ratio: 验证集比例 (例如 0.2 表示 20% 做验证)
    """
    
    # 1. 找到所有训练样本的 Case ID
    # nnU-Net 预处理后的文件名通常是 CaseID.npz 或 CaseID.npy
    # 我们直接读取 gt_segmentations 文件夹里的文件名最稳
    gt_dir = os.path.join(preprocessed_dir, "gt_segmentations")
    if not os.path.exists(gt_dir):
        print(f"❌ 错误：找不到路径 {gt_dir}，请确认是否已运行预处理。")
        return

    # 获取所有 Case ID (去掉 .nii.gz 或 .png 后缀)
    files = [f for f in os.listdir(gt_dir) if f.endswith('.nii.gz') or f.endswith('.png')]
    case_ids = sorted([os.path.splitext(f)[0] for f in files])
    
    print(f"找到 {len(case_ids)} 个训练样本。")

    # 2. 随机打乱 (设定种子保证可复现)
    np.random.seed(12345) 
    np.random.shuffle(case_ids)

    # 3. 划分 Train 和 Val
    num_val = int(len(case_ids) * val_ratio)
    val_patients = case_ids[:num_val]
    train_patients = case_ids[num_val:]
    
    # 打印确认
    print(f"训练集 ({len(train_patients)}): {train_patients[:5]} ...")
    print(f"验证集 ({len(val_patients)}): {val_patients}")

    # 4. 构建 split 字典
    # nnU-Net 的 split 格式是一个列表，列表第 0 项代表 "Fold 0"
    split_dict = [
        {
            "train": train_patients,
            "val": val_patients
        },
        # 如果你只跑 Fold 0，后面这些其实无所谓，留空或者凑数都行
        # 但为了格式完整，通常只放一个 dict 进去就行，或者复制 5 份
    ] * 5 

    # 5. 保存为 splits_final.json
    output_path = os.path.join(preprocessed_dir, "splits_final.json")
    
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=4)
    
    print(f"✅ 自定义划分已保存至: {output_path}")
    print("现在你可以使用 'nnUNetv2_train ... 0' 来训练了！")

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
    imagesTr = "./nnUNet_raw/Dataset020_HRF/imagesTr"
    labelsTr = "./nnUNet_raw/Dataset020_HRF/labelsTr"
    
    imagesTs = "./nnUNet_raw/Dataset020_HRF/imagesTs"
    labelsTs = "./nnUNet_raw/Dataset020_HRF/labelsTs"

    predictions = "./nnUNet_raw/Dataset020_HRF/predictions/nnLoGoNet_SRL"
    masksTs = "./nnUNet_raw/Dataset020_HRF/masksTs"
    masksTr = "./nnUNet_raw/Dataset020_HRF/masksTr"
    
    result_images = "./result_images/HRF/nnLoGONet_SRL"
    # convert_images_to_grayscale_inplace(imagesTs, method='green')

    preprocessed_dir = "./nnUNet_preprocessed/Dataset020_HRF"
    # create_fixed_split(preprocessed_dir, val_ratio=0.2)

    evaluate_final(labelsTs, predictions, masksTs)
    convert_labels(predictions, result_images)
    # python ./nnunetv2/dataset_conversion/Dataset020_HRF.py