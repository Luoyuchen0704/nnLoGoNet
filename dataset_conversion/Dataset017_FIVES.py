import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

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
    # 过滤掉已经重命名过的文件 (以 FIVES_ 开头)，防止重复执行导致混乱
    all_files = [f for f in os.listdir(target_dir) if f.endswith('.png')]
    raw_files = [f for f in all_files if not f.startswith("FIVES_")]
    
    if not raw_files:
        print("⚠️ 未找到需要重命名的原始文件 (或文件已全部重命名)")
        return

    # 3. 自然排序 (关键)
    # 提取文件名开头的数字: '1_A.png' -> 1, '10_A.png' -> 10
    try:
        raw_files.sort(key=lambda x: int(re.match(r'(\d+)', x).group(1)))
    except Exception as e:
        print(f"❌ 排序失败，文件名格式可能不统一: {e}")
        return

    print(f"找到 {len(raw_files)} 个原始文件，准备原地重命名...")
    print(f"  首个文件: {raw_files[0]}")
    print(f"  末尾文件: {raw_files[-1]}")

    count = 0
    # 4. 遍历并执行重命名
    for i, filename in enumerate(raw_files, start=1):
        
        # 限制只处理前 600 个 (如果需要)
        if i > 600:
            break

        # 构建新文件名: FIVES_001_0000.png
        new_filename = f"FIVES_{i:03d}.png"
        
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
    valid_extensions = ('.png', '.bmp')
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

if __name__ == '__main__':
    # ================= 配置路径 =================
    # 请直接填入包含图片的文件夹路径
    # 例如: "/path/to/FIVES/Original" 或 "/path/to/FIVES/GroundTruth"
    file_path = "./nnUNet_raw/Dataset017_FIVES/imagesTs_DF600" 
    # ===========================================

    convert_images_to_grayscale_inplace(file_path, method='green')