import os
from PIL import Image
import numpy as np

def convert_bmp_to_single_channel_tif(input_dir, output_dir):
    """
    将 RGB BMP 转换为 单通道 TIF (提取绿色通道)，并重命名为 _0000.tif
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有bmp文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.bmp')]
    files.sort()
    
    print(f"找到 {len(files)} 个文件，准备转换为单通道 TIF...")

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        
        try:
            # 1. 打开图片
            img = Image.open(file_path)
            
            # 2. 处理通道
            # 如果是 RGB，提取绿色通道 (通常眼底图绿色通道血管最清晰)
            if img.mode == 'RGB':
                r, g, b = img.split()
                target_img = g  # 选择绿色通道
            else:
                # 如果已经是灰度，直接用
                target_img = img

            # 3. 构建新文件名 (必须加 _0000)
            # 例如: DF600_001.bmp -> DF600_001_0000.tif
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}.tif"
            output_path = os.path.join(output_dir, new_filename)
            
            # 4. 保存为 TIF (不压缩或使用LZW，nnU-Net通常都能读)
            target_img.save(output_path, format='TIFF', compression='raw')
            
            # print(f"Converted: {filename} -> {new_filename}")

        except Exception as e:
            print(f"❌ 处理 {filename} 失败: {e}")

    print(f"✅ 完成！文件已保存在: {output_dir}")

if __name__ == '__main__':
    # -------- 请修改路径 --------
    input_folder = "/work/imc_lab/ys_z/nnUNet/nnUNet_raw/Dataset014_DRIVE/imagesTs" 
    output_folder = "/work/imc_lab/ys_z/nnUNet/nnUNet_raw/Dataset014_DRIVE/imagesTs_tif"
    # ---------------------------
    
    convert_bmp_to_single_channel_tif(input_folder, output_folder)