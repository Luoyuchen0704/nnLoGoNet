# import os  
# from batchgenerators.utilities.file_and_folder_operations import *  
  
# # 批量重命名文件  
# for num in range(1, 56):  
#     old_name = f'{num}.png'  
#     new_name = f'SS_{num:03d}.png'  
      
#     # 检查文件是否存在  
#     if os.path.exists(old_name):  
#         os.rename(old_name, new_name)  
#         print(f'已重命名: {old_name} -> {new_name}')  
#     else:  
#         print(f'文件不存在: {old_name}')
from skimage import io  
import numpy as np  
from batchgenerators.utilities.file_and_folder_operations import *  
  
labelsTr_dir = '/work/imc_lab/ys_z/nnUNet/nnUNet_raw/Dataset011_SS/labelsTr'  
label_files = subfiles(labelsTr_dir, suffix='.png', join=True)  
  
for label_file in label_files:  
    # 读取标注图像  
    seg = io.imread(label_file)  
      
    # 将布尔类型或 255 值转换为 1  
    # 首先转换为 uint8 类型  
    if seg.dtype == bool:  
        seg = seg.astype(np.uint8)  
    else:  
        seg = seg.astype(np.uint8)  
        seg[seg == 255] = 1  
      
    # 确保只有 0 和 1  
    seg = (seg > 0).astype(np.uint8)  
      
    # 保存,明确指定为 uint8  
    io.imsave(label_file, seg, check_contrast=False)  
    print(f'已转换: {label_file}')