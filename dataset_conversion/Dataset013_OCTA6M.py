import shutil  
from batchgenerators.utilities.file_and_folder_operations import *  
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json  
from nnunetv2.paths import nnUNet_raw  
from skimage import io  
import numpy as np  
  
  
def copy_and_rename_files(source_dir: str, target_dir: str, start_idx: int,   
                          end_idx: int, prefix: str, counter: int, channel: str,   
                          file_extension: str = '.bmp'):  
    """  
    复制并重命名文件的通用函数  
      
    Args:  
        source_dir: 源文件目录  
        target_dir: 目标文件目录  
        start_idx: 起始索引(如10301)  
        end_idx: 结束索引(如10500)  
        prefix: 文件名前缀(如'OCTA3M')
        counter: 文件名中缀(如'001')  
        channel: 通道标识符(如'0000', '0001', 或 None 表示标注文件)  
        file_extension: 文件扩展名  
    """  
    maybe_mkdir_p(target_dir)  
        
    for file_idx in range(start_idx, end_idx + 1):  
        source_file = join(source_dir, f'{file_idx}{file_extension}')  
        
         
        
        if not isfile(source_file):  
            print(f'警告: 文件不存在 {source_file}')  
            continue  
              
        # 构建目标文件名  
        if channel is not None:  
            target_filename = f'{prefix}_{counter:03d}_{channel}{file_extension}'  
        else:  
            target_filename = f'{prefix}_{counter:03d}{file_extension}'  
              
        target_file = join(target_dir, target_filename)  
        shutil.copy(source_file, target_file)  
        print(f'已复制: {source_file} -> {target_file}')  
        counter += 1  
  
  
def process_labels(source_dir: str, target_dir: str, start_idx: int,   
                   end_idx: int, prefix: str, counter: int, file_extension: str = '.bmp'):  
    """  
    处理标注文件:复制、重命名并转换标签值  
      
    Args:  
        source_dir: 源文件目录  
        target_dir: 目标文件目录  
        start_idx: 起始索引  
        end_idx: 结束索引  
        prefix: 文件名前缀 
        counter: 文件名中缀 
        file_extension: 文件扩展名  
    """  
    maybe_mkdir_p(target_dir)  
       
    for file_idx in range(start_idx, end_idx + 1):  
        source_file = join(source_dir, f'{file_idx}{file_extension}')  
        
         
        
        if not isfile(source_file):  
            print(f'警告: 文件不存在 {source_file}')  
            continue  
              
        # 读取标注图像  
        seg = io.imread(source_file)  
          
        # 转换标签值:将255转换为1  
        if seg.dtype == bool:  
            seg = seg.astype(np.uint8)  
        else:  
            seg = seg.astype(np.uint8)  
            seg[seg == 255] = 1  
          
        # 确保只有0和1  
        seg = (seg > 0).astype(np.uint8)  
          
        # 保存  
        target_filename = f'{prefix}_{counter:03d}{file_extension}'  
        target_file = join(target_dir, target_filename)  
        io.imsave(target_file, seg, check_contrast=False)  
        print(f'已处理标注: {source_file} -> {target_file}')  
        counter += 1  
  
  
if __name__ == '__main__':  
    # 数据集配置  
    dataset_id = 13  
    dataset_name = f'Dataset{dataset_id:03d}_OCTA6M'  
      
    # 源路径  
    oct_source = '/work/imc_lab/ys_z/Snake-SWin-OCTA/datasets/OCTA-500/6M/ProjectionMaps/OCT(ILM_OPL)'  
    octa_source = '/work/imc_lab/ys_z/Snake-SWin-OCTA/datasets/OCTA-500/6M/ProjectionMaps/OCTA(ILM_OPL)'  
    gt_source = '/work/imc_lab/ys_z/Snake-SWin-OCTA/datasets/OCTA-500/6M/GT_LargeVessel'  
      
    # 目标路径  
    target_base = join(nnUNet_raw, dataset_name)  
    imagesTr = join(target_base, 'imagesTr')  
    imagesTs = join(target_base, 'imagesTs')  
    labelsTr = join(target_base, 'labelsTr')  
    labelsTs = join(target_base, 'labelsTs') 
    
    # 创建目录  
    maybe_mkdir_p(target_base)  
    maybe_mkdir_p(imagesTr)  
    maybe_mkdir_p(imagesTs)  
    maybe_mkdir_p(labelsTr)  
    maybe_mkdir_p(labelsTs)
    
    # 文件索引范围  
    start_idx = 10001  
    end_idx = 10300  
    total_files = end_idx - start_idx + 1  # 300个文件  
    train_count = 240  
    test_count = 60
    counter = 1  
      
    # 需求一: 处理OCT通道(0000) - 前240张到imagesTr,后60张到imagesTs  
    print("\n处理OCT通道(0000)...")  
    # 训练集  
    copy_and_rename_files(oct_source, imagesTr, start_idx, start_idx + train_count - 1,  
                         'OCTA6M', counter, '0000', '.bmp')  
    # 测试集  
    copy_and_rename_files(oct_source, imagesTs, start_idx + train_count, end_idx,  
                         'OCTA6M', counter + train_count, '0000', '.bmp')  
      
    # 需求二: 处理OCTA通道(0001) - 前240张到imagesTr,后60张到imagesTs  
    print("\n处理OCTA通道(0001)...")  
    # 训练集  
    copy_and_rename_files(octa_source, imagesTr, start_idx, start_idx + train_count - 1,  
                         'OCTA6M', counter, '0001', '.bmp')  
    # 测试集  
    copy_and_rename_files(octa_source, imagesTs, start_idx + train_count, end_idx,  
                         'OCTA6M', counter + train_count, '0001', '.bmp')  
      
    # 需求三: 处理标注文件 - 只有训练集的160张  
    print("\n处理标注文件...")  
    process_labels(gt_source, labelsTr, start_idx, start_idx + train_count - 1,  
                  'OCTA6M', counter, '.bmp')  
    process_labels(gt_source, labelsTs, start_idx + train_count, end_idx,  
                  'OCTA6M', counter + train_count, '.bmp')
    
    # 生成dataset.json  
    print("\n生成dataset.json...")  
    generate_dataset_json(  
        target_base,  
        channel_names={0: 'OCT', 1: 'OCTA'},  
        labels={'background': 0, 'vessel': 1},  
        num_training_cases=train_count,  
        file_ending='.bmp',  
        dataset_name=dataset_name  
    )  
      
    print(f"\n数据集转换完成!")  
    print(f"训练集: {train_count} 个案例")  
    print(f"测试集: {test_count} 个案例")