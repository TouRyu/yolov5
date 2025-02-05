import os
import shutil
import argparse
import random
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="随机划分图像数据集并按照 YOLOv5 格式组织数据")
    
    # images_dir: 存放 jpg 图像的目录
    parser.add_argument(
        '--images_dir', 
        type=str, 
        required=True, 
        help="存放 jpg 图像的文件夹路径，例如：/path/to/images"
    )
    # labels_dir: 存放 txt 标签的目录
    parser.add_argument(
        '--labels_dir', 
        type=str, 
        required=True, 
        help="存放对应标签的文件夹路径，例如：/path/to/labels"
    )
    # train_ratio: 训练集比例，默认为0.8
    parser.add_argument(
        '--train_ratio', 
        type=float, 
        default=0.8, 
        help="训练集比例（0-1之间的小数），默认值为 0.8"
    )
    # seed: 随机种子，确保每次划分结果一致，默认值为 42
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="随机种子，默认为42"
    )
    
    args = parser.parse_args()
    return args

def create_dir_structure(base_dir):
    """
    在 base_dir 下创建 YOLOv5 所需的文件夹结构：
    base_dir/train/images, base_dir/train/labels, base_dir/valid/images, base_dir/valid/labels
    """
    dirs = [
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'train', 'labels'),
        os.path.join(base_dir, 'valid', 'images'),
        os.path.join(base_dir, 'valid', 'labels')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"创建或确认目录：{d}")
    return dirs

def split_and_copy(images_dir, labels_dir, train_ratio, seed):
    """
    功能：
        - 读取 images_dir 中的所有 jpg 文件。
        - 对每个图像，检查 labels_dir 中是否存在对应的 txt 文件。
        - 根据 train_ratio 将数据集随机划分为训练集和验证集。
        - 将划分后的图像和标签文件复制到相应的目录中。
    """
    random.seed(seed)  # 设置随机种子，确保划分结果可重复

    # 使用 glob 查找所有 jpg 文件（不区分大小写）
    pattern = os.path.join(images_dir, '*.jpg')
    image_files = glob(pattern)
    # 补充考虑 jpeg 格式的情况，如果有需要的话，也可以加入：
    image_files += glob(os.path.join(images_dir, '*.jpeg'))
    image_files = sorted(image_files)
    
    print(f"在 {images_dir} 中找到 {len(image_files)} 个图像文件。")
    
    # 过滤掉那些在 labels_dir 中没有对应标签的图像
    valid_pairs = []
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base_name + '.txt')
        if os.path.exists(label_path):
            valid_pairs.append( (img_path, label_path) )
        else:
            print(f"警告：图像 {img_path} 没有对应的标签文件 {label_path}，将被跳过。")
    
    print(f"有效的数据对数量：{len(valid_pairs)}")
    
    # 根据 train_ratio 划分数据集
    total = len(valid_pairs)
    train_count = int(total * train_ratio)
    # 随机打乱数据
    random.shuffle(valid_pairs)
    train_pairs = valid_pairs[:train_count]
    valid_pairs_ = valid_pairs[train_count:]
    
    print(f"划分训练集数量：{len(train_pairs)}，验证集数量：{len(valid_pairs_)}")
    
    # 确定输出根目录：images_dir 的同级目录，即 images_dir 的父目录
    base_dir = os.path.dirname(images_dir)
    print(f"输出数据将保存在：{base_dir} 下的 train/ 和 valid/ 子目录中。")
    
    # 创建目录结构
    train_images_dir = os.path.join(base_dir, 'train', 'images')
    train_labels_dir = os.path.join(base_dir, 'train', 'labels')
    valid_images_dir = os.path.join(base_dir, 'valid', 'images')
    valid_labels_dir = os.path.join(base_dir, 'valid', 'labels')
    create_dir_structure(base_dir)
    
    # 定义一个内部函数，用来复制图像和标签
    def copy_data(pairs, img_out_dir, label_out_dir, set_name=""):
        for img_path, label_path in pairs:
            # 复制图像
            img_filename = os.path.basename(img_path)
            dest_img_path = os.path.join(img_out_dir, img_filename)
            shutil.copy2(img_path, dest_img_path)
            # 复制标签
            label_filename = os.path.basename(label_path)
            dest_label_path = os.path.join(label_out_dir, label_filename)
            shutil.copy2(label_path, dest_label_path)
            print(f"[{set_name}] 已复制：{img_filename} 及对应标签 {label_filename}")
    
    print("开始复制训练集文件...")
    copy_data(train_pairs, train_images_dir, train_labels_dir, set_name="Train")
    
    print("开始复制验证集文件...")
    copy_data(valid_pairs_, valid_images_dir, valid_labels_dir, set_name="Valid")
    
    print("数据划分和复制完成。")

def main():
    args = parse_args()
    
    # 对输入路径进行绝对路径转换
    images_dir = os.path.abspath(args.images_dir)
    labels_dir = os.path.abspath(args.labels_dir)
    
    # 检查输入目录是否存在
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"指定的 images_dir 不存在：{images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"指定的 labels_dir 不存在：{labels_dir}")
    
    print("参数设置：")
    print(f"  images_dir: {images_dir}")
    print(f"  labels_dir: {labels_dir}")
    print(f"  train_ratio: {args.train_ratio}")
    print(f"  seed: {args.seed}")
    
    split_and_copy(images_dir, labels_dir, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()