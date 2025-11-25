import os
import math
import shutil
import random
import pathlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from enum import Enum

import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from collections import defaultdict

import os
import cv2
import numpy as np
import torch


def split_paths_random(arr, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    改进的伪随机划分，结合多种随机化技术
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例总和必须为1"

    # 设置随机种子（确保可复现）
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 方法1：使用numpy的随机排列（比random.shuffle更随机）
    # indices = np.random.permutation(len(arr))
    # shuffled_paths = [arr[i] for i in indices]

    # 方法2：或者使用random.sample（无放回抽样）
    shuffled_paths = random.sample(arr, len(arr))

    n_total = len(shuffled_paths)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))

    test_paths = shuffled_paths[:n_test]
    val_paths = shuffled_paths[n_test:n_test + n_val]
    train_paths = shuffled_paths[n_test + n_val:]

    return train_paths, val_paths, test_paths

def yolo_split(src, dest):
    if not os.path.exists(src):
        print(src + ' not exits')
        exit(0)
    os.makedirs(dest, exist_ok=True)
    os.makedirs(dest + '/images', exist_ok=True)
    os.makedirs(dest + '/images/train', exist_ok=True)
    os.makedirs(dest + '/images/test', exist_ok=True)
    os.makedirs(dest + '/images/val', exist_ok=True)
    os.makedirs(dest + '/labels', exist_ok=True)
    os.makedirs(dest + '/labels/train', exist_ok=True)
    os.makedirs(dest + '/labels/test', exist_ok=True)
    os.makedirs(dest + '/labels/val', exist_ok=True)

    name_arr = [pathlib.Path(file).stem for file in os.listdir(src + '/images')]
    train_file, val_file, test_file = split_paths_random(name_arr, 0.7, 0.15, 0.15)

    for file in tqdm(train_file, desc='拷贝训练集'):
        shutil.copy(src + '/images/' + file + '.tiff', dest + '/images/train/' + file + '.tiff')
        shutil.copy(src + '/labels/' + file + '.txt', dest + '/labels/train/' + file + '.txt')
    for file in tqdm(val_file, desc='拷贝验证集'):
        shutil.copy(src + '/images/' + file + '.tiff', dest + '/images/val/' + file + '.tiff')
        shutil.copy(src + '/labels/' + file + '.txt', dest + '/labels/val/' + file + '.txt')
    for file in tqdm(test_file, desc='拷贝测试集'):
        shutil.copy(src + '/images/' + file + '.tiff', dest + '/images/test/' + file + '.tiff')
        shutil.copy(src + '/labels/' + file + '.txt', dest + '/labels/test/' + file + '.txt')




ct_path = 'S:/文档/数据集/luna16/dataset'
# ct_path = 'I:/dataset'
mask_path = 'S:/文档/数据集/luna16/seg-lungs-LUNA16'  # 掩码文件路径
info_path = 'S:/文档/数据集/luna16/'
annotations_file = pd.read_csv(info_path + '/annotations.csv')
candidates_file = pd.read_csv(info_path + '/candidates_V2.csv')
#
# save_path = 'S:/文档/数据集/luna16/简单假阳性'

# ct_path = '/run/media/hexne/SSS/文档/数据集/luna16/dataset/'
# mask_path = 'S:/文档/数据集/luna16/seg-lungs-LUNA16'  # 掩码文件路径
# info_path = '/run/media/hexne/SSS/文档/数据集/luna16/'
# annotations_file = pd.read_csv(info_path + '/annotations.csv')
# candidates_file = pd.read_csv(info_path + '/candidates_V2.csv')

save_path = ''
save_images_path = save_path + '/images'
save_label_path = save_path + '/labels'

# 肺窗口
window_low = -1000
window_high = 600
window_width = abs(window_high - window_low)


class Mod(Enum):
    only_center = 1  # 结节中心所在切片
    center_1 = 2  # 结节中心所在切片及上下一张
    center_2 = 3  # 结节中心所在切片及上下两张
    center_3 = 4
    only_nodule = 5  # 结节所有切片
    all_ct = 6  # 完整的CT切片
    c1 = 7
    c2 = 8
    c3 = 9
    c4 = 10
    c5 = 11
    c6 = 12
    c7 = 13
    c8 = 14
    c9 = 15
    c10 = 16



def save_labels(labels, path, name):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}.txt")

    if labels[0] is None:
        open(file_path, 'w').close()
        return

    with open(file_path, 'w', encoding='utf-8') as f:
        for label in labels:
            type, rect = label
            f.write(f"{type} {rect[0]:.6f} {rect[1]:.6f} {rect[2]:.6f} {rect[3]:.6f}\n")

# return [ (index, rx_px, ry_px) , ... ]
def get_slice_range(center_slice, diameter_mm, spacing, max_slices, mod, have_label, padding_mm = 0):
    spacing_x, spacing_y, spacing_z = spacing
    r = (diameter_mm + padding_mm) / 2
    """根据模式获取切片范围"""
    if mod == Mod.center_1:
        start = max(0, center_slice - 1)
        end = min(max_slices, center_slice + 2)
    elif mod == Mod.center_2:
        start = max(0, center_slice - 2)
        end = min(max_slices, center_slice + 3)
    elif mod == Mod.center_3:
        start = max(0, center_slice - 3)
        end = min(max_slices, center_slice + 4)
    elif mod == Mod.only_nodule:
        tmp = int(r / spacing_z)
        start = max(0, center_slice - tmp)
        end = min(max_slices, center_slice + tmp + 1)
    elif mod == Mod.only_center:
        start = center_slice
        end = center_slice + 1
    else:
        start = 0
        end = max_slices

    ret = []
    for index in range(start, end):
        rx_px = None
        ry_px = None
        if have_label:
            z_mm = abs(index - center_slice) * spacing_z
            r_mm_squared = r ** 2 - z_mm ** 2  # r² = R² - z²
            r_mm = np.sqrt(r_mm_squared)  # 当前切面的实际半径
            rx_px = math.ceil(r_mm / spacing_x)
            ry_px = math.ceil(r_mm / spacing_y)
        ret.append((index, rx_px, ry_px))

    return ret

def load_mask(mask_file_path):
    """加载掩码文件"""
    if os.path.exists(mask_file_path):
        mask_info = sitk.ReadImage(mask_file_path)
        mask_array = sitk.GetArrayFromImage(mask_info)
        return mask_array
    return None

def apply_mask(ct_image, mask, mask_value=window_low):
    """应用掩码到CT图像"""
    if mask is not None:
        masked_image = np.where(mask > 0, ct_image, mask_value)
        return masked_image
    return ct_image

import os
import numpy as np
import torch
from PIL import Image
import tifffile

import os
import numpy as np
import torch
import tifffile

def save_image(ct_image, path, name, mask=None,
               window_low=-1000, window_high=400):
    os.makedirs(path, exist_ok=True)

    # 转 numpy
    if isinstance(ct_image, torch.Tensor):
        ct_image = ct_image.numpy()

    # [C, H, W] → [H, W, C]
    if ct_image.ndim == 3 and ct_image.shape[0] <= 10:
        ct_image = np.transpose(ct_image, (1, 2, 0))

    # 单通道图像扩展维度
    if ct_image.ndim == 2:
        ct_image = np.expand_dims(ct_image, axis=-1)  # [H, W] → [H, W, 1]

    # 掩码叠加（仅支持单通道）
    if mask is not None and ct_image.shape[-1] == 1:
        ct_image[..., 0] = apply_mask(ct_image[..., 0], mask)

    # 窗位处理
    window_width = window_high - window_low
    processed = []
    for c in range(ct_image.shape[-1]):
        ch = ct_image[..., c]
        ch = np.clip(ch, window_low, window_high)
        ch = ((ch - window_low) / window_width) * 255
        processed.append(ch.astype(np.uint8))
    image = np.stack(processed, axis=-1)  # [H, W, C]

    # 保存为 .tif（支持任意通道）
    filepath = os.path.join(path, f"{name}.tiff")
    tifffile.imwrite(filepath, image)



def ct_have_nodule(name):
    return not annotations_file[annotations_file['seriesuid'] == name].empty

# 判断假结节和真结节之间的间距
def have_distance_list(name, limit_distance):
    ret = []
    # 获取当前seriesuid对应的真结节和假结节
    true_list = annotations_file[annotations_file['seriesuid'] == name]
    false_list = candidates_file[candidates_file['seriesuid'] == name]

    for _, false_nodule in false_list.iterrows():
        flag = True
        for _, true_nodule in true_list.iterrows():
            distance = limit_distance + true_nodule['diameter_mm']
            dx = true_nodule['coordX'] - false_nodule['coordX']
            dy = true_nodule['coordY'] - false_nodule['coordY']
            dz = true_nodule['coordZ'] - false_nodule['coordZ']
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if dist <= distance:
                flag = False
        if flag:
            ret.append(false_nodule)
    return ret


def 获取简单假阳性(img_lab_map, count):
    remove_count = int(count * 0.3)

    # 创建删除前后30%后的数组
    ret = list(range(remove_count, count - remove_count))

    # 创建要删除的所有位置的集合
    indices_to_remove = set()

    for index, _ in img_lab_map.items():
        # 添加 [index-5, index+5] 范围内的所有数字
        start = max(remove_count, index - 10)  # 确保不小于删除后的最小值
        end = min(count - remove_count - 1, index + 10)  # 确保不大于删除后的最大值
        for i in range(start, end + 1):
            indices_to_remove.add(i)

    # 过滤掉所有要删除的位置
    ret = [x for x in ret if x not in indices_to_remove]
    random.shuffle(ret)
    return ret[:len(img_lab_map)]

def foreach_ct_file(mod, enable_mask):
    global save_path

    files_name = [Path(file).stem for file in os.listdir(ct_path)
                  if file.endswith('.mhd') and ct_have_nodule(Path(file).stem)]
    for ct_file_name in tqdm(files_name, desc='处理CT图像'):

        ct_file = f"{ct_path}/{ct_file_name}.mhd"
        ct_info = sitk.ReadImage(ct_file)
        images = sitk.GetArrayFromImage(ct_info)
        origin = ct_info.GetOrigin()  # [x, y, z]
        spacing = ct_info.GetSpacing()  # [x, y, z]

        cur_ct_nodules = annotations_file[annotations_file['seriesuid'] == ct_file_name]

        # (slice_index, (type, rect))
        img_lab_map = defaultdict(list)
        # 处理真结节，对一个CT中的某一个结节
        true_count = 0
        for _, cur_nodule in cur_ct_nodules.iterrows():
            world_point = [cur_nodule['coordX'], cur_nodule['coordY'], cur_nodule['coordZ']]
            pixel_point = ct_info.TransformPhysicalPointToIndex(world_point)
            center_x_px, center_y_px, center_z_index = pixel_point
            if mod == Mod.center_1 or mod == Mod.center_2 or mod == Mod.center_3\
                    or mod == mod.c1 or mod == mod.c2 or mod == mod.c3 or mod == mod.c4 or mod == mod.c5 or mod == mod.c6\
                    or mod == mod.c7 or mod == mod.c8 or mod == mod.c9 or mod == mod.c10:
                nodule_index_range = get_slice_range(center_z_index, cur_nodule['diameter_mm'], spacing, images.shape[0], Mod.only_center, True, 0)
            else:
                nodule_index_range = get_slice_range(center_z_index, cur_nodule['diameter_mm'], spacing, images.shape[0], mod, True, 0)

            # 对于单个结节的多个切片
            for cur_nodule_index, rw_px, rh_px in nodule_index_range:
                w = rw_px * 2
                h = rh_px * 2
                rect = [
                            center_x_px / images.shape[2],
                            center_y_px / images.shape[1],
                            w / images.shape[2],
                            h / images.shape[1]
                        ]
                # img_lab_map[cur_nodule_index].append((300_基准, rect))
                img_lab_map[cur_nodule_index].append((0, rect))
                true_count += 1

        # 处理假结节, 从文件中的读取的候选
        # cur_false_nodules = have_distance_list(ct_file_name, 10)
        # for index in range(0, true_count):
        #     nodule = cur_false_nodules[index]
        #     world_point = [nodule['coordX'], nodule['coordY'], nodule['coordZ']]
        #     pixel_point = ct_info.TransformPhysicalPointToIndex(world_point)
        #     center_x_px, center_y_px, center_z_index = pixel_point
        #     nodule_index_range = get_slice_range(center_z_index, 10, spacing, images.shape[0], Mod.only_center, False, 0)
        #     for slice_nodule_index, rw_px, rh_px in nodule_index_range:
        #         if rw_px is not None and rh_px is not None:
        #             w = rw_px * 2
        #             h = rh_px * 2
        #             rect = [
        #                 center_x_px / images.shape[2],
        #                 center_y_px / images.shape[300_基准],
        #                 w / images.shape[2],
        #                 h / images.shape[300_基准]
        #             ]
        #             img_lab_map[slice_nodule_index].append((0, rect))
        #         else:
        #             img_lab_map[slice_nodule_index].append(None)
        #

        # 简单假阳性
        # cur_false_nodules = 获取简单假阳性(img_lab_map, images.shape[0])
        # for index in cur_false_nodules:
        #     img_lab_map[index].append(None)

        # 处理这个图片上被选取的所有结节
        file_index = 0
        for index, labs in img_lab_map.items():
            current_mask = None
            if enable_mask:
                mask_file = os.path.join(mask_path, f"{ct_file_name}.mhd")
                mask_array = load_mask(mask_file)
                if mask_array is None:
                    print(f"警告: 未找到掩码文件 {mask_file}")
                current_mask = mask_array[index] if mask_array is not None else None
            if mod == Mod.c1:
                image = torch.stack([
                    torch.from_numpy(images[index]).float(),
                ], dim=0)  # [3, H, W]
            elif mod == Mod.c2:
                image = torch.stack([
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index + 1]).float()
                ], dim=0)  # [3, H, W]
            elif mod == Mod.c3:
                image = torch.stack([
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index]).float(),
                    torch.from_numpy(images[index + 1]).float()
                ], dim=0)  # [3, H, W]
            elif mod == Mod.c4:
                image = torch.stack([
                    torch.from_numpy(images[index - 2]).float(),
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index + 1]).float(),
                    torch.from_numpy(images[index + 2]).float()
                ], dim=0)  # [5, H, W]
            elif mod == Mod.c5:
                image = torch.stack([
                    torch.from_numpy(images[index - 2]).float(),
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index]).float(),
                    torch.from_numpy(images[index + 1]).float(),
                    torch.from_numpy(images[index + 2]).float(),
                ], dim=0)  # [5, H, W]
            elif mod == Mod.c6:
                image = torch.stack([
                    torch.from_numpy(images[index - 3]).float(),
                    torch.from_numpy(images[index - 2]).float(),
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index + 1]).float(),
                    torch.from_numpy(images[index + 2]).float(),
                    torch.from_numpy(images[index + 3]).float()
                ], dim=0)  # [5, H, W]
            elif mod == Mod.c7:
                image = torch.stack([
                    torch.from_numpy(images[index - 3]).float(),
                    torch.from_numpy(images[index - 2]).float(),
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index]).float(),
                    torch.from_numpy(images[index + 1]).float(),
                    torch.from_numpy(images[index + 2]).float(),
                    torch.from_numpy(images[index + 3]).float(),
                ])
            elif mod == Mod.c8:
                image = torch.stack([
                    torch.from_numpy(images[index - 4]).float(),
                    torch.from_numpy(images[index - 3]).float(),
                    torch.from_numpy(images[index - 2]).float(),
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index + 1]).float(),
                    torch.from_numpy(images[index + 2]).float(),
                    torch.from_numpy(images[index + 3]).float(),
                    torch.from_numpy(images[index + 4]).float()
                ], dim=0)  # [5, H, W]
            elif mod == Mod.c9:
                image = torch.stack([
                    torch.from_numpy(images[index - 4]).float(),
                    torch.from_numpy(images[index - 3]).float(),
                    torch.from_numpy(images[index - 2]).float(),
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index]).float(),
                    torch.from_numpy(images[index + 1]).float(),
                    torch.from_numpy(images[index + 2]).float(),
                    torch.from_numpy(images[index + 3]).float(),
                    torch.from_numpy(images[index + 4]).float()
                ], dim=0)  # [5, H, W]
            elif mod == Mod.c10:
                image = torch.stack([
                    torch.from_numpy(images[index - 5]).float(),
                    torch.from_numpy(images[index - 4]).float(),
                    torch.from_numpy(images[index - 3]).float(),
                    torch.from_numpy(images[index - 2]).float(),
                    torch.from_numpy(images[index - 1]).float(),
                    torch.from_numpy(images[index + 1]).float(),
                    torch.from_numpy(images[index + 2]).float(),
                    torch.from_numpy(images[index + 3]).float(),
                    torch.from_numpy(images[index + 4]).float(),
                    torch.from_numpy(images[index + 5]).float()
                ], dim=0)  # [5, H, W]

            save_image(image, save_images_path, f"{ct_file_name}_{file_index}", None)
            # save_image(images[index], save_images_path, f"{ct_file_name}_{file_index}", current_mask)
            save_labels(labs, save_label_path, f"{ct_file_name}_{file_index}")
            file_index += 1


def batch_ct_files(mod, sp):
    global save_path, save_images_path, save_label_path
    save_path = sp
    save_images_path = save_path + '/images'
    save_label_path = save_path + '/labels'

    if not os.path.exists(ct_path) or not os.path.exists(info_path):
        print('CT影像路径或信息文件不存在')
        exit(0)

    os.makedirs(save_images_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)


    foreach_ct_file(mod, False)


if __name__ == '__main__':
    mod = Mod.c1
    src_path = './1_tiff'
    to_path = './1'
    batch_ct_files(mod, src_path)
    yolo_split(src_path, to_path)

    # 生成掩码('300_基准', '300_基准')