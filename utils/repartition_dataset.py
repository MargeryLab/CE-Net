# repartition_dataset.py
import os
import math
import random

def partition_data(dataset_dir, ouput_root):
    """
    Divide the raw data into training sets and validation sets
    :param dataset_dir: path root of dataset
    :param ouput_root: the root path to the output file
    :return:
    """
    image_names = []
    mask_names = []
    val_size = 0.11
    train_names = []
    val_names = []

    for file in os.listdir(os.path.join(dataset_dir, "imgs")):
        image_names.append(file)
        image_names.sort()
    for file in os.listdir(os.path.join(dataset_dir, "masks_vis")):
        mask_names.append(file)
        mask_names.sort()

    rawdata_size = len(image_names)
    random.seed(361)
    val_indices = random.sample(range(0, rawdata_size), math.floor(rawdata_size * val_size))
    train_indices = []
    for i in range(0, rawdata_size):
        if i not in val_indices:
            train_indices.append(i)

    with open(os.path.join(ouput_root, 'val.txt'), 'w') as f:
        for i in val_indices:
            val_names.append(image_names[i])
            f.write(image_names[i])
            f.write('\n')

    with open(os.path.join(ouput_root, 'train.txt'), 'w') as f:
        for i in train_indices:
            train_names.append(image_names[i])
            f.write(image_names[i])
            f.write('\n')
    train_names.sort(), val_names.sort()
    return train_names, val_names


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'imgs')
        mask_path = os.path.join(root, 'masks_vis')

        if 'Augdata' in root:
            data_list = os.listdir(os.path.join(root, 'Images'))
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'test':
        img_path = os.path.join(root, 'imgs')
        mask_path = os.path.join(root, 'masks_vis')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    else:
        pass
    return items

if __name__ == '__main__':
    dataset_dir = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/Data/data_png_png'
    output_root = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/Data/data_png_png'
    train_names,  val_names = partition_data(dataset_dir, output_root)
    print(len(train_names))
    print(len(val_names))
    # root_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/Data/data_png_png'
    # make_dataset(root_path, 'train')
