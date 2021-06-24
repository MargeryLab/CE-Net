import csv

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import sklearn.metrics as metrics
from utils.metrics import *
from sklearn.metrics import roc_curve, auc
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import helpers
from utils.metrics import *

from time import time
from PIL import Image

import warnings
from loss import dice_bce_loss
from networks.cenet import CE_Net_

metrics_path = 'metrics.csv'
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
palette = [[0], [127], [255]]

BATCHSIZE_PER_CARD = 16

def onehot(data, n):
    palette = [[0], [127], [255]]
    """onehot ecoder"""
    # buf = np.zeros(data.shape + (n,))
    # nmsk = np.arange(data.size) * n + data.ravel()  #創建等差數組
    # buf.ravel()[nmsk - 1] = 1
    # return buf
    semantic_map = []
    for colour in palette:
        equality = np.equal(data, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map

def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()

    label_1D = label_1D / 255.
    try:
        auc = metrics.roc_auc_score(label_1D, result_1D)
        # print("AUC={0:.4f}".format(auc))
    except ValueError:
        return None

    return auc

# def accuracy(pred_mask, label):
#     '''
#     acc=(TP+TN)/(TP+FN+TN+FP)
#     '''
#     pred_mask = pred_mask.astype(np.uint8)
#     TP, FN, TN, FP = [0, 0, 0, 0]
#     for i in range(label.shape[0]):
#         for j in range(label.shape[1]):
#             if label[i][j] == 1:
#                 if pred_mask[i][j] == 1:
#                     TP += 1
#                 elif pred_mask[i][j] == 0:
#                     FN += 1
#             elif label[i][j] == 0:
#                 if pred_mask[i][j] == 1:
#                     FP += 1
#                 elif pred_mask[i][j] == 0:
#                     TN += 1
#     """
#     灵敏度=A/(A+C)，即有病诊断阳性的概率
#     特异度=D/(B+D)，即无病诊断阴性的概率
#     准确度＝(A+D)/(A+B+C+D)，即总阳性占总的概率,即用真阳性与真阴性人数之和占受试人数的百分率表示。
#     True negative(TN)，称为真阴率，表明实际是负样本预测成负样本的样本数
#     False positive(FP)，称为假阳率，表明实际是负样本预测成正样本的样本数
#     False negative(FN)，称为假阴率，表明实际是正样本预测成负样本的样本数
#     True positive(TP)，称为真阳率，表明实际是正样本预测成正样本的样本数
#     """
#     acc, sensitivity, specificity = 0, 0, 0
#     if (TP + FN + TN + FP)!=0:
#         acc = (TP + TN) / (TP + FN + TN + FP)   #所有中完全分对的（按像素）
#     else:
#         acc = None
#     if (TP+FN) != 0:
#         sensitivity = TP / (TP + FN)  # 漏诊（所有正类中预测为正类的）
#     else:
#         sensitivity = None
#     if (TP+FN) != 0:
#         specificity = TN / (TN + FP)  # 漏诊（所有正类中预测为正类的）
#     else:
#         specificity = None
#     return acc, sensitivity, specificity

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (352, 352))

        #相当于对测试集数据增强
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]

        # img4_90 = np.array(np.rot90(img4))
        # img5 = np.concatenate([img4[None], img4_90[None]]).transpose(0, 3, 1, 2)
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda()) #numpy转tensor? torch.Size([8, 3, 256, 256])

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1) <class 'tuple'>: (8, 256, 256)
        mask0c = mask[:,0,:,:]
        mask1c = mask[:,1,:,:]
        mask2c = mask[:,2,:,:]
        mask1 = mask0c[:4] + mask0c[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]   #<class 'tuple'>: (256, 256)

        mask1_ = mask1c[:4] + mask1c[4:, :, ::-1]
        mask2_ = mask1_[:2] + mask1_[2:, ::-1]
        mask3_ = mask2_[0] + np.rot90(mask2_[1])[::-1, ::-1]

        mask1_2 = mask2c[:4] + mask2c[4:, :, ::-1]
        mask2_2 = mask1_2[:2] + mask1_2[2:, ::-1]
        mask3_2 = mask2_2[0] + np.rot90(mask2_2[1])[::-1, ::-1]

        mask_com = np.array([mask3,mask3_,mask3_2]).transpose(1,2,0)
        return mask_com

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model,False)


def test_ce_net_vessel():
    path_txt = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/Data/data_png_png'
    data_list = [l.strip('\n') for l in open(os.path.join(path_txt, 'val.txt')).readlines()]
    disc = 20
    solver = TTAFrame(CE_Net_)
    solver.load('weights/CE-Net.th')
    # solver.load('best_model.pth')
    tic = time()
    target = './submits'
    if not os.path.exists(target):
        os.mkdir(target)
    gt_root = './dataset/DRIVE/test/masks/'
    total_m1 = 0

    hausdorff = 0
    total_acc = []
    total_sen = []
    total_spec = []
    threshold = 3.2#0.5*8,如果阈值为0.4，则应是0.4*8=3.2
    total_auc = []
    total_dice = []
    total_tumor_dice = []
    total_wall_dice = []

    dice_class = dice_bce_loss()
    for i in range(len(data_list)):
        image_path = os.path.join(path_txt, 'imgs', data_list[i])
        mask = solver.test_one_img_from_path(image_path)

        new_mask = mask.copy()

        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0
        # mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)   #(256, 256, 3)

        ground_truth_path = os.path.join(path_txt, 'masks_vis', data_list[i].split('.')[0] + '.png')

        ground_truth = np.array(Image.open(ground_truth_path))  #（512，512），uint8
        new_mask = cv2.resize(new_mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

        mask = cv2.resize(mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))   #预测的mask插值成真值512，(512, 512, 3)
        ground_truth = np.expand_dims(ground_truth, axis=2)
        ground_truth = onehot(ground_truth, 3)

        # auc_tumor = calculate_auc_test(new_mask[:,:,1] / 8., ground_truth[:,:,1])#八个数据（包括数据增强）的平均值
        # auc_wall = calculate_auc_test(new_mask[:, :, 2] / 8., ground_truth[:, :, 2])
        # auc = (auc_tumor + auc_wall) / 2

        # if auc is not None:
        #     total_auc.append(auc)

        predi_mask = np.zeros(shape=np.shape(mask))
        predi_mask[mask > disc] = 1
        gt = np.zeros(shape=np.shape(ground_truth))
        gt[ground_truth > 0] = 1

        dice_tumor = dice_class.soft_dice_coeff(torch.from_numpy(gt[:,:,1]),torch.from_numpy(predi_mask[:,:,1]))
        dice_wall = dice_class.soft_dice_coeff(torch.from_numpy(gt[:,:,2]),torch.from_numpy(predi_mask[:,:,2]))
        dice = (dice_tumor + dice_wall) / 2
        print(data_list[i].split('.')[0], dice.item(),'tumor dice:{},wall dice:{}'.format(dice_tumor,dice_wall))
        total_dice.append(dice)
        total_tumor_dice.append(dice_tumor)
        total_wall_dice.append(dice_wall)

        sen_t = sensitivity(torch.from_numpy(predi_mask[:,:,1]), torch.from_numpy(gt[:,:,1]))
        sen_w = sensitivity(torch.from_numpy(predi_mask[:, :, 2]), torch.from_numpy(gt[:, :, 2]))

        speci_t = specificity(torch.from_numpy(predi_mask[:,:,1]), torch.from_numpy(gt[:,:,1]))
        speci_w = specificity(torch.from_numpy(predi_mask[:,:,2]), torch.from_numpy(gt[:,:,2]))

        with open(metrics_path, 'a+', newline='') as file:
            csv_file = csv.writer(file)
            datas = [sen_t.item(), speci_t.item(), sen_w.item(), speci_w.item()]
            csv_file.writerow(datas)

        # print(name.split('.')[0], acc, sen, auc)
        # predi_mask_t = torch.tensor(predi_mask.transpose(2,0,1))
        # mask_t = torch.tensor(mask.transpose(2,0,1))
        # acc = accuracy(predi_mask_t, mask_t)
        # total_acc.append(acc)
        # p = precision(predi_mask_t, mask_t)
        # total_sen.append(p)
        # r = recall(predi_mask_t, mask_t)
        # total_spec.append(r)

        pred = helpers.onehot_to_mask(predi_mask, palette)

        cv2.imwrite(os.path.join(target, data_list[i].split('.')[0] + '-masks.png'), pred.astype(np.uint8))
    print("dice_mean: {}，{},{}, dice_std: {}".format(np.mean(total_dice), np.mean(total_tumor_dice),np.mean(total_wall_dice),np.std(total_dice)))
    print("accuracy_mean: {}, accuracy_std: {}".format(np.mean(total_acc), np.std(total_acc)))
    print("sensitivity_mean: {}, sensitivity_std: {}".format(np.mean(total_sen), np.std(total_sen)))
    print("specificity_mean: {}, specificity_std: {}".format(np.mean(total_spec), np.std(total_spec)))


if __name__ == '__main__':
    with open(metrics_path, "w+", newline='') as file:
        csv_file = csv.writer(file)
        head = ["TumorSen", "TumorSpec", "WallSen", "WallSpec"]
        csv_file.writerow(head)
    test_ce_net_vessel()