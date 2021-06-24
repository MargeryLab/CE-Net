import torch
from torch.autograd import Variable as V

import cv2
import os,glob
import numpy as np
from PIL import Image
from utils import helpers

import warnings
from networks.cenet import CE_Net_
import SimpleITK as itk

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
palette = [[0], [127], [255]]

BATCHSIZE_PER_CARD = 16

def dcm_to_png(raw_data_path,png_path):
    # id_ls = [id[:-7] for id in os.listdir(label_path)]
    # print(len(id_ls))
    if not os.path.exists(raw_data_path):
        os.mkdir(raw_data_path)

    for ID in os.listdir(raw_data_path):
        # if ID not in id_ls:
        #     continue
        data_path = os.path.join(raw_data_path, ID, ID, 'HRT2')
        if os.path.exists(data_path):
            reader = itk.ImageSeriesReader()
            dcm_path = reader.GetGDCMSeriesFileNames(data_path)
            reader.SetFileNames(dcm_path)
            imgs = reader.Execute()

            imgs_arr = itk.GetArrayFromImage(imgs)
            imgs_arr = (imgs_arr / np.max(imgs_arr) * 255).astype(np.uint8)

            index = '00'
            for i in range(imgs_arr.shape[0]):
                if (index[0] == '0') and (index != '09'):
                    index = '0' + str(int(index[-1]) + 1)
                elif index == '09':
                    index = '10'
                else:
                    index = str(int(index) + 1)
                cv2.imwrite(os.path.join(png_path,ID+'-'+index+'.png'), imgs_arr[i])

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


def save_array_as_nii_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    if(reference_name is not None):
        reader = itk.ImageSeriesReader()
        dicom_path = reader.GetGDCMSeriesFileNames(reference_name)
        reader.SetFileNames(dicom_path)
        img_ref = reader.Execute()

        img_arr = itk.GetArrayFromImage(img_ref)
        if img_arr.shape[1] <100:
            data = data.transpose(2,0,1)    #（z,x,y)->(y,z,x）
        elif img_arr.shape[2] < 100:
            data = data.transpose(2, 1, 0)
        img = itk.GetImageFromArray(data)
        img.CopyInformation(img_ref)    #(512,25,512）
    else:
        img = itk.GetImageFromArray(data)
    itk.WriteImage(img, filename)


def test_ce_net_vessel(source, target):
    if not os.path.exists(target):
        os.mkdir(target)
    val = os.listdir(source)
    np.random.shuffle(val)
    disc = 20
    solver = TTAFrame(CE_Net_)
    solver.load('weights/best_model.pth')
    # contour_path= './submits/47TumorWallContour'
    if not os.path.exists(target):
        os.mkdir(target)
    # gt_root = './dataset/DRIVE/test/images/'
    threshold = 3.2   #0.5*8,如果阈值为0.4，则应是0.4*8=3.2
    for i, name in enumerate(val):
        image_path = os.path.join(source, name.split('.')[0] + '.png')
        mask = solver.test_one_img_from_path(image_path)

        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0
        # mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)   #(256, 256, 3)

        img_path = os.path.join(source, name.split('.')[0] + '.png')

        img_arr = np.array(Image.open(img_path))  #（512，512），uint8

        mask = cv2.resize(mask, dsize=(np.shape(img_arr)[1], np.shape(img_arr)[0]))   #预测的mask插值成真值512，(512, 512, 3)

        # t_mask = mask[:,:,1:2].astype(np.uint8)[:,:,0]
        # ret, tumor_binary = cv2.threshold(t_mask, 128, 255, cv2.THRESH_BINARY)
        # if np.max(tumor_binary) == 0:
        #     tumor_edge = t_mask
        # else:
        #     # tumor_binary = cv2.GaussianBlur(tumor_binary, (3, 3), 0)
        #     tumor_edge = cv2.Canny(tumor_binary, 10, 150)
        #     # tumor_edge = cv2.morphologyEx(tumor_edge, cv2.MORPH_CLOSE, kernel=(3, 3), iterations=3)
        #     # tumor_edge = cv2.adaptiveThreshold(tumor_edge,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,0)
        #     # tumor_edge = cv2.Laplacian(mask[:,:,1:2].astype(np.uint8)[:,:,0], cv2.CV_8U, ksize=3)
        #     # tumor_edge = cv2.Sobel(mask[:, :, 1:2].astype(np.uint8)[:, :, 0],cv2.CV_8U, 1, 1)
        # w_mask = mask[:,:,2:3].astype(np.uint8)[:,:,0]
        # ret_, wall_binary = cv2.threshold(w_mask, 128, 255, cv2.THRESH_BINARY)
        # if np.max(wall_binary) == 0:
        #     wall_edge = w_mask
        # else:
        #     # wall_binary = cv2.GaussianBlur(wall_binary, (3, 3), 0)
        #     wall_edge = cv2.Canny(wall_binary, 10, 150)
        #     # wall_edge = cv2.morphologyEx(wall_edge, cv2.MORPH_CLOSE, kernel=(3, 3), iterations=3)
        #     # wall_edge = cv2.adaptiveThreshold(wall_edge,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,0)
        #     # wall_edge = cv2.Laplacian(mask[:,:,2:3].astype(np.uint8)[:,:,0], cv2.CV_8U, ksize=3)
        #     # wall_edge = cv2.Sobel(mask[:, :, 2:3].astype(np.uint8)[:, :, 0], cv2.CV_8U, 1, 1)

        # mask1 = mask[:,:,0:1].astype(np.uint8)[:,:,0]
        # edge = np.concatenate(mask1,tumor_edge,wall_edge)
        # edge = np.array([mask1,tumor_edge,wall_edge])

        predi_mask = np.zeros(shape=np.shape(mask))
        predi_mask[mask > disc] = 1
        # edge = edge.transpose(1,2,0)
        # edge_bi = np.zeros(shape=np.shape(edge))
        # edge_bi[edge > disc] = 1
        if not os.path.exists(os.path.join(target,name.split('.')[0][:7])):
            os.mkdir(os.path.join(target,name.split('.')[0][:7]))
        # if not os.path.exists(os.path.join(contour_path,name.split('.')[0][:7])):
        #     os.mkdir(os.path.join(contour_path,name.split('.')[0][:7]))

        pred = helpers.onehot_to_mask(predi_mask, palette)
        # edge_mask = helpers.onehot_to_mask(edge_bi, palette)

        id = format(name.split('.')[0][:7])
        cv2.imwrite(os.path.join(target,id,name), pred.astype(np.uint8))
        # cv2.imwrite(os.path.join(contour_path, id, name), edge_mask.astype(np.uint8))


def png_to_nii(mask_id_path, raw_dcm_path, res_path):
    for id in sorted(os.listdir(mask_id_path)):
        id_path = os.path.join(mask_id_path,id)
        imgs = glob.glob(str(id_path) + str("/*"))
        imgs.sort()
        one = cv2.imread(imgs[0], flags=0)
        all_img_arr = np.zeros((len(imgs),one.shape[0],one.shape[1]),dtype=np.uint8)

        for i in range(len(imgs)):
            img_path = imgs[i]
            mask_array = cv2.imread(img_path,flags=0)
            # mask_array = Image.open(img_path)
            mask_array[mask_array == 126] = 1
            mask_array[mask_array == 127] = 1
            mask_array[mask_array == 255] = 2
            all_img_arr[i,:,:] = mask_array.astype(np.uint8)
        save_array_as_nii_volume(all_img_arr,res_path+'/{}.nii'.format(id),os.path.join(raw_dcm_path,id,id,'HRT2'))


if __name__ == '__main__':
    raw_data_path = '/media/margery/4ABB9B07DF30B9DB/DARA-SIRRUNRUN/MengPing_20210614/dcmDataS126'
    png_path = '/media/margery/4ABB9B07DF30B9DB/DARA-SIRRUNRUN/MengPing_20210614/dcmDataS126PNG'
    mask_png_pred_path = 'submits/126'
    res_path = '/media/margery/4ABB9B07DF30B9DB/DARA-SIRRUNRUN/MengPing_20210614/DataS126PredNii'

    dcm_to_png(raw_data_path,png_path)
    test_ce_net_vessel(png_path, mask_png_pred_path)
    png_to_nii(mask_png_pred_path, raw_data_path, res_path)