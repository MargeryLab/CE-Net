from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from networks.cenet import CE_Net_
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from Visualizer import Visualizer
from torch.autograd import Variable as V

import Constants
from tqdm import tqdm
# -Xms128m
# -Xmx628m
# -XX:ReservedCodeCacheSize=240m
# -XX:+UseConcMarkSweepGC
# -XX:SoftRefLRUPolicyMSPerMB=50
# -ea
# -XX:CICompilerCount=2
# -Dsun.io.useCanonPrefixCache=false
# -Djava.net.preferIPv4Stack=true
# -Djdk.http.auth.tunneling.disabledSchemes=""
# -XX:+HeapDumpOnOutOfMemoryError
# -XX:-OmitStackTraceInFastThrow


# Please specify the ID of graphics cards that you want to use
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# train_img_root = os.path.join(Constants.ROOT, 'training/images')
# val_img_root = os.path.join(Constants.ROOT, 'validation/images')
# n_train = len(os.listdir(train_img_root))
# n_val = len(os.listdir(val_img_root))


# def eval_net(solver, val_iter, device):
#     for i in solver.net.modules():
#         if isinstance(i, nn.BatchNorm2d):
#             i.eval()
#     tot = 0
#
#     with tqdm(total=len(val_iter), desc='Validation round', unit='batch', leave=False) as pbar:
#         # data_loader_iter = iter(val_iter)
#         for batch in val_iter:
#             imgs, true_masks = batch[0], batch[1]
#             # imgs = imgs.to(device=device, dtype=torch.float32)
#             # true_masks = true_masks.to(device=device,dtype=torch.float32)
#             imgs = V(imgs.cuda(),volatile=False)
#             true_masks = V(true_masks.cuda(),volatile=False)
#
#             with torch.no_grad():
#                 pred = solver.net.forward(imgs)
#             pred = (pred.cuda() > 0.5).float()
#             tot += dice_bce_loss().soft_dice_coeff(true_masks, pred.cuda())
#             # tot += dice_coeff(pred.cuda(), true_masks).item()
#             pbar.update()
#     for i in solver.net.modules():
#         if isinstance(i, nn.BatchNorm2d):
#             i.train()
#     return tot/n_val


def CE_Net_Train():
    NAME = 'CE-Net' + Constants.ROOT.split('/')[-1] #'CE-NetDRIVE'

    # run the Visdom
    # viz = Visualizer(env=NAME)

    solver = MyFrame(CE_Net_, dice_bce_loss, Constants.LEARNING_RATE)
    # solver.load('best_model.pth')
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD

    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    dataset = ImageFolder(root_path=Constants.ROOT, datasets='DRIVE')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)

    # valid_dataset = ImageFolder(root_path=Constants.ROOT, datasets='DRIVE', mode='valid')
    # val_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=batchsize,
    #     shuffle=True,
    #     num_workers=batchsize
    # )

    # start the logging files
    # mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()

    no_optim = 0
    global_step = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS

    # writer = SummaryWriter(comment=f'LR_{Constants.LEARNING_RATE}_BS_{batchsize}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    best_dice = 0
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        # val_loader_iter = iter(val_loader)
        train_epoch_loss = 0
        index = 0

        for img, mask, name in data_loader_iter:
            solver.set_input(img, mask)
            train_loss, dice, pred = solver.optimize()
            if dice > best_dice:
                best_dice = dice
                solver.save('best_model.pth')
                # solver.save('./weights/best_model.pth')
            train_epoch_loss += train_loss
            index = index + 1

            # global_step += 1
            # writer.add_scalar('Loss/train', train_loss.item(), global_step)
            #
            # show_image = (img + 1.6) / 3.2 * 255.
            # writer.add_images('images', show_image, global_step)
            # writer.add_images('mask/true', mask, global_step)
            # writer.add_images('masks/pred', pred > 0.5, global_step)

        # val_score = eval_net(solver, val_loader_iter, device)
        # writer.add_scalar('learning_rate', solver.optimizer.param_groups[0]['lr'], epoch)
        # writer.add_scalar('Dice/valid', val_score, epoch)


        # show the original images, predication and ground truth on the visdom.
        show_image = (img + 1.6) / 3.2 * 255.
        # viz.img(name='images', img_=show_image[0, :, :, :])
        # viz.img(name='labels0', img_=mask[0, 0, :, :])
        # viz.img(name='labels1', img_=mask[0, 1, :, :])
        # viz.img(name='labels2', img_=mask[0, 2, :, :])
        # viz.img(name='prediction0', img_=pred[0, 0, :, :])
        # viz.img(name='prediction1', img_=pred[0, 1, :, :])
        # viz.img(name='prediction2', img_=pred[0, 2, :, :])

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        # print(mylog, '********')
        # print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        # print(mylog, 'train_loss:', train_epoch_loss)
        # print(mylog, 'SHAPE:', Constants.Image_size)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        # print('SHAPE:', Constants.Image_size)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '.th')
        if no_optim > Constants.NUM_EARLY_STOP:
            # print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:#如果连着10次loss都没有降低
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '.th')
            # solver.update_lr(2.0, factor=True, mylog=mylog)
            solver.update_lr(2.0, factor=True, mylog=None)
        # mylog.flush()

    # print(mylog, 'Finish!')
    # print('Finish!')
    # mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    CE_Net_Train()



