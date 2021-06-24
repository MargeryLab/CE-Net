# validate.py
import os
import cv2
from PIL import Image

from utils import helpers
from utils.metrics import *
from networks.cenet import CE_Net_
from data import ImageFolder
import Constants
from framework import MyFrame

batch_size = 1
palette = [[0], [127], [255]]


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        return self.test_one_img_from_path_1(path)

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model,False)

def auto_val():
    # 效果展示图片数
    iters = 0
    SIZES = 8
    imgs = []
    preds = []
    gts = []
    dices = 0
    wall_dices = 0
    tumor_dices = 0

    solver = MyFrame(CE_Net_, lr=Constants.LEARNING_RATE)
    # solver.load('submits/CE-Net.th')
    solver.load('best_model.pth')

    target = './submits/log_CE_Net/'
    if not os.path.exists(target):
        os.mkdir(target)

    dataset = ImageFolder(root_path=Constants.ROOT, datasets='DRIVE')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    data_loader_iter = iter(data_loader)
    for img, mask, name in data_loader_iter:
        img = img.cuda()
        mask = mask.cuda()
        solver.set_input(img, mask)
        pred = solver.test_one_img(img)

        pred = pred.cpu().detach()
        mask = mask.cpu().detach()
        iters += batch_size
        tumor_dice = diceCoeff(pred[:, 1:2, :], mask[:, 1:2, :])
        wall_dice = diceCoeff(pred[:, 2:3, :], mask[:, 2:3, :])
        mean_dice = (tumor_dice + wall_dice) / 2
        dices += mean_dice
        wall_dices += wall_dice
        tumor_dices += tumor_dice
        acc = accuracy(pred, mask)
        p = precision(pred, mask)
        r = recall(pred, mask)
        print('{},mean_dice={:.4}, tumor_dice={:.4}, wall_dice={:.4}, accuracy={:.4}, precision={:.4}, recall={:.4}'
              .format(name[:--4], mean_dice.item(), tumor_dice.item(), wall_dice.item(),
                      acc, p, r))
        gt = mask.numpy()[0].transpose([1, 2, 0])
        gt = helpers.onehot_to_mask(gt, palette)
        pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
        pred = helpers.onehot_to_mask(pred, palette)
        # cv2.imwrite(os.path.join(target,name), pred.astype(np.uint8))
        im = img[0].cpu().numpy().transpose([1, 2, 0])
        if len(imgs) < SIZES:
            imgs.append((im + 1.6) / 3.2 * 255.)
            preds.append(pred)
            gts.append(gt)
    val_mean_dice = dices / (len(data_loader) / batch_size)
    val_wall_dice = wall_dices / (len(data_loader) / batch_size)
    val_tumor_dice = tumor_dices / (len(data_loader) / batch_size)
    print('Val Mean Dice = {:.4}, Val Tumor Dice = {:.4}, Val Wall Dice = {:.4}'
          .format(val_mean_dice, val_tumor_dice, val_wall_dice))

    # imgs = np.hstack([*imgs])
    # preds = np.hstack([*preds])
    # gts = np.hstack([*gts])
    # show_res = np.vstack(np.uint8([imgs, preds, gts]))
    # cv2.imshow("top is mri , middle is pred,  bottom is gt", show_res)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # val(model)
    auto_val()
