"""
Defines the loss functions used to train a model.
"""

import torch
import torch.nn as nn
import numpy as np
import model.metric as metric

from pathlib import Path
from PIL import Image

# from https://github.com/EKami/carvana-challenge/blob/original_unet/src/nn/losses.py (modified)
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        
        return score


def soft_dice_loss(outputs, ground_truths):
    """
    Compute the Dice loss (F1 loss) given outputs and ground truths.
    Applies Sigmoid function to outputs.
    Args:
        outputs: (Tensor) dimension batch_size x ground_truth_width x ground_truth_height - output of the model
        ground_truths: (Tensor) dimension batch_size x ground_truth_width x ground_truth_height - ground truth images
    Returns:
        loss (Tensor): binary cross entropy loss for all images in the batch
    """
    loss_fn = SoftDiceLoss()
    return loss_fn(outputs, ground_truths)


def cross_entropy_loss(outputs, ground_truths):
    """
    Compute the binary cross entropy loss given outputs and ground truths.
    Applies Sigmoid function to outputs.
    Args:
        outputs: (Tensor) dimension batch_size x ground_truth_width x ground_truth_height - output of the model
        ground_truths: (Tensor) dimension batch_size x ground_truth_width x ground_truth_height - ground truth images
    Returns:
        loss (Tensor): binary cross entropy loss for all images in the batch
    """
    loss_fn = nn.BCELoss()
    return loss_fn(torch.sigmoid(outputs), ground_truths)


def soft_dice_and_cross_entropy_loss(outputs, ground_truths):
    return cross_entropy_loss(outputs, ground_truths) + soft_dice_loss(outputs, ground_truths)
    

### Testing Section ###
#size = 20
#image = Image.open(Path('data/heart_scans/ALFE-BL/ALFE-BL_CineMR_ti00_sl00_ENDO.png'))
#image = image.resize((size, size), Image.BILINEAR)
#image_np = np.array(image)
#image_np_complete = np.expand_dims(image_np, axis=0)
#
#image = Image.open(Path('data/heart_scans/ALFE-BL/ALFE-BL_CineMR_ti10_sl08_EPI.png'))
#image = image.resize((size, size), Image.BILINEAR)
#image_np = np.array(image)
#image_np_complete = np.resize(image_np_complete, (2,size,size))
#image_np_complete[1] = image_np
#print(image_np_complete)
#
#image = Image.open(Path('data/heart_scans/ALFE-BL/ALFE-BL_CineMR_ti00_sl01_ENDO.png'))
#image = image.resize((size, size), Image.BILINEAR)
#image_np = np.array(image)
#image_np_complete_y = np.expand_dims(image_np, axis=0)
#
#image = Image.open(Path('data/heart_scans/ALFE-BL/ALFE-BL_CineMR_ti10_sl07_EPI.png'))
#image = image.resize((size, size), Image.BILINEAR)
#image_np = np.array(image)
#image_np_complete_y = np.resize(image_np_complete_y, (2,size,size))
#image_np_complete_y[1] = image_np
#print(image_np_complete_y)
#
#image_np_complete = (torch.from_numpy(image_np_complete)).type(torch.FloatTensor)
#image_np_complete_y = (torch.from_numpy(image_np_complete_y)).type(torch.FloatTensor)
#
#print("Cross Entropy Loss: {}".format(cross_entropy_loss(image_np_complete, image_np_complete_y)))
#
#print("Soft Dice Loss: {}".format(soft_dice_loss(image_np_complete, image_np_complete_y)))   


#######################
    