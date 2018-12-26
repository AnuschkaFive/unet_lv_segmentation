"""
Defines the loss functions used to train a model.
"""

import torch
import torch.nn as nn
import numpy as np
import model.metric as metric

from pathlib import Path
from PIL import Image

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
    outputs_f = torch.sigmoid(outputs).view( -1)
    ground_truths_f = ground_truths.view(-1)
    intersection = (outputs_f * ground_truths_f).sum()
    cardinalities = outputs_f.sum(1) + ground_truths_f.sum()
    return (2. * intersection + 1.) / (cardinalities + 1.)
    #return torch.tensor(1. - metric.DSC((torch.sigmoid(outputs)).detach().numpy(), ground_truths.numpy()))


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
#print(cross_entropy_loss(image_np_complete, image_np_complete_y))
#
#print(soft_dice_loss(image_np_complete, image_np_complete_y))   


#######################
    