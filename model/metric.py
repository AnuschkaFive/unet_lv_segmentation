"""
Defines the metrics used in evaluating a model.
"""

#import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Constant to be added to both nominators and denominators, to prevent division by 0.
SMOOTH = 1e-5

def accuracy(outputs, ground_truths, debug=False):
    """
    Compute the accuracy, given the outputs and ground truths for all images.
    (TP + TN / TP + TN + FP + FN)
    Args:
        outputs: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - output of the model
        ground_truths: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - ground truth
        debug: (bool) True, when debug statements should be printed. False, otherwise.
    Returns: (float) accuracy in [0,1]
    """
    if debug:
        print('Accuracy:')
    num = outputs.shape[0]    
    outputs_f = outputs.reshape(num, -1)
    ground_truths_f = ground_truths.reshape(num, -1)
    intersection = (outputs_f * ground_truths_f).sum(1)
    if debug:
        print('    Intersection: {}'.format(intersection))
    cardinalities = outputs_f.sum(1) + ground_truths_f.sum(1)
    union = cardinalities - intersection
    if debug:
        print('    Union: {}'.format(union))
    score = (intersection + (outputs_f.shape[1] - union)) / outputs_f.shape[1]
    if debug:
        print('    Score before Averaging: {}'.format(score))
    return score.sum() / num


def precision(outputs, ground_truths, debug=False):
    """
    Compute the precision, given the outputs and ground truths for all images.
    (TP / TP + FP)
    Args:
        outputs: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - output of the model
        ground_truths: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - ground truth
        debug: (bool) True, when debug statements should be printed. False, otherwise.
    Returns: (float) precision in [0,1]
    """
    if debug:
        print('Precision:')
    num =  outputs.shape[0]
    outputs_f = outputs.reshape(num, -1)
    ground_truths_f = ground_truths.reshape(num, -1)
    intersection = (outputs_f * ground_truths_f).sum(1)
    if debug:
        print('    Intersection: {}'.format(intersection))
    outputs_cardinality = outputs_f.sum(1)
    if debug:
        print('    Outputs Cardinality: {}'.format(outputs_cardinality))
    score = (intersection + SMOOTH) / (outputs_cardinality + SMOOTH)
    if debug:
        print('    Score before Averaging: {}'.format(score))
    return score.sum() / num


def recall(outputs, ground_truths, debug=False):
    """
    Compute the recall, given the outputs and ground truths for all images.
    (TP / TP + FN)
    Args:
        outputs: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - output of the model
        ground_truths: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - ground truth
        debug: (bool) True, when debug statements should be printed. False, otherwise.
    Returns: (float) recall in [0,1]
    """
    if debug:
        print('Recall:')
    num =  outputs.shape[0]
    outputs_f = outputs.reshape(num, -1)
    ground_truths_f = ground_truths.reshape(num, -1)
    intersection = (outputs_f * ground_truths_f).sum(1)
    if debug:
        print('    Intersection: {}'.format(intersection))
    ground_truth_cardinality = ground_truths_f.sum(1)
    if debug:
        print('    Ground Truth Cardinality: {}'.format(ground_truth_cardinality))
    score = (intersection + SMOOTH) / (ground_truth_cardinality + SMOOTH)
    if debug:
        print('    Score before Averaging: {}'.format(score))
    return score.sum() / num


def DSC(outputs, ground_truths, debug=False):
    """
    Compute the Dice coefficient (F1 score), given the outputs and ground truths for all images. 
    (2 * TP / 2TP + FP + FN)
    Args:
        outputs: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - output of the model
        ground_truths: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - ground truth
        debug: (bool) True, when debug statements should be printed. False, otherwise.
    Returns: (float) Dice coefficient in [0,1]
    """
    if debug:
        print('DSC:')
    num =  outputs.shape[0]
    outputs_f = outputs.reshape(num, -1)
    ground_truths_f = ground_truths.reshape(num, -1)
    intersection = (outputs_f * ground_truths_f).sum(1)
    if debug:
        print('    Intersection: {}'.format(intersection))
    cardinalities = outputs_f.sum(1) + ground_truths_f.sum(1)
    if debug:
        print('    Cardinalities: {}'.format(cardinalities))
    score = (2. * intersection + SMOOTH) / (cardinalities + SMOOTH)
    if debug:
        print('    Score before Averaging: {}'.format(score))
    return score.sum() / num
    

def IOU(outputs, ground_truths, debug=False):
    """
    Compute the Intersection over Union (Jaccard index), given the outputs and ground truths for all images.
    (TP / TP + FP + FN)
    Args:
        outputs: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - output of the model
        ground_truths: (np.ndarray) dimension batch_size x ground_truth_height x ground_truth_width - ground truth
        debug: (bool) True, when debug statements should be printed. False, otherwise.
    Returns: (float) Intersection over Union in [0,1]
    """
    if debug:
        print('IOU:')
    num =  outputs.shape[0]
    outputs_f = outputs.reshape(num, -1)
    ground_truths_f = ground_truths.reshape(num, -1)
    intersection = (outputs_f * ground_truths_f).sum(1)
    if debug:
        print('    Intersection: {}'.format(intersection))
    cardinalities = outputs_f.sum(1) + ground_truths_f.sum(1)
    union = cardinalities - intersection
    if debug:
        print('    Union: {}'.format(union))
    score = (intersection + SMOOTH) / (union + SMOOTH)
    if debug:
        print('    Score before Averaging: {}'.format(score))
    return score.sum() / num

    
# maintain all metrics required in this dictionary - these are used in the training and evaluation loops
metrics_dict = {
    'dsc': DSC,
    'iou': IOU,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall
}

### Testing Section ###
#size = 20
#image = Image.open(Path('data/heart_scans/ALFE-BL/ALFE-BL_CineMR_ti00_sl00_ENDO.png'))
#print("Mode: {}".format(image.mode))
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
#print(DSC(image_np_complete, image_np_complete_y, debug=True))
#
#print(IOU(image_np_complete, image_np_complete_y, debug=True))
#
#print(precision(image_np_complete, image_np_complete_y, debug=True))
#
#print(recall(image_np_complete, image_np_complete_y, debug=True))
#
#print(accuracy(image_np_complete, image_np_complete_y, debug=True))
########################