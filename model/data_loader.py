import random
import os
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import util.cross_validation as cv


class Heart2DSegmentationDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, endo_or_epi, transform=None):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) Directory containing the dataset
            endo_or_epi: (string)
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.endo_or_epi = endo_or_epi
        self.ground_truth_filenames = [str(path) for path in Path(data_dir).glob('**/*' + endo_or_epi + '*.png')]
        self.transform = transform
        self.convert_to_tensor = transforms.ToTensor()

    def __len__(self):
        # return size of dataset
        return len(self.ground_truth_filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            scan: (Tensor) transformed ORIG scan image
            ground_truth: (Tensor) corresponding transformed ENDO or EPI ground truth image
            ground_truth_name: (string) file name of the ENDO or EPI ground truth image
        """
        ground_truth = Image.open(self.ground_truth_filenames[idx])
        scan = Image.open(self.ground_truth_filenames[idx].replace(self.endo_or_epi, 'ORIG'))
        
        scan = self.convert_to_tensor(scan).type(torch.FloatTensor)
        ground_truth = self.convert_to_tensor(ground_truth).type(torch.FloatTensor)

        if(self.transform):
            scan = self.transform(scan)
            #ground_truth = self.transform(ground_truth)

        #return scan.type(torch.FloatTensor), ground_truth.type(torch.FloatTensor), self.ground_truth_filenames[idx]
        return scan, ground_truth, self.ground_truth_filenames

def fetch_dataloader(types, data_dir, hyper_params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        is_cv: (int) The number of folds for cross validation. -1, if no cross validation.
        data_dir: (string) directory containing the dataset
        hyper_params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    
    mean, std = mean_std_calc(DataLoader(Heart2DSegmentationDataset(str(Path(data_dir) / "train_heart_scans"), hyper_params.endo_or_epi)))
    print("Mean: {}, Std: {}".format(mean.item(), std.item()))
    # borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    # and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # define a training image loader that specifies transforms on images. See documentation for more details.
    train_transformer = transforms.Compose([
        #transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        #transforms.RandomVerticalFlip(), #randomly flip image vertically
        #transforms.RandomRotation((80, 100), Image.BILINEAR),
        #transforms.ToTensor(),  # transform it into a torch tensor
        transforms.Normalize(mean=[mean.item()], std=[std.item()])
        ])
    
    # loader for evaluation, no horizontal flip
    eval_transformer = transforms.Compose([
        #transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        #transforms.RandomVerticalFlip(), #randomly flip image vertically
        #transforms.RandomRotation((80, 100), Image.BILINEAR),
        #transforms.ToTensor(),  # transform it into a torch tensor
        transforms.Normalize(mean=[mean.item()], std=[std.item()])
        ])

    for split in ['train', 'test']:
        if split in types:
            # TODO: ich glaube, hier soll das was anderes sein, nämlich nur eine string concatenation...? Kein Unterordner?
            path = str(Path(data_dir) / "{}_heart_scans".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            # TODO: hier, falls k-fold-cross-validation, dann in folds aufteilen?
            if split == 'train':
                # if is_cv != -1:
                # - make array of dict with is_cv as size
                # - use SubsetRandomSampler, with seed and shuffle, to create dataloaders of train and val, for each array space
                # - else: just make train (of all)
                dl = DataLoader(Heart2DSegmentationDataset(path, hyper_params.endo_or_epi, train_transformer), 
                                batch_size=hyper_params.batch_size, 
                                shuffle=True,
                                num_workers=hyper_params.num_workers,
                                pin_memory=hyper_params.cuda)
            else:
                dl = DataLoader(Heart2DSegmentationDataset(path, hyper_params.endo_or_epi, eval_transformer), 
                                batch_size=hyper_params.batch_size, 
                                shuffle=False,
                                num_workers=hyper_params.num_workers,
                                pin_memory=hyper_params.cuda)

            dataloaders[split] = dl

    return dataloaders


def mean_std_calc(dataloader):
    """
    Function to calculate the mean and standard
    deviation of a given dataset.
    Parameter:
       dataloader(torch.dataloader): Dataloader of the dataset
       to calculate the mean and standard deviation.
    Inspired from 'ptrblck' at https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    """
    mean = 0
    std = 0
    samples = 0
    for data, _, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        samples += batch_samples

    return (mean / samples),(std / samples)


# TODO: Indices should probably be shuffled...?
#ds = Heart2DSegmentationDataset('data/64x64_heart_scans/train_heart_scans', 'ENDO', train_transformer)
#for train_idx, test_idx in cv.k_folds(n_splits = 5, subjects = ds.__len__(), frames=1):
#    dataset_train = torch.utils.data.Subset(ds, train_idx)
#    dataset_test = torch.utils.data.Subset(ds, train_idx)
#    train_loader = torch.utils.data.DataLoader(dataset = ds, batch_size = 10)    
#    test_loader = torch.utils.data.DataLoader(dataset = ds, batch_size = 10)
    

### Testing Section ###
#size = 20
#image = Image.open(Path('data/heart_scans/ALFE-BL/ALFE-BL_CineMR_ti00_sl00_ORIG.png'))
#image = image.resize((size, size), Image.BILINEAR)
#print(image)
#
#demo_transformer = transforms.Compose([
#        #transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
#        #transforms.RandomVerticalFlip(), #randomly flip image vertically
#        #transforms.RandomRotation((80, 100), Image.BILINEAR),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[303.23663330078125], std=[435.22442626953125])
#        ])
#    
#print(image.ToTensor())    
#demo_tensor_transf = transforms.ToTensor()
#demo_normalize_transf = transforms.Normalize(mean=[303.23663330078125], std=[435.22442626953125])  
#    
#transf_image = demo_tensor_transf(image)    
#print(transf_image)
#
#transf_image_floats = transf_image.type(torch.FloatTensor)
#print(transf_image_floats)
#
#transf_image_n = demo_normalize_transf(transf_image)    
#print(transf_image_n)
#
#transf_image_n = demo_normalize_transf(transf_image_floats)    
#print(transf_image_n)

#######################    