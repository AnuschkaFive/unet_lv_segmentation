"""
Dataloader for getting batches of normalized dataset samples.

Originally based on https://cs230-stanford.github.io/pytorch-getting-started.html and 
https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision.
"""

import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import build_dataset

class Heart2DSegmentationDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, endo_or_epi, transform=None):
        """
        Store the filenames of the PNGs to use. Specifies normalization to apply on images.
        Args:
            data_dir: (string) Directory containing the dataset
            endo_or_epi: (string) Whether the dataset is using ENDO or EPI ground truth images.
            transform: (torchvision.transforms) Normalization to apply on image.
        """
        self.endo_or_epi = endo_or_epi
        self.ground_truth_filenames = build_dataset.stratify_filenames([str(path) for path in Path(data_dir).glob('**/*' + endo_or_epi + '*.png')])
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
            scan: (Tensor) Normalized ORIG scan image.
            ground_truth: (Tensor) corresponding transformed ENDO or EPI ground truth image
            ground_truth_name: (string) file name of the ENDO or EPI ground truth image
        """
        ground_truth = Image.open(self.ground_truth_filenames[idx])
        scan = Image.open(self.ground_truth_filenames[idx].replace(self.endo_or_epi, 'ORIG'))
        
        scan = self.convert_to_tensor(scan).type(torch.FloatTensor)
        ground_truth = self.convert_to_tensor(ground_truth).type(torch.FloatTensor)

        if(self.transform):
            scan = self.transform(scan)

        return scan, ground_truth, self.ground_truth_filenames[idx]

def fetch_dataloader(types, data_dir, hyper_params, train_idx=None, val_idx=None):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        hyper_params: (Params) hyperparameters
        train_idx: (int) When not none, k-fold CV is used. Specifies subset to be used as CV training set.
        val_idx: (int) When not none, k-fold CV is used. Specifies subset to be used as CV validation set.
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    
    # TODO: write this to hyper_params, make hyper_params an out variable? then save? yes, AND: when ONLY test is requested, load from hyperparams!
    # TODO: also, add 3rd variation of types: for testing, only read it from hyper_params (DO I NEED TO READ HYPER_PARAMS FOR JUST TESTING?)
    if train_idx is not None:
        mean, std = mean_std_calc(DataLoader(Subset(Heart2DSegmentationDataset(str(Path(data_dir) / "train_heart_scans"), hyper_params.endo_or_epi), train_idx)))
        hyper_params.mean = mean.item()
        hyper_params.std = std.item()
    else:
        if 'train' in types:
            mean, std = mean_std_calc(DataLoader(Heart2DSegmentationDataset(str(Path(data_dir) / "train_heart_scans"), hyper_params.endo_or_epi)))
            hyper_params.mean = mean.item()
            hyper_params.std = std.item()
        else:
            mean, std = torch.tensor(hyper_params.mean), torch.tensor(hyper_params.std)
    
    print("Mean: {}, Std: {}".format(mean.item(), std.item()))
    # borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    # and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    train_transformer = transforms.Compose([
        transforms.Normalize(mean=[mean.item()], std=[std.item()])
        ])
    
    eval_transformer = transforms.Compose([
        transforms.Normalize(mean=[mean.item()], std=[std.item()])
        ])

    for split in ['train', 'val', 'test']:
        if split in types:
            path = str(Path(data_dir) / "{}_heart_scans".format(split if split != 'val' else 'train'))

            if split == 'train':
                if train_idx is not None:
                    dl = DataLoader(Subset(Heart2DSegmentationDataset(path, hyper_params.endo_or_epi, train_transformer), train_idx), 
                                    batch_size=hyper_params.batch_size, 
                                    shuffle=True,
                                    num_workers=hyper_params.num_workers,
                                    pin_memory=hyper_params.cuda)
                else:
                    dl = DataLoader(Heart2DSegmentationDataset(path, hyper_params.endo_or_epi, train_transformer), 
                                    batch_size=hyper_params.batch_size, 
                                    shuffle=True,
                                    num_workers=hyper_params.num_workers,
                                    pin_memory=hyper_params.cuda)
            else:
                if (split == 'val') and (val_idx is not None): 
                    dl = DataLoader(Subset(Heart2DSegmentationDataset(path, hyper_params.endo_or_epi, eval_transformer), val_idx), 
                                    batch_size=hyper_params.batch_size, 
                                    shuffle=False,
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
    Function to calculate the mean and standard deviation of a given dataset.
    Inspired from 'ptrblck' at https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    Args:
       dataloader: (torch.dataloader) Dataloader of the dataset to calculate the mean and standard deviation.
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