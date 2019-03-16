"""
Preprocess the dataset, applying various options, and saving it to a new location.

Originally based on https://cs230-stanford.github.io/pytorch-getting-started.html and 
https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision.
"""
import argparse
import random
import torchvision.transforms as transforms
import numpy as np
import SimpleITK as sitk

from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMG_SIZE = 320

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/heart_scans', help="Directory with the Heart Segmentation dataset")
parser.add_argument('--output_dir', default='data/{}x{}_heart_scans'.format(IMG_SIZE, IMG_SIZE), help="Where to write the new data")

def stratify_filenames(filenames):
    """
    Sorts the filenames in a stratified way.
    Args:
        filenames: (string-Array) The paths to the files (only ENDO, EPI or ORIG).
    Returns:
        out: (string-Array) Copy of the input array, sorted in a stratified way.
    """
    # Array mit Unterarray: [[alle t00 und sl00], [alle ^t00 und sl00], ..., [alle t00 und sl09], [alle ^t00 und sl09]]
    categories = []
    
    # TODO: find max slNUMB, for range max!
    for numb in range(0, 11):
        subcategory = [x for x in filenames if (("ti00" in x) and ("sl{:02d}".format(numb) in x))]
        categories.append(subcategory)
        subcategory = [x for x in filenames if (("ti00" not in x) and ("sl{:02d}".format(numb) in x))]
        categories.append(subcategory)
        
    # Make all sublists the same length, then cast to numpy array for easier calculations.
    length = len(sorted(categories,key=len, reverse=True)[0])
    categories_np = np.array([subcat+[None]*(length-len(subcat)) for subcat in categories])

    assert (categories_np.size - (categories_np == None).sum()) == len(filenames), "Categories has {} elements, but Filenames has {}.".format(categories_np.size - (categories_np == None).sum(), len(filenames))

    # Shuffle categories.    
    np.random.seed(230)
    np.random.shuffle(categories_np)

    # Shuffle subcategories.
    for subcat in categories_np:
        np.random.shuffle(subcat)
    
    # Flatten column-first.
    categories_np = categories_np.flatten("F")
    
    # Delete empty elements.
    categories_np = np.delete(categories_np, np.where(categories_np == None))
    
    assert categories_np.size == len(filenames), "Categories has {} elements, but Filenames has {}.".format(categories_np.size, len(filenames))
    
    # Cast back to Python list and return.
    return categories_np.tolist()


def create_missing_endo_or_epi(data_dir):
    """
    Checks whether ORIG scan files have either only ENDO ground truth file
    or EPI ground truth file. If not, and creates black image of the same size.
    Args:
        data_dir: (string) The path to the directory with the original dataset.
    """
    print('Checking for missing ENDO or EPI ...')
    # Create list of all ENDO and EPI ground truths, each. Rename the '_ENDO'/'_EPI' label.
    endo_filenames = [str(path).replace('_ENDO', '_PLACEHOLDER') for path in Path(data_dir).glob('**/*_ENDO*.png')] 
    epi_filenames = [str(path).replace('_EPI', '_PLACEHOLDER') for path in Path(data_dir).glob('**/*_EPI*.png')]
    # Make union, then subtract each of the 2 lists (for ENDO and EPI).
    endo_epi_union = set().union(endo_filenames, epi_filenames)
    union_without_endo = endo_epi_union.difference(endo_filenames)
    union_without_epi = endo_epi_union.difference(epi_filenames)
    
    no_of_endo = 0
    # Create a new black image for the missing ENDO/EPI ground truths.    
    for missing_endo in union_without_endo:
        corresponding_epi = Image.open(missing_endo.replace('_PLACEHOLDER', '_EPI'))
        new_endo = Image.new(corresponding_epi.mode, corresponding_epi.size)
        new_endo.save(str(missing_endo).replace('_PLACEHOLDER', '_ENDO'), 'PNG')
        no_of_endo += 1
        
    no_of_epi = 0
    for missing_epi in union_without_epi:
        corresponding_endo = Image.open(missing_epi.replace('_PLACEHOLDER', '_ENDO'))
        new_epi = Image.new(corresponding_endo.mode, corresponding_endo.size)
        new_epi.save(str(missing_epi).replace('_PLACEHOLDER', '_EPI'), 'PNG')
        no_of_epi += 1
        
    print('   Done. Found {} missing ENDO and {} missing EPI.'.format(no_of_endo, no_of_epi))
    
    
def split_filenames_into_train_test(data_dir, split_ratio):
    """ 
    Shuffles the filenames of all ENDO ground truth files and splits them into 
    a training and a test set. Then adds the corresponding ORIG scan files 
    and the EPI ground truths to both sets.
    Args:
        data_dir: (string) The path to the directory with the original dataset.
        split_ratio: (float) The ratio of training data vs. test data.
    Returns:
        out: (dict) Filenames for train and test set.
    """
    print('Split filenames into train and test set ...')
    
    # Get all filenames that contain '_ENDO' and '.png'.
    filenames = [str(path) for path in Path(data_dir).glob('**/*_ENDO*.png')]  
    
#    # Set a random seed, to create reproducible results. Then sort the filenames,
#    # before randomly shuffling them.
#    random.seed(230)
#    filenames.sort()
#    random.shuffle(filenames) 
    
    # Sort filenames in a stratified way.
    filenames = stratify_filenames(filenames)
    
    # Split the filenames into train and test set.
    split = int(split_ratio * len(filenames))
    train_filenames = filenames[:split]
    test_filenames = filenames[split:]
    
    assert len(filenames) == len(train_filenames) + len(test_filenames), '{} filenames total, but only {} in train and {} in test set'.format(len(filenames), len(train_filenames), len(test_filenames))
    
    # Add the ORIG scans and the EPI ground truths to the list of ENDO ground truths.
    for i in range(0, len(train_filenames)):
        epi_file = train_filenames[i].replace('_ENDO', '_EPI')
        scan_file = train_filenames[i].replace('_ENDO', '_ORIG')
        train_filenames.append(epi_file)
        train_filenames.append(scan_file)
        
    for i in range(0, len(test_filenames)):
        epi_file = test_filenames[i].replace('_ENDO', '_EPI')
        scan_file = test_filenames[i].replace('_ENDO', '_ORIG')
        test_filenames.append(epi_file)
        test_filenames.append(scan_file)    
    
    assert len(filenames) * 3 == (len(train_filenames) + len(test_filenames)), 'Some EPI or ORIGINAL files are missing, {}/{}'.format((len(train_filenames) + len(test_filenames)), len(filenames) * 3)
    
    # Make a dictionary with all relevant filenames.
    filenames = {'train': train_filenames,
                 'test': test_filenames}
    
    print('    Done.')
    
    return filenames


def resize_and_save(filename, output_dir, img_size=IMG_SIZE):
    """
    Resize the image contained in `filename` and save it to the `output_dir`.
    Args:
        filename: (string) Filename of an image.
        output_dir: (string) Location to save the resized file.
        img_size: (int) New size of the image.
    """
    image = Image.open(filename)

    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((img_size, img_size), Image.BILINEAR)
    image.save(Path(output_dir) / Path(filename).parts[-1])


def find_ROI_crop_area(file_dir, margin=0):
    """
    Finds the largest possible bounding box and crops all images accordingly.
    Args:
        file_dir: (string) Location of the images to be cropped.
    """
    # Find largest field of EPI.
    max_EPI = np.zeros((320, 320), np.int64)
    for split in ['train', 'test']:
        file_dir_split = Path(file_dir) / '{}_heart_scans'.format(split)
        filenames = [str(path) for path in Path(file_dir_split).glob('**/*_EPI*.png')]
        for filename in filenames:
            curr_EPI = Image.open(filename)
            curr_EPI = np.array(curr_EPI)
            max_EPI = np.maximum(curr_EPI, max_EPI)
            
    # Find bounding box.
    max_EPI = (max_EPI > 0)
    rows = np.any(max_EPI, axis=1)
    cols = np.any(max_EPI, axis=0)
    rmin, rmax = np.argmax(rows), max_EPI.shape[0] - 1 - np.argmax(np.flipud(rows))
    cmin, cmax = np.argmax(cols), max_EPI.shape[1] - 1 - np.argmax(np.flipud(cols))
    
    # Make bounding box squared.
    rmin = min(rmin, cmin)
    cmin = min(rmin, cmin)
    rmax = max(rmax, cmax)
    cmax = max(rmax, cmax)       
        
    return rmin - margin, rmax + margin, cmin - margin, cmax + margin   


def n4_bias_correction(scan_filename):
    """
    Applies N4 Bias Field Correction to a scan image.
    Args:
        scan_filename: (string) The filename of the ORIG scan image.
    """
    origScan = sitk.ReadImage(scan_filename)
    scan = sitk.Cast(origScan, sitk.sitkFloat64)
    maskImage = sitk.OtsuThreshold(scan, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    result = corrector.Execute(scan, maskImage)
    result = sitk.Cast(result, origScan.GetPixelID())
    sitk.WriteImage(result, scan_filename)    

def transform_and_save(scan_filename, output_dir, rot=True, h_flip=True, v_flip=True, scale=True):
    """
    Transforms the ORIG scan image and the corresponding ENDO and EPI ground truth
    images. Saves the transformations in the same folder as the scan image, with
    '_AUG{number}' added to the filename.
    Args:
        scan_filename: (string) The filename of the ORIG scan image.
        output_dir: (string) Where to save the transformed pictures.
    """
    scan = Image.open(scan_filename) 
    endo = Image.open(scan_filename.replace("_ORIG", "_ENDO"))
    epi = Image.open(scan_filename.replace("_ORIG", "_EPI"))
    
    random.seed(230)
    if rot:
    	for i in range(0, 3):
            degree = random.randint(70 + (i * 90), 110 + (i * 90))
            scan_rotated = scan.rotate(degree, resample=Image.BILINEAR, expand=0)
            endo_rotated = endo.rotate(degree, resample=Image.BILINEAR, expand=0)
            epi_rotated = epi.rotate(degree, resample=Image.BILINEAR, expand=0)
            scan_rotated.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_ORIG_AUG{}".format(i)))
            endo_rotated.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_ENDO_AUG{}".format(i)))
            epi_rotated.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_EPI_AUG{}".format(i)))
    
    #Horizontal Flip
    if h_flip:
        scan_flippedH =scan.transpose(Image.FLIP_TOP_BOTTOM)
        endo_flippedH = endo.transpose(Image.FLIP_TOP_BOTTOM)
        epi_flippedH = epi.transpose(Image.FLIP_TOP_BOTTOM)
        scan_flippedH.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_ORIG_AUG{}".format(3)))
        endo_flippedH.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_ENDO_AUG{}".format(3)))
        epi_flippedH.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_EPI_AUG{}".format(3)))
    
    #Vertical Flip
    if v_flip:
        scan_flippedV =scan.transpose(Image.FLIP_LEFT_RIGHT)
        endo_flippedV = endo.transpose(Image.FLIP_LEFT_RIGHT)
        epi_flippedV = epi.transpose(Image.FLIP_LEFT_RIGHT)
        scan_flippedV.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_ORIG_AUG{}".format(4)))
        endo_flippedV.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_ENDO_AUG{}".format(4)))
        epi_flippedV.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_EPI_AUG{}".format(4)))
    
    #Scaling
    if scale:
        crop_percentage = random.uniform(0.15, 0.25)
        crop_pixel = int(crop_percentage * IMG_SIZE)
        scan_cropped = scan.crop((crop_pixel, crop_pixel, IMG_SIZE - crop_pixel, IMG_SIZE - crop_pixel))
        endo_cropped = endo.crop((crop_pixel, crop_pixel, IMG_SIZE - crop_pixel, IMG_SIZE - crop_pixel))
        epi_cropped = epi.crop((crop_pixel, crop_pixel, IMG_SIZE - crop_pixel, IMG_SIZE - crop_pixel))
        scan_scaled = scan_cropped.resize([IMG_SIZE,IMG_SIZE],Image.ANTIALIAS)
        endo_scaled = endo_cropped.resize([IMG_SIZE,IMG_SIZE],Image.ANTIALIAS)
        epi_scaled = epi_cropped.resize([IMG_SIZE,IMG_SIZE],Image.ANTIALIAS)
        scan_scaled.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_ORIG_AUG{}".format(5)))
        endo_scaled.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_ENDO_AUG{}".format(5)))
        epi_scaled.save(Path(output_dir) / Path(scan_filename).parts[-1].replace("_ORIG", "_EPI_AUG{}".format(5)))

def main(data_dir, output_dir, n4=False, roi=False, rot=True, h_flip=False, v_flip=False, scale=False):   
    """
    Builds the dataset by creating the necessary directories and calling the
    necessary functions.
    Args:
        data_dir: (string) The path to the directory with the original dataset.
        output_dir: (string) Where to save the preprocessed dataset.
        n4: (bool) Whether to apply N4 Bias Field Correction to the ORIG scan images.
        roi: (bool) Whether to crop all images to thei ROI.
        rot: (bool) Whether to augment the data by rotating it.
        h_flip: (bool) Whether to augment the data by horizontally flipping it.
        v_flip: (bool) Whether to augment the data by vertically flipping it.
        scale: (bool) Whether to augment the data by scaling it.
    """
    assert Path(data_dir).is_dir(), "Couldn't find the dataset at {}".format(data_dir)
    
    create_missing_endo_or_epi(data_dir)
    
    filenames = split_filenames_into_train_test(data_dir, 0.8333)
    
    # Rename location of preprocessed dataset to make compatible with JSON hyperparameter configuration later on.
    if n4:
        #add "_N4"
        output_dir = output_dir + "_N4"
    if roi:
        #add "_ROI"
        output_dir = output_dir + "_ROI"
    if not (rot or h_flip or v_flip or scale):
        #add "_none"
        output_dir = output_dir + "_none"
    else:
        if rot:
            #add "_rot"
            output_dir = output_dir + "_rot"
        if h_flip or v_flip:
            #add "_flip"
            output_dir = output_dir + "_flip"
        if scale:
            #add "_scale"
            output_dir = output_dir + "_scale"

    # Define the data directories    
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()
    else:
        print("Warning: output dir {} already exists".format(output_dir))

    # Preprocess train and test
    for split in ['train', 'test']:
        output_dir_split = Path(output_dir) / '{}_heart_scans'.format(split)
        if not Path(output_dir_split).exists():
            Path(output_dir_split).mkdir()
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Resizing {} data, saving resized data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, img_size=IMG_SIZE)          
         
        if n4:
            print("Applying N4 bias fiel correction to {} data, saving data to {}".format(split, output_dir_split))
            for filename in tqdm(filenames[split]):
                if "_ORIG" in filename:
                    n4_bias_correction(str(Path(output_dir_split) / Path(filename).parts[-1]))               
        
        print("Augmenting {} data, saving augmented data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            if "_ORIG" in filename and not "_AUG" in filename:
                transform_and_save(str(Path(output_dir_split) / Path(filename).parts[-1]), output_dir_split, rot=rot, h_flip=h_flip, v_flip=v_flip, scale=scale)
    
    if roi:
        print("Get Bounding Box coordinates")
        (rmin, rmax, cmin, cmax) = find_ROI_crop_area(output_dir, margin=5)
        print("RMin: {}, RMax: {}, CMin: {}, CMax: {}".format(rmin, rmax, cmin, cmax))
    
        for split in ['train', 'test']:
            output_dir_split = Path(output_dir) / '{}_heart_scans'.format(split)
            output_filenames = [str(path) for path in Path(output_dir_split).glob('**/*.png')]
            print("Crop to Bounding Box and resize {} again, saving cropped and resized data to {}".format(split, output_dir_split))
            for filename in tqdm(output_filenames):
                resize_and_save(filename, output_dir_split, img_size=IMG_SIZE, crop=(cmin, rmin, cmax, rmax))

    print("Done building dataset.")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
    
    
#### Test Section ####
#filenames = [str(path) for path in Path('data/heart_scans').glob('**/*_ENDO*.png')]
#
#print(stratify_filenames(filenames))
#####################