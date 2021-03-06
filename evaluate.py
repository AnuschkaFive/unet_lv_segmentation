"""
Evaluates the model

Originally based on https://cs230-stanford.github.io/pytorch-getting-started.html and 
https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision.
"""

import argparse
import logging

import numpy as np
import torch

from torch.autograd import Variable
from pathlib import Path
import utils
import model.net as net
import model.data_loader as data_loader
import model.loss as loss
import model.metric as metric
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/320x320_heart_scans', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing hyper_params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, model_dir, hyper_params):
    """Evaluate the model.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: (Function) a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        model_dir: (string) Location to save the output images in.
        hyper_params: (HyperParams) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    
    # dictionary of results for every separate image
    all_single_metrics = {}
    
    with torch.no_grad():
        # compute metrics over the dataset
        for idx, (scan_batch, ground_truth_batch, ground_truth_filename) in enumerate(dataloader):
    
            # move to GPU if available
            if hyper_params.cuda:
                scan_batch, ground_truth_batch = scan_batch.to(device = hyper_params.cuda), ground_truth_batch.to(device = hyper_params.cuda)
            # fetch the next evaluation batch
            scan_batch, ground_truth_batch = Variable(scan_batch), Variable(ground_truth_batch)
            
            # compute model output
            output_batch = model(scan_batch)
            loss = loss_fn(output_batch, ground_truth_batch)
            
            if model_dir is not "":
                # compute loss for every single file of this batch
                for i in range(0, output_batch.shape[0]):
                    all_single_metrics[str(Path(ground_truth_filename[i]).parts[-1])] = {}
                    all_single_metrics[str(Path(ground_truth_filename[i]).parts[-1])]['loss'] = loss_fn(
                                            torch.index_select(output_batch, 0, torch.tensor([i], device='cuda:'+str(hyper_params.cuda)[-1] if hyper_params.cuda is not -1 else 'cpu')), 
                                            torch.index_select(ground_truth_batch, 0, torch.tensor([i], device='cuda:'+str(hyper_params.cuda)[-1] if hyper_params.cuda is not -1 else 'cpu'))
                                            ).item()
                    #print("Old shape: {}, new shape: {}".format(output_batch.shape, torch.index_select(output_batch, 0, torch.tensor([i], device='cuda:'+str(hyper_params.cuda)[-1] if hyper_params.cuda is not -1 else 'cpu')).shape))
            
            # make output_batch one-hot encoded
            output_batch = torch.sigmoid(output_batch)
            output_batch = (output_batch > hyper_params.treshold).float() 
    
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            ground_truth_batch = ground_truth_batch.data.cpu().numpy() 
            error_batch = np.absolute(np.subtract(ground_truth_batch, output_batch))
            
            # save result images
            if model_dir is not "":
                #print("Output batch shape: {}/{}".format(output_batch.shape[0], output_batch.shape))                
                for i in range(0, output_batch.shape[0]):
                    image = Image.fromarray(output_batch[i][0], 'I')                    
                    image.save(Path(model_dir) / str(Path(ground_truth_filename[i]).parts[-1]).replace(".png", "_SEG.png"))
                    image = Image.fromarray(error_batch[i][0], 'I')
                    image.save(Path(model_dir) / str(Path(ground_truth_filename[i]).parts[-1]).replace(".png", "_SEG_ERROR.png"))
            
            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, ground_truth_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            
            if model_dir is not "":
                # compute all metrics for every single file of this batch
                for i in range(0, output_batch.shape[0]):
                    #print("Original shape vs. Reduced shapes: {} / {}".format(output_batch.shape, output_batch[i:i+1][:].shape))
                    saved_loss = all_single_metrics[str(Path(ground_truth_filename[i]).parts[-1])]['loss']
                    all_single_metrics[str(Path(ground_truth_filename[i]).parts[-1])] = {metric: metrics[metric](output_batch[i:i+1][:], ground_truth_batch[i:i+1][:])
                             for metric in metrics}
                    all_single_metrics[str(Path(ground_truth_filename[i]).parts[-1])]['loss'] = saved_loss

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    
    if model_dir is not "":
        sorted_all_single_metrics = {}
        for key in sorted(all_single_metrics, key=lambda x: (all_single_metrics[x]['dsc']), reverse=True):
            sorted_all_single_metrics[key] = all_single_metrics[key]
        # write all_single_metrics to a json
        all_single_metrics_path = str(Path(model_dir) / "metrics_test_single_file_names.json")
        utils.save_dict_to_json(sorted_all_single_metrics, all_single_metrics_path)
    return metrics_mean


def main(data_dir, model_dir, restore_file):
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    json_path = Path(model_dir) / 'hyper_params.json'
    assert json_path.is_file(), "No json configuration file found at {}".format(json_path)
    hyper_params = utils.HyperParams(json_path)

    # use GPU if available
    #hyper_params.cuda = torch.cuda.is_available()     # use GPU is available
    hyper_params.cuda = torch.device('cuda:0') if torch.cuda.is_available() else -1

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if hyper_params.cuda is not -1: 
        with torch.cuda.device(str(hyper_params.cuda)[-1]):
            torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(Path(model_dir) / 'evaluate.log')

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], data_dir + hyper_params.augmentation, hyper_params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = getattr(net, hyper_params.model, None)
    assert model is not None, "Model {} couldn't be found!".format(hyper_params.model)
    model = model(hyper_params).to(device=hyper_params.cuda) if hyper_params.cuda is not -1 else model(hyper_params)
    
    loss_fn = getattr(loss, hyper_params.loss, None)
    assert loss_fn is not None, "Loss Fn {} couldn't be found!".format(hyper_params.loss)
    
    metrics_dict = metric.metrics_dict
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(str(Path(model_dir) / (restore_file + '.pth.tar')), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics_dict, model_dir, hyper_params)
    save_path = str(Path(model_dir) / "metrics_test_{}.json".format(restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    
if __name__ == '__main__':    
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.restore_file)