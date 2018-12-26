"""Train the model"""

import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from pathlib import Path

import utils
import model.net as net
import model.loss as loss
import model.metric as metric
import model.data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/320x320_heart_scans', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing hyper_params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

def train(model, optimizer, loss_fn, dataloader, metrics_dict, hyper_params):
    """
    Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics_dict: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (scan_batch, ground_truth_batch, _) in enumerate(dataloader):
            # move to GPU if available
            if hyper_params.cuda:
                scan_batch, ground_truth_batch = scan_batch.cuda(async=True), ground_truth_batch.cuda(async=True)
            # convert to torch Variables
            scan_batch, ground_truth_batch = Variable(scan_batch), Variable(ground_truth_batch)

            # compute model output and loss
            output_batch = model(scan_batch)
            loss = loss_fn(output_batch, ground_truth_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()
            
            # make output_batch one-hot encoded
            output_batch = torch.sigmoid(output_batch)
            output_batch = (output_batch > hyper_params.treshold).float() 
            
            # Evaluate summaries only once in a while
            #if i % hyper_params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch_np = output_batch.data.cpu().numpy()
            ground_truth_batch_np = ground_truth_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric:metrics_dict[metric](output_batch_np, ground_truth_batch_np)
                                 for metric in metrics_dict}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics_dict, hyper_params, model_dir,
                       restore_file=None):
    """
    Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics_dict: (dict) a dictionary of functions that compute a metric using the outputs and ground truths of each batch
        hyper_params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = str(Path(args.model_dir) / (restore_file + '.pth.tar'))
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_dsc = 0.0

    for epoch in range(hyper_params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, hyper_params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics_dict, hyper_params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics_dict, '', hyper_params)

        val_dsc = val_metrics['dsc']
        is_best = val_dsc>=best_val_dsc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best DSC")
            best_val_dsc = val_dsc

            # Save best val metrics in a json file in the model directory
            best_json_path = str(Path(model_dir) / "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = str(Path(model_dir) / "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


def main(data_dir, model_dir, restore_file):
    # Load the parameters from json file    
    json_path = Path(model_dir) / 'hyper_params.json'
    assert json_path.is_file(), "No json configuration file found at {}".format(json_path)
    hyper_params = utils.HyperParams(json_path)

    # use GPU if available
    hyper_params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if hyper_params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(Path(model_dir) / 'train.log')

    # Create the input data pipeline
    logging.info("Loading the datasets...")
        
    #if k_fold is not 1:
    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'test'], data_dir, hyper_params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model and optimizer
    model = getattr(net, hyper_params.model, None)
    assert model is not None, "Model {} couldn't be found!".format(hyper_params.model)
    model = model(hyper_params).cuda() if hyper_params.cuda else model(hyper_params)
    
    optimizer = getattr(optim, hyper_params.optimizer, None)
    assert optimizer is not None, "Optimizer {} couldn't be found!".format(hyper_params.model)
    optimizer = optimizer(model.parameters(), lr=hyper_params.learning_rate)

    # fetch loss function and metrics
    loss_fn = getattr(loss, hyper_params.loss, None)
    assert loss_fn is not None, "Loss Fn {} couldn't be found!".format(hyper_params.loss)
    
    metrics_dict = metric.metrics_dict
    
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(hyper_params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics_dict, hyper_params, model_dir, restore_file)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.restore_file)    