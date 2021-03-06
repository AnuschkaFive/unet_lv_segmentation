"""
Train the model, with or without k-fold Cross Validation.

Originally based on https://cs230-stanford.github.io/pytorch-getting-started.html and 
https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision.
"""

import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

import utils
import model.net as net
import model.loss as loss
import model.metric as metric
import model.data_loader as data_loader
import util.cross_validation as cv
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/320x320_heart_scans', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing hyper_params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training. Used for transfer learning.")

def train(model, optimizer, loss_fn, dataloader, metrics_dict, hyper_params):
    """
    Train the model.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics_dict: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        hyper_params: (Params) hyperparameters
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
            if hyper_params.cuda is not -1:
                scan_batch, ground_truth_batch = scan_batch.to(device = hyper_params.cuda), ground_truth_batch.to(device = hyper_params.cuda)
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
    #return metrics_mean, to append to list, to average later on for k-fold cross validation
    return metrics_mean


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
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar). Used for transfer learning.
    """
    # reload weights from restore_file if specified - used for transfer learning, so optimizer isn't transferred
    if restore_file is not None:
        restore_path = str(Path(model_dir) / (restore_file + '.pth.tar'))
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer=None)

    best_val_dsc = 0.0
    
    all_val_metrics = {}
    all_train_metrics = {}

    for epoch in range(hyper_params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, hyper_params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics_dict, hyper_params)
        all_train_metrics['epoch_{:02d}'.format(epoch + 1)] = train_metrics

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics_dict, '', hyper_params)
        all_val_metrics['epoch_{:02d}'.format(epoch + 1)] = val_metrics

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
    
    return (all_train_metrics, all_val_metrics)
        

def main(data_dir, model_dir, restore_file=None, k_folds=5):
    """
    Train a new model, either with k-fold Cross Validation (phase 1, model evaluation) or 
    without (phase 2, model building).
    Args:
        data_dir: (string) Directory containing the dataset.
        model_dir: (string) Directory containing hyper_params.json.
        restore_file: (string) Name of the file in model_dir containing weights to reload before training. Used for transfer learning."
        k_folds: (int) Specifies the number of folds for k-fold CV. -1, if no k-fold CV is used. Model will then be finally trained (phase 2, model building).
    """
    # Load the parameters from json file    
    json_path = Path(model_dir) / 'hyper_params.json'
    assert json_path.is_file(), "No json configuration file found at {}".format(json_path)
    hyper_params = utils.HyperParams(json_path)
    
    # Append directory name of data
    data_dir = data_dir + hyper_params.augmentation

    # Set the logger
    utils.set_logger(Path(model_dir) / 'train.log')
    writer = SummaryWriter(str(Path('tensor_log') / model_dir))

    # use GPU if available
    hyper_params.cuda = torch.device('cuda:0') if torch.cuda.is_available() else -1
   
    # For training with k-fold Cross Validation (model evaluation, phase 1):
    if k_folds != 0:
        idx = 0
        list_train_metrics = {}
        list_val_metrics = {}
        ds = data_loader.Heart2DSegmentationDataset(Path(data_dir) / 'train_heart_scans', hyper_params.endo_or_epi)
        for train_idx, val_idx in cv.k_folds(n_splits = k_folds, subjects = ds.__len__(), frames=1):
            idx += 1
            # Set the random seed for reproducible experiments
            torch.manual_seed(230)
            if hyper_params.cuda is not -1: 
                with torch.cuda.device(str(hyper_params.cuda)[-1]):
                    torch.cuda.manual_seed(230)
        
            # Define the model and optimizer
            model = getattr(net, hyper_params.model, None)
            assert model is not None, "Model {} couldn't be found!".format(hyper_params.model)
            model = model(hyper_params).to(device=hyper_params.cuda) if hyper_params.cuda is not -1 else model(hyper_params)
            
            optimizer = getattr(optim, hyper_params.optimizer, None)
            assert optimizer is not None, "Optimizer {} couldn't be found!".format(hyper_params.model)
            optimizer = optimizer(model.parameters(), lr=hyper_params.learning_rate, weight_decay=hyper_params.weight_decay)
        
            # fetch loss function and metrics
            loss_fn = getattr(loss, hyper_params.loss, None)
            assert loss_fn is not None, "Loss Fn {} couldn't be found!".format(hyper_params.loss)
            
            metrics_dict = metric.metrics_dict
            
            logging.info("Loading the datasets...")
            dataloaders = data_loader.fetch_dataloader(['train', 'val'], data_dir, hyper_params, train_idx, val_idx)
            
            # Save hyperparameters, now containing used mean and bias.
            cuda = hyper_params.cuda
            hyper_params.cuda = ""
            hyper_params.save(Path(model_dir) / 'hyper_params.json')
            hyper_params.cuda = cuda
            
            train_dl = dataloaders['train']
            val_dl = dataloaders['val']            
            logging.info("- done.")
            logging.info("For k-fold {}/{}:".format(idx, k_folds))
            logging.info("Starting training for {} epoch(s)".format(hyper_params.num_epochs)) 
            (all_train_metrics, all_val_metrics) = train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics_dict, hyper_params, model_dir, restore_file=restore_file)
            # Write to Tesorboard.
            for epoch_name in all_train_metrics:            
                writer.add_scalars('loss', {'k_fold_{}/train'.format(idx): all_train_metrics[epoch_name]['loss'], 'k_fold_{}/val'.format(idx): all_val_metrics[epoch_name]['loss']}, epoch_name[-2:])
                for metric_label in metrics_dict:
                    writer.add_scalars(metric_label, {'k_fold_{}/train'.format(idx): all_train_metrics[epoch_name][metric_label], 'k_fold_{}/val'.format(idx): all_val_metrics[epoch_name][metric_label]}, epoch_name[-2:])

            list_train_metrics['k_fold_{}'.format(idx)] = all_train_metrics
            list_val_metrics['k_fold_{}'.format(idx)] = all_val_metrics
        
        # Calculate average across all folds.
        list_train_metrics_mean = {}
        list_val_metrics_mean = {}        
        for epoch in list_train_metrics['k_fold_1']:
            avg_train_dict = {}
            avg_val_dict = {}
            for metric_name in metrics_dict:
                avg_train_dict[metric_name] = 0.0
                avg_train_dict['loss'] = 0.0
                avg_val_dict[metric_name] = 0.0
                avg_val_dict['loss'] = 0.0
                for k_fold in list_train_metrics:
                    avg_train_dict[metric_name] += list_train_metrics[k_fold][epoch][metric_name]
                    avg_train_dict['loss'] += list_train_metrics[k_fold][epoch]['loss']
                    avg_val_dict[metric_name] += list_val_metrics[k_fold][epoch][metric_name]
                    avg_val_dict['loss'] += list_val_metrics[k_fold][epoch]['loss']
                avg_train_dict[metric_name] /= k_folds
                avg_val_dict[metric_name] /= k_folds
                avg_train_dict['loss'] /= k_folds
                avg_val_dict['loss'] /= k_folds
            list_train_metrics_mean[epoch] = avg_train_dict
            list_val_metrics_mean[epoch] = avg_val_dict
        
        # Write to Tensorboard.
        for epoch_name in list_train_metrics_mean:            
            writer.add_scalars('loss', {'average/train': list_train_metrics_mean[epoch_name]['loss'], 'average/val': list_val_metrics_mean[epoch_name]['loss']}, epoch_name[-2:])
            for metric_label in metrics_dict:
                writer.add_scalars(metric_label, {'average/train': list_train_metrics_mean[epoch_name][metric_label], 'average/val': list_val_metrics_mean[epoch_name][metric_label]}, epoch_name[-2:])
        
        list_train_metrics['average'] = list_train_metrics_mean
        list_val_metrics['average'] = list_val_metrics_mean
        
        # Save all k-fold cross validation metrices in a Json.
        k_fold_train_path = str(Path(model_dir) / "metrics_k_fold_train.json")
        utils.save_dict_to_json(list_train_metrics, k_fold_train_path)
        k_fold_val_path = str(Path(model_dir) / "metrics_k_fold_val.json")
        utils.save_dict_to_json(list_val_metrics, k_fold_val_path)
        
        # Save last k-fold cross validation average metrics in a Json.
        last_json_path = str(Path(model_dir) / "metrics_k_fold_val_average_last.json")
        utils.save_dict_to_json(list_val_metrics['average']['epoch_{:02d}'.format(hyper_params.num_epochs)], last_json_path)
        
        # Save best k-fold cross validation average metrics in a Json.
        best_val_dsc = 0.0
        best_val_metrics_list = {}
        for x in list_val_metrics['average']:
            if list_val_metrics['average'][x]['dsc'] > best_val_dsc:
                best_val_dsc = list_val_metrics['average'][x]['dsc']
                best_val_metrics_list = list_val_metrics['average'][x]
        best_json_path = str(Path(model_dir) / "metrics_k_fold_val_average_best.json")
        utils.save_dict_to_json(best_val_metrics_list, best_json_path)
    
    # For final training of models, without k-fold Cross Validation (model building, phase 2):    
    else:
        # Set the random seed for reproducible experiments
        torch.manual_seed(230)
        if hyper_params.cuda is not -1: 
            with torch.cuda.device(str(hyper_params.cuda)[-1]):
                torch.cuda.manual_seed(230)
    
        # Define the model and optimizer
        model = getattr(net, hyper_params.model, None)
        assert model is not None, "Model {} couldn't be found!".format(hyper_params.model)
        model = model(hyper_params).to(device=hyper_params.cuda) if hyper_params.cuda is not -1 else model(hyper_params)
        
        optimizer = getattr(optim, hyper_params.optimizer, None)
        assert optimizer is not None, "Optimizer {} couldn't be found!".format(hyper_params.model)
        optimizer = optimizer(model.parameters(), lr=hyper_params.learning_rate, weight_decay=hyper_params.weight_decay)
    
        # fetch loss function and metrics
        loss_fn = getattr(loss, hyper_params.loss, None)
        assert loss_fn is not None, "Loss Fn {} couldn't be found!".format(hyper_params.loss)
        
        metrics_dict = metric.metrics_dict
        
        # fetch dataloaders
        logging.info("Loading the datasets...")
        dataloaders = data_loader.fetch_dataloader(['train', 'test'], data_dir, hyper_params)
        
        # Save hyperparameters, now containing used mean and bias.
        cuda = hyper_params.cuda
        hyper_params.cuda = ""
        hyper_params.save(Path(model_dir) / 'hyper_params.json')
        hyper_params.cuda = cuda
            
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']        
        logging.info("- done.")
        logging.info("Starting training for {} epoch(s)".format(hyper_params.num_epochs))        
        # Train the model
        (all_train_metrics, all_test_metrics) = train_and_evaluate(model, train_dl, test_dl, optimizer, loss_fn, metrics_dict, hyper_params, model_dir, restore_file)
        
        # Write results to Jsons.
        train_path = str(Path(model_dir) / "metrics_train.json")
        utils.save_dict_to_json(all_train_metrics, train_path)
        test_path = str(Path(model_dir) / "metrics_test.json")
        utils.save_dict_to_json(all_test_metrics, test_path)
        
        # Write results to Tensorboard.
        for epoch_name in all_train_metrics:            
            writer.add_scalars('loss', {'final/train': all_train_metrics[epoch_name]['loss'], 'final/test': all_test_metrics[epoch_name]['loss']}, epoch_name[-2:])
            for metric_label in metrics_dict:
                writer.add_scalars(metric_label, {'final/train': all_train_metrics[epoch_name][metric_label], 'final/test': all_test_metrics[epoch_name][metric_label]}, epoch_name[-2:])
    
    writer.export_scalars_to_json(str(Path(model_dir) / "all_scalars.json"))    
    writer.close()
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.restore_file)    