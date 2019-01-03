#https://cs230-stanford.github.io/logging-hyperparams.html

"""Peform hyperparemeters search"""

import argparse
from pathlib import Path
from subprocess import check_call
import sys
import train

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing hyper_params.json')
parser.add_argument('--data_dir', default='data/320x320_heart_scans', help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, hyper_params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        parent_dir: (string) directory containing hyper_params.json
        data_dir: (string) directory containing the dataset
        job_name: (string)
        hyper_params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = Path(parent_dir) / job_name
    if not Path(model_dir).exists():
        Path(model_dir).mkdir()

    # Write parameters in json file
    json_path = Path(model_dir) / 'hyper_params.json'
    hyper_params.save(json_path)

    # Launch training with this config
#    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
#                                                                                   data_dir=data_dir)
#    print(cmd)
#    check_call(cmd, shell=True)
    train.main(model_dir=model_dir, data_dir=data_dir)

def main(parent_dir, data_dir):
    json_path = Path(parent_dir) / 'hyper_params.json'
    assert json_path.is_file(), "No json configuration file found at {}".format(json_path)
    hyper_params = utils.HyperParams(json_path)

    hyper_param_key_to_vary = ''
    
    # Perform hypersearch over one parameter
    for hyper_param_key in hyper_params.dict:
        if isinstance(hyper_params.dict[hyper_param_key], list):
            hyper_param_key_to_vary = hyper_param_key
            
    #learning_rates = [1e-4, 1e-3, 1e-2]
    assert hyper_param_key_to_vary is not '', "No hyperparameter with a list (variable values) found!"

#    hyper_param_values = hyper_params.dict[hyper_param_key_to_vary]
#    hyper_params.dict[hyper_param_key_to_vary] = 0.

    #for learning_rate in learning_rates:
    for hyper_param_value in hyper_params.dict[hyper_param_key_to_vary]:
        # Modify the relevant parameter in params
        #hyper_params.learning_rate = learning_rate
        hyper_params.dict[hyper_param_key_to_vary] = hyper_param_value
        # Launch job (name has to be unique)
        #job_name = "learning_rate_{}".format(learning_rate)
        job_name = "{}_{}".format(hyper_param_key_to_vary, hyper_param_value)
        launch_training_job(parent_dir, data_dir, job_name, hyper_params)

if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    main(args.parent_dir, args.data_dir)
   