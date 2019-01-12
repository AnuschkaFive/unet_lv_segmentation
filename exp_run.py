import search_hyperparams

if __name__ == '__main__':
    search_hyperparams.main(parent_dir="experiments/ENDO/model_ModelA_random/batch_size/batch_size_3/dropout", 
                            data_dir="data/320x320_heart_scans")
    search_hyperparams.main(parent_dir="experiments/ENDO/model_ModelA_random/batch_size/batch_size_3/weight_decay", 
                            data_dir="data/320x320_heart_scans")