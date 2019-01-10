import search_hyperparams

if __name__ == '__main__':
    search_hyperparams.main(parent_dir="experiments/ENDO/model_ModelA_random/learning_rate/loss_soft_dice_loss", 
                            data_dir="data/320x320_heart_scans")
    search_hyperparams.main(parent_dir="experiments/EPI/model_ModelA_random/learning_rate/loss_soft_dice_loss", 
                            data_dir="data/320x320_heart_scans")
