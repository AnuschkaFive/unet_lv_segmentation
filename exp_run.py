import search_hyperparams

if __name__ == '__main__':
    search_hyperparams.main(parent_dir="experiments/ENDO/model_UNetOriginal_random/learning_rate/loss_cross_entropy_loss", data_dir="data/320x320_heart_scans")
    search_hyperparams.main(parent_dir="experiments/ENDO/model_UNetOriginal_random/learning_rate/loss_soft_dice_loss", data_dir="data/320x320_heart_scans")
    search_hyperparams.main(parent_dir="experiments/ENDO/model_UNetOriginal_random/learning_rate/loss_cross_entropy_loss/learning_rate_0.0001/dropout", data_dir="data/320x320_heart_scans")
