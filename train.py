from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager

import os
import wget 
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import OmegaConf
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="folder where the data is stored", type=str, default="Data")
parser.add_argument("-w", "--work", help="folder where the config file is stored. Config file is downloaded from link", type=str, default="Work")
args = parser.parse_args()


DATA_DIR = args.data
WORK_DIR = args.work

# download the model's configuration file 
MODEL_CONFIG = "text_classification_config.yaml"
CONFIG_DIR = WORK_DIR + '/configs/'

os.makedirs(CONFIG_DIR, exist_ok=True)
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
if not os.path.exists(CONFIG_DIR + MODEL_CONFIG):
    print('Downloading config file...')
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/nlp/text_classification/conf/' + MODEL_CONFIG, CONFIG_DIR)
    print('Config file downloaded!')
else:
    print ('config file already exists')
config_path = f'{WORK_DIR}/configs/{MODEL_CONFIG}'
config = OmegaConf.load(config_path)

config.model.dataset.num_classes = 9
config.model.train_ds.file_path = os.path.join(DATA_DIR, 'train.txt')
config.model.validation_ds.file_path = os.path.join(DATA_DIR, 'test.txt')
# Name of the .nemo file where trained model will be saved.
config.save_to = 'model.nemo'
config.model.train_ds.batch_size = 8

print("Train dataloader's config: \n")
# OmegaConf.to_yaml() is used to create a proper format for printing the train dataloader's config
# You may change other params like batch size or the number of samples to be considered (-1 means all the samples)
print(OmegaConf.to_yaml(config.model.train_ds))

print("Trainer config - \n")
# OmegaConf.to_yaml() is used to create a proper format for printing the trainer config
print(OmegaConf.to_yaml(config.trainer))

# lets modify some trainer configs
# checks if we have GPU available and uses it
config.trainer.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 1

# for mixed precision training, uncomment the lines below (precision should be set to 16 and amp_level to O1):
# config.trainer.precision = 16
# config.trainer.amp_level = O1

# disable distributed training when using Colab to prevent the errors
config.trainer.strategy = None

# setup max number of steps to reduce training time for demonstration purposes of this tutorial
# Training stops when max_step or max_epochs is reached (earliest)
config.trainer.max_epochs = 10

# The experiment manager of a trainer object can not be set twice. We repeat the trainer creation code again here to prevent getting error when this cell is executed more than once. 
trainer = pl.Trainer(**config.trainer)
exp_dir = exp_manager(trainer, config.exp_manager)

# specify the BERT-like model, you want to use
# set the `model.language_modelpretrained_model_name' parameter in the config to the model you want to use
config.model.language_model.pretrained_model_name = "bert-base-uncased"

model = nemo_nlp.models.TextClassificationModel(cfg=config.model, trainer=trainer)

# start model training
trainer.fit(model)
model.save_to(config.save_to)

# extract the path of the best checkpoint from the training, you may update it to any checkpoint
checkpoint_path = trainer.checkpoint_callback.best_model_path
# Create an evaluation model and load the checkpoint
eval_model = nemo_nlp.models.TextClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

# create a dataloader config for evaluation, the same data file provided in validation_ds is used here
# file_path can get updated with any file
eval_config = OmegaConf.create({'file_path': config.model.validation_ds.file_path, 'batch_size': 64, 'shuffle': False, 'num_samples': -1})
eval_model.setup_test_data(test_data_config=eval_config)
#eval_dataloader = eval_model._create_dataloader_from_config(cfg=eval_config, mode='test')

# a new trainer is created to show how to evaluate a checkpoint from an already trained model
# create a copy of the trainer config and update it to be used for final evaluation
eval_trainer_cfg = config.trainer.copy()
eval_trainer_cfg.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu' # it is safer to perform evaluation on single GPU as PT is buggy with the last batch on multi-GPUs
eval_trainer_cfg.strategy = None # 'ddp' is buggy with test process in the current PT, it looks like it has been fixed in the latest master
eval_trainer = pl.Trainer(**eval_trainer_cfg)
eval_trainer.test(model=eval_model, verbose=False) # test_dataloaders=eval_dataloader
