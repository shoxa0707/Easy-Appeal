from nemo.collections.nlp.models.text_classification import TextClassificationModel

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input text file", type=str, default="example.txt")
parser.add_argument("-o", "--output", help="output text file", type=str, default="output.txt")
parser.add_argument("-m", "--model", help="path to nemo model", type=str, default="model.nemo")
args = parser.parse_args()

cfg = OmegaConf.load('config/inference_config.yaml')
cfg.model.nemo_path = args.model
trainer = pl.Trainer(strategy=None, **cfg.trainer)


model = TextClassificationModel(cfg.model, trainer=trainer)
model.eval()
with open(args.input) as f:
    infer = f.read().split('\n')
    
results = model.classifytext(queries=infer, batch_size=32, max_seq_length=512)

with open(args.output, 'w') as f:
    f.write('\n'.join([cfg.classes[i] for i in results]))
