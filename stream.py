import streamlit as st
from nemo.collections.nlp.models.text_classification import TextClassificationModel

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

cfg = OmegaConf.load('inference_config.yaml')
trainer = pl.Trainer(strategy=None, **cfg.trainer)


model = TextClassificationModel(cfg.model, trainer=trainer)
model.eval()

st.header("Text classification with Nvidia Nemo")

txt = st.text_area('Enter your appeal', height=200)

if txt != '':
    st.write('Result:', cfg.classes[model.classifytext(queries=[txt], batch_size=32, max_seq_length=512)[0]])
    