import os

os.system(".venv/bin/python topobench/run.py dataset=graph/NCI1 model=simplicial/gsan model.backbone.n_layers=1 trainer.max_epochs=1 logger=csv trainer.accelerator=cpu trainer.devices=auto")
