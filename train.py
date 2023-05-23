import os
import glob
import json
import torch
import random
import numpy as np

from data import *
from BERT import *
from utils.remi import *
from utils.utils import *
from utils.vocab import *

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor


### load config file
with open("./config.json", "r") as f:
    config = json.load(f)


### fix random seed
random_seed = config["random_seed"]

# it may slow computing performance
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

random.seed(random_seed)
np.random.seed(random_seed)

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


#### Pytorch-lightning 1.9.4 for interal precision
torch.set_float32_matmul_precision("high")


#### initialize model with GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


### load file_list
folder_path = "./data/lmd_full_remi/"
folder_list = glob.glob(os.path.join(folder_path, "*"))

train_folder, val_folder, test_folder = dataset_split(folder_list)

train_files = folder_to_file(train_folder)
val_files = folder_to_file(val_folder)
test_files = folder_to_file(test_folder)

random.shuffle(train_files)

print(
    f"train_files : {len(train_files)}, val_files : {len(val_files)}, test_files : {len(test_files)}"
)


### load dataloader
train_module = DataModule(
    train_files,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    masking=config["masking"],
    replace=config["replace"],
    phase="train",
)

val_module = DataModule(
    val_files,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    phase="val",
)

test_module = DataModule(
    test_files,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    phase="test",
)

train_set = train_module.return_dataloader()
val_set = val_module.return_dataloader()
test_set = test_module.return_dataloader()


### define model
model = BERT_Lightning(
    dim=config["dim"],
    depth=config["depth"],
    heads=config["heads"],
    dim_head=int(config["dim"] / config["heads"]),
    mlp_dim=int(4 * config["dim"]),
    max_len=config["max_len"],
    rate=config["rate"],
    loss_weights=config["loss_weights"],
    lr=config["lr"],
    warm_up=config["warm_up"],
    temp=config["temp"],
    mode=config["mode"],
).to(device)


### callback functions
model_name = [key + "_" + str(value) for key, value in config.items()]
model_name = "-".join(param for param in model_name)
model_name = "BERT-" + model_name + "-{epoch}-{val_loss:.4f}"

lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint = ModelCheckpoint(
    filename=model_name,
    dirpath="./model/",
    monitor="val_loss",
    mode="min",
)

### train model
trainer = pl.Trainer(
    num_nodes=1,
    precision=16,
    max_epochs=config["epochs"],
    accelerator="gpu",
    devices=config["gpus"],
    callbacks=[lr_monitor, checkpoint],
)

trainer.fit(model, train_set, val_set)
