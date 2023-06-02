from __future__ import absolute_import
import torch
from data.REDS_loader import getDataLoader
from model.generator import Generator
from train.earlyStop import EarlyStopping
from train.train_one import trainOne
from train.validate import validate
from train.loss import Loss
from pathlib import Path
import os

def lookForResume(conf) -> bool:
    path = os.path.join("trained_models", conf["TRAINING"]["name_model"])
    file = Path(path)
    return file.exists()

def train(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dlTrain = getDataLoader(conf, "train")
    dlVal = getDataLoader(conf, "val")

    gen = Generator(conf).to(device)

    if(lookForResume(conf)):
        gen.load_state_dict(torch.load(os.path.join("trained_models", conf["TRAINING"]["name_model"])))
        print("Checkpoint found! Resuming the training")
    else:
        print("No previous checkpoint found. Starting a training from scratch")

    optimizerGen = torch.optim.Adam(gen.parameters(),lr=conf["TRAINING"].getfloat("generator_learning_rate"))

    genEr = EarlyStopping(gen, conf["TRAINING"].getfloat("generator_es_minIncrement"), conf["TRAINING"].getint("generator_es_epochsNoImprovement"), conf["TRAINING"]["name_model"])

    lossfac = Loss()

    for i in range(conf["TRAINING"].getint("max_epochs")):

        avgtrain=trainOne(gen, dlTrain, optimizerGen, device, lossfac, conf)

        avgval = validate(gen, dlVal, lossfac, device)

        stop = genEr(avgval)
        print(f"[Epoch {i}]: avg training loss={avgtrain}       validation-loss={avgval}    No improvements since {genEr.getNoImprovement()} epoch(s)")

        if(stop):
            break
    

