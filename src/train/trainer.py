from __future__ import absolute_import
import torch
from data.REDS_loader import getDataLoader
from model.generator import Generator
from train.earlyStop import EarlyStopping
from train.train_one import trainOne
from train.validate import validate
from train.loss import Loss
def train(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dlTrain = getDataLoader(conf, "train")
    dlVal = getDataLoader(conf, "val")

    gen = Generator(conf).to(device)

    optimizerGen = torch.optim.Adam(gen.parameters(),lr=conf["TRAINING"].getfloat("generator_learning_rate"))

    genEr = EarlyStopping(gen, conf["TRAINING"].getfloat("generator_es_minIncrement"), conf["TRAINING"].getint("generator_es_epochsNoImprovement"), conf["TRAINING"]["name_model"])

    lossfac = Loss()

    losses=[]

    for i in range(conf["TRAINING"].getint("max_epochs")):

        loss=trainOne(gen, dlTrain, optimizerGen, device, lossfac)

        validate()

        print("Epoch %f: loss=%f ",i,loss)
        losses.append(loss)
    return losses
