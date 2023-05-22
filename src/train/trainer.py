from __future__ import absolute_import
import torch
from data.REDS_loader import getDataLoader
from model.discriminator import Discriminator
from model.generator import Generator
from train.earlyStop import EarlyStopping
from train.train_one import trainOne_generator, trainOne_discriminator
from train.validate import validate
from train.loss import generatorLoss
def train(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dlTrain = getDataLoader(conf, "train")
    dlVal = getDataLoader(conf, "val")

    gen = Generator(conf).to(device)
    disc = Discriminator(conf).to(device)

    optimizerGen = torch.optim.Adam(gen.parameters(),lr=conf["TRAINING"].getfloat("generator_learning_rate"))
    optimizerDisc = torch.optim.Adam(disc.parameters(),lr=conf["TRAINING"].getfloat("discriminator_learning_rate"))

    genEr = EarlyStopping(gen, conf["TRAINING"].getfloat("generator_es_minIncrement"), conf["TRAINING"].getint("generator_es_epochsNoImprovement"), conf["TRAINING"]["name_model"])

    genLoss = generatorLoss()

    for i in range(conf["TRAINING"].getint("max_epochs")):

        trainOne_generator(gen, dlTrain, optimizerGen, device, genLoss)

        validate()

        print("Epoch summary bla bla bla")
