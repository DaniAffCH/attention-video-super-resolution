from __future__ import absolute_import
import torch
from data.REDS_loader import getDataLoader
from model.discriminator import Discriminator
from model.generator import Generator

def train(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dlTrain = getDataLoader(conf)
    #TODO: add validation dl

    gen = Generator(conf).to(device)
    disc = Discriminator(conf).to(device)

    optimizerGen = torch.optim.Adam(gen.parameters(),lr=conf["TRAINING"].getfloat("generator_learning_rate"))
    optimizerDisc = torch.optim.Adam(disc.parameters(),lr=conf["TRAINING"].getfloat("discriminator_learning_rate"))
