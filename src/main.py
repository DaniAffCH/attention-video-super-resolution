from __future__ import absolute_import

import argparse
import configparser

from debug.autotest import autotest
from train.trainer import train
from evaluation.evaluate import evaluate
from inference.sr_inference import inference


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--debug',
                        help='executes automated tests to check if everything is working',
                        action='store_true')
    
    parser.add_argument('--train',
                        help='trains the network',
                        action='store_true')
    
    parser.add_argument('--evaluate',
                        help='trains the network')
    
    parser.add_argument('--inference',
                        help='perform inference of the current model',
                        action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read('config/conf.ini')

    if config['DEFAULT'].getboolean("debugging"):
        print("settings loaded")

    if(args.debug):
        autotest(config)
    
    elif(args.train):
        train(config)
    
    elif(args.evaluate):
        evaluate(config, args.evaluate)
    
    elif(args.inference):
        inference(config)


if __name__ == '__main__':
    main()