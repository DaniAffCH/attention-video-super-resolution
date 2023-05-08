from __future__ import absolute_import

import argparse
import configparser

from debug.autotest import autotest

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--debug',
                        help='executes automated tests to check if everything is working',
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
    
if __name__ == '__main__':
    main()