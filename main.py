import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    train(args)

def load_json(setting_path):
    with open(setting_path, encoding='utf-8') as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='TerraSAP training and evaluation')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to an experiment JSON config.')
    return parser

if __name__ == '__main__':
    main()
