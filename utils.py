import argparse
import json


def get_args():
    parser = argparse.ArgumentParser(description='Segments histology images in input directory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config-file', type=str, help='Config File', dest='config', default="./config.json")
    return parser.parse_args()

def parse_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(config_path+" not found! Try running gen_config to generate a default config file")

def gen_output_conf(model, path):
    with open(path, 'w') as f:
        data  = {"classes": model.classes}
        json.dump(data, f)
    print("Saved classes to "+path)