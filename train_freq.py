import torch
from BayesianSeg.models import Frequentist_UNet
from BayesianSeg.loss import *
from BayesianSeg.datasets.histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torchvision
from BayesianSeg.datasets.augs import *
from BayesianSeg.metrics.metrics import IoU
import json
import os
import argparse

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

def validate_model(model, loader, device):
    model.eval()
    total_TI = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = batch[0].to(device=device, dtype=torch.float32)
            true_masks = batch[1].to(device=device, dtype=torch.float32)
            pred_mask  = model(imgs)
            total_TI += IoU(pred=pred_mask, true=true_masks).item()
    
    return total_TI/len(loader)

def train(config):    
    EPOCHS = config["epochs"]
    BATCH_SIZE = config["train_batch_size"]
    LR = config["lr"]
    CLASSES = config["dataset"]["classes"] if len(config["dataset"]["classes"])>0 else os.listdir(os.path.join(config["dataset"]["dataset_dir"], "train", "GT"))
    NUM_CLASSES = len(CLASSES)
    TRAIN_PATH = os.path.join(config["dataset"]["dataset_dir"], "train")

    if not os.path.exists(config["cp_dir"]):
        os.makedirs(config["cp_dir"])
        print("Created logs directory at "+config["cp_dir"])
    cp_dir = config["cp_dir"]

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
        print("Created logs directory at "+config["log_dir"])

    writer=SummaryWriter(config["log_dir"])

    try:
        device = torch.device(config["device"])
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kwargs = {'num_workers': config["dataset"]["num_workers"], 'pin_memory': True} if 'cuda' in device.type  else {'num_workers': config["dataset"]["num_workers"], 'pin_memory': False}

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
        print("Created logs directory at "+config["log_dir"])
    
    train_set = histologyDataset(os.path.join(TRAIN_PATH, "images"), os.path.join(TRAIN_PATH, "GT"),
                            color=True, transform=BSCompose([Rotate(), ToTensor(), Resize(size=(256, 256))]))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    model = Frequentist_UNet(3, NUM_CLASSES, classes=CLASSES)
    model.to(device=device)
    criterion_m = CELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # gen_output_conf(model, config["model_config"])

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        mIoU = 0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{EPOCHS}') as pbar:
            for i, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                imgs = batch[0].to(device=device, dtype=torch.float32)
                true_masks = batch[1].to(device=device, dtype=torch.float32)
                pred_mask  = model(imgs)
                loss = criterion_m(pred_mask, true_masks)
                mIoU += IoU(pred=pred_mask, true=true_masks).item()
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                pbar.update()
                pbar.set_postfix(**{f"{criterion_m}": loss.item(), f"mIoU": mIoU})
        writer.add_scalar(f"training loss",
                            epoch_loss/len(train_loader),
                            epoch)
        
        writer.add_scalar(f'Training mIOU',
                            mIoU/len(train_loader),
                            epoch)
        
        writer.add_image("input image", imgs[0, :, :, :], dataformats='CHW', global_step=epoch)
        writer.add_image("predicted mask", pred_mask[0, 1, :, :], dataformats='HW', global_step=epoch)
        writer.add_image("True mask", true_masks[0, 1, :, :], dataformats='HW', global_step=epoch)

        torch.save(model.state_dict(), cp_dir + f'model_ep{str(epoch)}.pth')
        print(f"\nSaved Checkpoint. Train_loss: {epoch_loss/len(train_loader)}")

    torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
    print("Done")

if __name__=="__main__":
    args = get_args()
    config = parse_config(args.config)
    train(config)