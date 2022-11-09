import torch
from BayesianSeg.models.bayes_unet import *
from BayesianSeg.loss.losses import *
from BayesianSeg.datasets.histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from BayesianSeg.datasets.augs import *
from BayesianSeg.metrics.metrics import get_TI
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


def train(config):    
    EPOCHS = config["epochs"]
    BATCH_SIZE = config["batch_size"]
    LR = config["lr"]

    if not os.path.exists(config["cp_dir"]):
        os.makedirs(config["cp_dir"])
        print("Created logs directory at "+config["cp_dir"])
    cp_dir = config["cp_dir"]

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
        print("Created logs directory at "+config["log_dir"])

    writer=SummaryWriter(config["log_dir"])

    device = torch.device(config["device"])

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
        print("Created logs directory at "+config["log_dir"])
    
    train_set = histologyDataset(os.path.join(config["dataset"]["dataset_dir"], "images/"), os.path.join(config["dataset"]["dataset_dir"], "GT/"), color=True, 
                                transform=transforms.Compose([Rotate(), ToTensor(), Resize(size=(256, 256))]))

    n_train = len(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model = Bayesian_UNet(3, train_set.num_classes, classes=train_set.classes)
    model.to(device=device)
    criterion_m = DiceBCELoss()
    criterion_kl = BKLLoss(last_layer_only=False)
    kl_weight = 0.1
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    gen_output_conf(model, config["model_config"])

    data = next(iter(train_loader))
    writer.add_graph(model,data['image'].to(device=device, dtype=torch.float32))
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        kl = 0
        total_TI = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for i, batch in enumerate(train_loader, 1):
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                pred_mask  = model(imgs)
                loss_m = criterion_m(pred_mask, true_masks)
                loss_kl = criterion_kl(model)
                loss = loss_m + kl_weight*loss_kl
                total_TI += (1 - get_TI(pred=pred_mask, true=true_masks, alpha=1, beta=1, smooth=0, gamma=1).item())
                epoch_loss += loss.item()

                pbar.set_postfix(**{'TI (batch)': loss_m.item(), 'KL Div (batch)': loss_kl.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                kl += loss_kl.item()
            
        writer.add_scalar('training loss',
                            epoch_loss/len(train_loader),
                            epoch)
        
        writer.add_scalar('KL Divergence',
                            kl/len(train_loader),
                            epoch)
        
        writer.add_scalar('Training mIOU',
                            total_TI/len(train_loader),
                            epoch)
        
        writer.add_image("input image", imgs, dataformats='NCHW', global_step=epoch)
        writer.add_image("predicted mask", torch.unsqueeze(pred_mask[:, 1, :, :], 1), dataformats='NCHW', global_step=epoch)
        writer.add_image("True mask", torch.unsqueeze(true_masks[:, 1, :, :], 1), dataformats='NCHW', global_step=epoch)

        # scheduler.step(val_loss/n_val)
        torch.save(model.state_dict(), cp_dir + f'model_ep{str(epoch)}.pth')
        print(f"\nSaved Checkpoint. Train_loss: {epoch_loss/len(train_loader)}")

    torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
    print("Done")

if __name__=="__main__":
    args = get_args()
    config = parse_config(args.config)
    train(config)