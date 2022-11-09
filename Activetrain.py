import torch
from BayesianSeg.loss.losses import *
from BayesianSeg.datasets.histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from BayesianSeg.models.bayes_unet import *
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from BayesianSeg.datasets.augs import *
import os
import math
import random
import matplotlib.pyplot as plt
from BayesianSeg.metrics.metrics import IoU, get_entropies
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

def train(config):
    LABEL_PERCENT = config["label_percent"]
    EPOCHS = config["epochs"]
    BATCH_SIZE = config["batch_size"]
    LR = config["lr"]

    IMAGES = os.listdir(os.path.join(config["dataset"]["dataset_dir"], "images"))
    CLASSES = os.listdir(os.path.join(config["dataset"]["dataset_dir"], "GT")) if config["dataset"]["classes"]==0 else config["dataset"]["classes"]==0

    random.shuffle(IMAGES)

    n_train = int(len(IMAGES))
    n_label = int(n_train * LABEL_PERCENT)
    n_unlabel = len(IMAGES) - n_label

    train_set = IMAGES[:n_train]
    label_set = train_set[:n_label]
    unlabel_set = train_set[n_label:]

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

    active_batch_size = int(len(IMAGES) * LABEL_PERCENT)

    n = 0
    while(len(unlabel_set)>0):
        label_dataset = histologyDataset("./histology_dataset/30/train/images/", "./histology_dataset/30/train/GT/", color=True, 
                            transform=transforms.Compose([Brightness(100), Rotate(), ToTensor(), Resize(size=(256, 256))]),
                            classes=['AS', 'Cartilage', 'RS', 'SM'], im_names=label_set)
        model = train_model(label_dataset, n_label, n, device, LR, BATCH_SIZE, EPOCHS, writer)
        torch.save(model.state_dict(), cp_dir + str(len(label_set)/len(IMAGES)*100) + f'.pth')
        unlabel_dataset = histologyDataset("./histology_dataset/30/train/images/", "./histology_dataset/30/train/GT/", color=True, 
                            transform=transforms.Compose([Brightness(100), Rotate(), ToTensor(), Resize(size=(256, 256))]),
                            classes=['AS', 'Cartilage', 'RS', 'SM'], im_names=unlabel_set)
        torch.cuda.empty_cache()
        new_ims = sample_images(model, unlabel_dataset, device, k=math.floor(len(IMAGES) * LABEL_PERCENT))
        for i in new_ims:
            label_set.append(i[0])
            unlabel_set.remove(i[0])

        n+=1


    torch.save(model.state_dict(), cp_dir + f'Final_model.pth')

def train_model(label_set, n_label, active_epoch, device, LR, BATCH_SIZE, EPOCHS, writer):
    model = Bayesian_UNet(3, label_set.num_classes, classes=label_set.classes)
    model.to(device=device)
    criterion_m = DiceBCELoss()
    criterion_kl = BKLLoss(last_layer_only=False)
    kl_weight = 0.1
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(label_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model.train()
    TIs = []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        kl = 0
        total_TI = 0

        
        with tqdm(total=n_label, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for i, batch in enumerate(train_loader, 1):
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                pred_mask  = model(imgs)
                loss_m = criterion_m(pred_mask, true_masks)
                loss_kl = criterion_kl(model)
                loss = loss_m + kl_weight*loss_kl
                total_TI += IoU(pred=pred_mask, true=true_masks).item()
                epoch_loss += loss.item()
                kl += loss_kl

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
        TIs.append(total_TI/len(train_loader))
    
    fig, axs = plt.subplots(1, 1)
    axs.plot(list(range(EPOCHS)), TIs)
    writer.add_figure("Active epoch", fig, global_step=active_epoch)

    return model

def sample_images(model, unlabel_dataset, device, k=10, num_iter=30):
    new_ims = []
    model.eval()

    for i in tqdm(range(len(unlabel_dataset))):
        im_name = unlabel_dataset.im_names[i]
        preds = []
        for _ in range(num_iter):
            im = unlabel_dataset[i]['image']
            out = torch.squeeze(model(torch.unsqueeze(im, dim=0).to(device=device, dtype=torch.float32)))
            preds.append(torch.nn.functional.softmax(out, dim=0).detach().cpu())
        E = get_entropies(preds, mode="RAND")
        new_ims.append((im_name, E.item()))
    
    new_ims.sort(key = lambda x: x[1], reverse=True)

    return new_ims[:k]    


if __name__=="__main__":
    args = get_args()
    config = parse_config(args.config)
    train(config)

