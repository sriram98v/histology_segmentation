import torch
from BayesianSeg.loss import Loss
from BayesianSeg.metrics import Metric, Sampler
from evaluate import evaluate
from BayesianSeg.datasets.histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from BayesianSeg.models.bayes_unet import *
from torch import optim
from tqdm import tqdm
from BayesianSeg.datasets.augs import *
import os
import math
import random
import matplotlib.pyplot as plt
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
    TRAIN_BATCH_SIZE = config["dataset"]["train_batch_size"]
    TEST_BATCH_SIZE = config["dataset"]["test_batch_size"]
    LR = config["lr"]
    ALPHA = config["alpha"]
    TRAIN_PATH = os.path.join(config["dataset"]["dataset_dir"], "train")
    VAL_PATH = os.path.join(config["dataset"]["dataset_dir"], "test")
    SAMPLING_METHOD = config["sampling_method"]
    KL_WEIGHT = config["kl_weight"]
    PRIOR_MU = config["model"]["prior_mu"]
    PRIOR_SIGMA = config["model"]["prior_sigma"]

    IMAGES = os.listdir(os.path.join(TRAIN_PATH, "images"))
    CLASSES = os.listdir(os.path.join(TRAIN_PATH, "GT")) if config["dataset"]["classes"]==0 else config["dataset"]["classes"]==0
    BILINEAR = config["model"]["bilinear"]
    
    LOSS = config["loss"]["name"]
    LOSS_KWARGS = config["loss"]["kwargs"]
    LOSS_KL = config["div"]["name"]
    LOSS_KL_KWARGS = config["div"]["kwargs"]
    
    EVAL_METRIC = config["eval_metric"]["name"]
    EVAL_KWARGS = config["eval_metric"]["kwargs"]

    SAMPLER = Sampler(SAMPLING_METHOD)

    TRAIN_PATH = os.path.join(config["dataset"]["dataset_dir"], "train")

    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)

    random.shuffle(IMAGES)

    n_train = int(len(IMAGES))
    n_label = int(n_train * LABEL_PERCENT)
    n_unlabel = len(IMAGES) - n_label

    train_set = IMAGES[:n_train]
    label_set = train_set[:n_label]
    unlabel_set = train_set[n_label:]

    val_set = histologyDataset(os.path.join(VAL_PATH, "images"), os.path.join(VAL_PATH, "GT"),
                            color=True, transform=BSCompose([Brightness(100), Rotate(), ToTensor(), Resize(size=(256, 256))]))
    val_loader = DataLoader(val_set, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

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


    n = 0
    while(len(unlabel_set)>0):
        print(f"Active iter: {n}")
        label_dataset = histologyDataset(os.path.join(TRAIN_PATH, "images"), os.path.join(TRAIN_PATH, "GT"),
                            color=True, transform=BSCompose([Brightness(100), Rotate(), ToTensor(), Resize(size=(256, 256))]),
                            im_names=label_set)
        print("Training model")
        model = train_model(PRIOR_MU, PRIOR_SIGMA, label_dataset, device, LR, TRAIN_BATCH_SIZE, EPOCHS, KL_WEIGHT, LOSS, LOSS_KL, LOSS_KWARGS, LOSS_KL_KWARGS, EVAL_METRIC, EVAL_KWARGS)
        print("Validating model")
        mIoU = evaluate(model, val_loader, device=device, metric=criterion_acc)

        unlabel_dataset = histologyDataset(os.path.join(TRAIN_PATH, "images"), os.path.join(TRAIN_PATH, "GT"),
                            color=True, transform=BSCompose([Brightness(100), Rotate(), ToTensor(), Resize(size=(256, 256))]),
                            im_names=unlabel_set)
        torch.cuda.empty_cache()
        print("Sampling images")
        new_ims = sample_images(model, unlabel_dataset, device, alpha=ALPHA, sampler=SAMPLER, k=math.floor(len(IMAGES) * LABEL_PERCENT))
        for i in new_ims:
            label_set.append(i[0])
            unlabel_set.remove(i[0])

        writer.add_scalar(f"mIoU_alpha_{ALPHA}_{SAMPLING_METHOD}_kl_weight_{KL_WEIGHT}_priorm_{PRIOR_MU}_priors_{PRIOR_SIGMA}",
                            mIoU,
                            n)

        n+=1



def train_model(PRIOR_MU, PRIOR_SIGMA, label_set, device, LR, BATCH_SIZE, EPOCHS, KL_weight, LOSS, LOSS_KL, LOSS_KWARGS, LOSS_KL_KWARGS, EVAL_METRIC, EVAL_KWARGS):
    model = Bayesian_UNet(PRIOR_MU, PRIOR_SIGMA, 3, label_set.num_classes, classes=label_set.classes)
    model.to(device=device)
    criterion_m = Loss(LOSS, device, **LOSS_KWARGS)
    criterion_kl = Loss(LOSS_KL, device, **LOSS_KL_KWARGS)
    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)
    kl_weight = KL_weight
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(label_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model.train()
    TIs = []
    for epoch in tqdm(range(EPOCHS)):
        epoch_loss = 0
        kl = 0
        total_TI = 0

        for i, batch in enumerate(train_loader, 1):
            imgs = batch[0].to(device=device, dtype=torch.float32)
            true_masks = batch[1].to(device=device, dtype=torch.float32)
            pred_mask  = model(imgs)
            loss_m = criterion_m(pred_mask, true_masks)
            loss_kl = criterion_kl(model)
            loss = loss_m + kl_weight*loss_kl
            total_TI += criterion_acc(pred=pred_mask, true=true_masks).item()
            epoch_loss += loss.item()
            kl += loss_kl


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        TIs.append(total_TI/len(train_loader))
    
    return model

def sample_images(model, unlabel_dataset, device, alpha, sampler, k=10, num_iter=10):
    new_ims = []
    model.eval()

    for i in tqdm(range(len(unlabel_dataset))):
        im_name = unlabel_dataset.im_names[i]
        preds = []
        for _ in range(num_iter):
            im = unlabel_dataset[i][0]
            out = torch.squeeze(model(torch.unsqueeze(im, dim=0).to(device=device, dtype=torch.float32)))
            preds.append(torch.nn.functional.softmax(out, dim=0).detach())
        E = sampler(torch.flatten(torch.stack(preds), start_dim=-3, end_dim=-1), alpha=alpha)
        new_ims.append((im_name, E))
    
    new_ims.sort(key = lambda x: x[1], reverse=True)

    return new_ims[:k]    


if __name__=="__main__":
    args = get_args()
    config = parse_config(args.config)
    train(config)
    print("Done")

