import torch
from BayesianSeg.loss import Loss
from BayesianSeg.metrics import Metric, Sampler
from evaluate import evaluate
from BayesianSeg.datasets import *
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
    DS_NAME = config["dataset"]["name"]
    ROOT_DIR = config["dataset"]["root_dir"]
    TRAIN_PATH = os.path.join(config["dataset"]["root_dir"], "train")
    SAMPLING_METHOD = config["sampling_method"]
    KL_WEIGHT = config["kl_weight"]
    PRIOR_MU = config["model"]["prior_mu"]
    PRIOR_SIGMA = config["model"]["prior_sigma"]
    OUT_LAYER = config["model"]["out_layer"]

    IMAGES = os.listdir(os.path.join(TRAIN_PATH, "images"))
    BILINEAR = config["model"]["bilinear"]
    
    LOSS = config["loss"]["name"]
    LOSS_KWARGS = config["loss"]["kwargs"]
    LOSS_KL = config["div"]["name"]
    LOSS_KL_KWARGS = config["div"]["kwargs"]
    
    EVAL_METRIC = config["eval_metric"]["name"]
    EVAL_KWARGS = config["eval_metric"]["kwargs"]

    SAMPLER = Sampler(SAMPLING_METHOD)

    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)

    random.shuffle(IMAGES)

    n_train = int(len(IMAGES))
    n_label = int(n_train * LABEL_PERCENT*2)

    train_set = IMAGES
    label_set = train_set[:n_label]
    unlabel_set = train_set[n_label:]

    val_set = DS(name=DS_NAME, root_dir=ROOT_DIR, split="val", transforms=[Resize(size=(256, 256))])
    
    val_loader = DataLoader(val_set, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
        print("Created logs directory at "+config["log_dir"])

    writer=SummaryWriter(config["log_dir"])

    try:
        device = torch.device(config["device"])
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Bayesian_UNet(PRIOR_MU, PRIOR_SIGMA, 3, len(val_set.classes), classes=val_set.classes, bilinear=BILINEAR, out_layer=OUT_LAYER, crf_out=False)
    torch.save(model.state_dict(), "checkpoints/model_init.pth")
    model.to(device)

    mIoUs = []

    n = 1
    while(len(unlabel_set)>0):
        print(f"Active iter: {n}, training_sample_size: {(len(label_set)/(len(label_set)+len(unlabel_set)))*100:.2f}%")
        label_dataset = DS(name=DS_NAME, root_dir=ROOT_DIR, split="train",
                            color=True, transforms=[Resize(size=(256, 256)), Rotate(180, fill=1)],
                            im_names=label_set)
        print(f"Training model")
        model = train_model(PRIOR_MU, PRIOR_SIGMA, label_dataset, device, LR, TRAIN_BATCH_SIZE, EPOCHS, KL_WEIGHT, LOSS, LOSS_KL, LOSS_KWARGS, LOSS_KL_KWARGS, EVAL_METRIC, EVAL_KWARGS, BILINEAR, OUT_LAYER, writer, n)
        print("Validating model")
        mIoUs.append(evaluate(model, val_loader, device=device, metric=criterion_acc, writer=writer, step=n))

        unlabel_dataset = DS(name=DS_NAME, root_dir=ROOT_DIR, split="train",
                            color=True, transforms=[Resize(size=(256, 256)), Rotate(180, fill=1)],
                            im_names=unlabel_set)
        torch.cuda.empty_cache()
        print("Sampling images")
        new_ims = sample_images(model, unlabel_dataset, device, alpha=ALPHA, sampler=SAMPLER, k=math.floor(len(IMAGES) * LABEL_PERCENT))
        for i in new_ims:
            label_set.append(i[0])
            unlabel_set.remove(i[0])

        writer.add_scalar(f"mIoU_alpha_{ALPHA}_{SAMPLING_METHOD}_kl_weight_{KL_WEIGHT}_priorm_{PRIOR_MU}_priors_{PRIOR_SIGMA}",
                            max(mIoUs),
                            (len(label_set)/(len(label_set)+len(unlabel_set)))*100)
        
        n+=1



def train_model(PRIOR_MU, PRIOR_SIGMA, label_set, device, LR, BATCH_SIZE, EPOCHS, KL_weight, LOSS, LOSS_KL, LOSS_KWARGS, LOSS_KL_KWARGS, EVAL_METRIC, EVAL_KWARGS, BILINEAR, OUT_LAYER, writer=None, step=0):
    model = Bayesian_UNet(PRIOR_MU, PRIOR_SIGMA, 3, len(label_set.classes), classes=label_set.classes, bilinear=BILINEAR, out_layer=OUT_LAYER, crf_out=False)
    model.to(device=device)
    model.load_state_dict(torch.load("checkpoints/model_init.pth"))
    criterion_m = Loss(LOSS, **LOSS_KWARGS)
    criterion_kl = Loss(LOSS_KL, **LOSS_KL_KWARGS)
    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)
    kl_weight = KL_weight
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    train_loader = DataLoader(label_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model.train_backbone()
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

    if isinstance(writer, SummaryWriter):
        writer.add_images("Training Input", imgs, step)
        writer.add_images("Training GT", true_masks, step)
        writer.add_images("Training Prediction", pred_mask, step)
    
    return model


@torch.no_grad()
def sample_images(model, unlabel_dataset, device, alpha, sampler, k=8):
    new_ims = []
    model.eval()

    for i in tqdm(range(len(unlabel_dataset))):
        im_name = unlabel_dataset.im_names[i]
        preds = model(unlabel_dataset[i][0].repeat(k, 1, 1, 1).to(device=device, dtype=torch.float32)).detach()
        E = sampler(preds, alpha=alpha)
        # print(E)
        new_ims.append((im_name, E))
    
    
    new_ims.sort(key = lambda x: x[1], reverse=True)

    return new_ims[:k]    


if __name__=="__main__":
    random.seed(10)
    torch.manual_seed(0)
    args = get_args()
    config = parse_config(args.config)
    print(config["sampling_method"])
    train(config)
    print("Done")

