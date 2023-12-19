import torch
from BayesianSeg.loss import Loss
from BayesianSeg.metrics import Metric, Sampler, Uncertainty
from evaluate import evaluate
from BayesianSeg.datasets import *
from BayesianSeg.misc import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from BayesianSeg.models.bayes_unet import *
from BayesianSeg.models.freq_unet import *
from BayesianSeg.modules import CRFLayer
from torch import optim
from tqdm import tqdm
from BayesianSeg.datasets.augs import *
import os
import math
import random
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
    MODE = config["mode"]
    LABEL_PERCENT = config["label_percent"]
    EPOCHS = config["epochs"]
    TRAIN_BATCH_SIZE = config["dataset"]["train_batch_size"]
    TEST_BATCH_SIZE = config["dataset"]["test_batch_size"]
    LR = config["lr"]
    CRF = config["model"]["crf"]
    CRF_LR = config["crf_lr"]
    CRF_ITER = config["model"]["crf_iter_train"]
    ALPHA = config["alpha"]
    DS_NAME = config["dataset"]["name"]
    ROOT_DIR = config["dataset"]["root_dir"]
    TRAIN_PATH = os.path.join(config["dataset"]["root_dir"], "train")
    SAMPLING_METHOD = config["sampling_method"]
    KL_WEIGHT = config["kl_weight"]
    PRIOR_MU = config["model"]["prior_mu"]
    PRIOR_SIGMA = config["model"]["prior_sigma"]
    WEIGHT_DECAY = config["weight_decay"]
    IMAGES = os.listdir(os.path.join(TRAIN_PATH, "images"))
    BILINEAR = config["model"]["bilinear"]
    
    LOSS = config["loss"]["name"]
    LOSS_KWARGS = config["loss"]["kwargs"]
    LOSS_KL = config["div"]["name"]
    LOSS_KL_KWARGS = config["div"]["kwargs"]
    
    EVAL_METRIC = config["eval_metric"]["name"]
    EVAL_KWARGS = config["eval_metric"]["kwargs"]

    SAMPLER = Sampler(SAMPLING_METHOD)

    LOG_DIR = os.path.join(config["log_dir"], DS_NAME, MODE, SAMPLING_METHOD)
    CHKPT_DIR = os.path.join(config["chkpt_dir"], DS_NAME, MODE, SAMPLING_METHOD)

    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)

    random.shuffle(IMAGES)

    n_train = int(len(IMAGES))
    n_label = int(n_train * LABEL_PERCENT*2)

    train_set = IMAGES
    label_set = train_set[:n_label]
    unlabel_set = train_set[n_label:]

    val_set = DS(name=DS_NAME, root_dir=ROOT_DIR, split="val", transforms=[Resize(size=(256, 256))])
    
    val_loader = DataLoader(val_set, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"Created logs directory at {LOG_DIR}")

    if not os.path.exists(CHKPT_DIR):
        os.makedirs(CHKPT_DIR)
        print(f"Created logs directory at {CHKPT_DIR}")

    writer=SummaryWriter(LOG_DIR)

    try:
        device = torch.device(config["device"])
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Bayesian_UNet(PRIOR_MU, PRIOR_SIGMA, 3, len(val_set.classes), classes=val_set.classes, bilinear=BILINEAR) if MODE=="bayes" else Frequentist_UNet(3, len(val_set.classes), classes=val_set.classes, bilinear=BILINEAR)
    crf_out_layer = CRFLayer(CRF_ITER, 3, len(val_set.classes)) if CRF else None
    MODEL_TYPE = str(model)
    model.to(device)
    if CRF:
        crf_out_layer.to(device)
    print(MODEL_TYPE)
    torch.save(model.state_dict(), os.path.join(CHKPT_DIR, "model_init.pth"))

    LOG_STR = f"Dataset_{DS_NAME}_Model_{MODEL_TYPE}_alpha_{ALPHA}_{SAMPLING_METHOD}_kl_weight_{KL_WEIGHT}_priorm_{PRIOR_MU}_priors_{PRIOR_SIGMA}"

    mIoUs = []

    n = 1
    while(len(unlabel_set)>0):
        print()
        print(f"Active iter: {n}, training_sample_size: {(len(label_set)/(len(label_set)+len(unlabel_set)))*100:.2f}%")
        print("Resetting model parameters")
        model = Bayesian_UNet(PRIOR_MU, PRIOR_SIGMA, 3, len(val_set.classes), classes=val_set.classes, bilinear=BILINEAR) if MODE=="bayes" else Frequentist_UNet(3, len(val_set.classes), classes=val_set.classes, bilinear=BILINEAR)
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(CHKPT_DIR, "model_init.pth"), map_location=device))
        label_dataset = DS(name=DS_NAME, root_dir=ROOT_DIR, split="train",
                            color=True, transforms=[Resize(size=(256, 256)), Rotate(180, fill=1), gauss_noise(5)],
                            im_names=label_set)
        print(f"Training model")
        with torch.autograd.set_detect_anomaly(True):
            model = train_model(model, crf_out_layer, label_dataset, device, LR, CRF_LR, TRAIN_BATCH_SIZE, EPOCHS, KL_WEIGHT, LOSS, LOSS_KL, LOSS_KWARGS, LOSS_KL_KWARGS, EVAL_METRIC, EVAL_KWARGS, WEIGHT_DECAY, writer, n, log_str=LOG_STR)
        print("Validating model")
        model.freeze()
        mIoUs.append(evaluate(model, val_loader, crf_out_layer=crf_out_layer, device=device, metric=criterion_acc, writer=writer, step=n, log_str=LOG_STR))
        model.unfreeze()

        unlabel_dataset = DS(name=DS_NAME, root_dir=ROOT_DIR, split="train",
                            color=True, transforms=[Resize(size=(256, 256))],
                            im_names=unlabel_set)
        torch.cuda.empty_cache()
        print("Sampling images")
        new_ims = sample_images(model, unlabel_dataset, device, alpha=ALPHA, sampler=SAMPLER, num_runs=20, k=math.floor(len(IMAGES) * LABEL_PERCENT))

        eu_maps = []
        au_maps = []
        sampled_ims = []
        u_value = 0
        with torch.no_grad():
            for i in new_ims:
                sampled_ims.append(i[2].cpu())
                eu_maps.append(grayscale_to_heatmap(Uncertainty.epistemic(model, i[2], 20).cpu()))
                au_maps.append(grayscale_to_heatmap(Uncertainty.aleatoric(model, i[2], 20).cpu()))
                u_value += i[1]
                label_set.append(i[0])
                unlabel_set.remove(i[0])
        
        writer.add_images(f"{LOG_STR}_Sampled Image", np.vstack(sampled_ims), n, dataformats='NCHW')
        writer.add_images(f"{LOG_STR}_PEU", np.stack(eu_maps, axis=0), n, dataformats='NHWC')
        writer.add_images(f"{LOG_STR}_PAU", np.stack(au_maps, axis=0), n, dataformats='NHWC')

        writer.add_scalar(f"{LOG_STR} Uncertainty", u_value, n)
        writer.add_scalar(f"{LOG_STR} mIoU",
                            max(mIoUs),
                            (len(label_set)/(len(label_set)+len(unlabel_set)))*100)
        writer.flush()
        
        n+=1
    
    print("Saving final model")
    torch.save(model.state_dict(), os.path.join(CHKPT_DIR, "model_init.pth"))
    if CRF:
        print("Saving output CRF layer")
        torch.save(crf_out_layer.state_dict(), os.path.join(CHKPT_DIR, "crf_final.pth"))

def train_model(model, crf_layer, label_set, device, LR, CRF_LR, BATCH_SIZE, EPOCHS, KL_weight, LOSS, LOSS_KL, LOSS_KWARGS, LOSS_KL_KWARGS, EVAL_METRIC, EVAL_KWARGS, WEIGHT_DECAY, writer=None, step=0, log_str=""):
    criterion_m = Loss(LOSS, **LOSS_KWARGS)
    criterion_kl = Loss(LOSS_KL, **LOSS_KL_KWARGS)
    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)
    kl_weight = KL_weight
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(label_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model.train()
    model.unfreeze()
    TIs = []
    for epoch in tqdm(range(EPOCHS)):
        epoch_loss = 0
        kl = 0
        total_TI = 0

        for i, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            imgs = batch[0].to(device=device, dtype=torch.float32)
            true_masks = batch[1].to(device=device, dtype=torch.float32)
            pred_logits  = model(imgs)
            loss_m = criterion_m(pred_logits, true_masks)
            if kl_weight>0:
                loss_kl = criterion_kl(model)
            else:
                loss_kl = 0
            loss = loss_m + kl_weight*loss_kl
            total_TI += criterion_acc(pred=normalize(pred_logits), true=true_masks).item()
            epoch_loss += loss.item()
            kl += loss_kl

            loss.backward()
            optimizer.step()

        TIs.append(total_TI/len(train_loader))
    
    if isinstance(crf_layer, nn.Module):
        model.freeze()
        optimizer = optim.Adam(crf_layer.parameters(), lr=CRF_LR, weight_decay=WEIGHT_DECAY)
        for epoch in tqdm(range(EPOCHS)):
            epoch_loss = 0
            total_TI = 0

            for i, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                imgs = batch[0].to(device=device, dtype=torch.float32)
                true_masks = batch[1].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    pred_logits = model(imgs)
                pred_logits_crf  = crf_layer(pred_logits, imgs)
                loss_m = criterion_m(pred_logits_crf, true_masks)
                loss = loss_m
                total_TI += criterion_acc(pred=normalize(pred_logits_crf), true=true_masks).item()
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

            TIs.append(total_TI/len(train_loader))

    if isinstance(writer, SummaryWriter):
        writer.add_images(f"{log_str}_Training Input", imgs, step)
        writer.add_images(f"{log_str}_Training GT", true_masks, step)
        writer.add_images(f"{log_str}_Training Prediction", normalize(pred_logits), step)
        if isinstance(crf_layer, nn.Module):
            writer.add_images(f"{log_str}_Training Prediction CRF", normalize(pred_logits_crf), step)
    
    return model


@torch.no_grad()
def sample_images(model, unlabel_dataset, device, alpha, sampler, num_runs=20, k=8):
    new_ims = []
    for i in tqdm(range(len(unlabel_dataset))):
        im_name = unlabel_dataset.im_names[i]
        input_im = torch.unsqueeze(unlabel_dataset[i][0], dim=0).to(device=device, dtype=torch.float32)
        E = sampler(model, input_im, num_runs, alpha)
        new_ims.append((im_name, E, input_im.detach()))
    
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

