import json

config = {
    "epochs": 400,
    "lr": 0.01,
    "label_percent": 0.05,
    "cp_dir": "checkpoints/",
    "device": "cuda",
    "model_config": "model.json",
    "model_path": "model.pth",
    "kl_weight": 0.01,
    "log_dir": "logs/",
    "loss": {
        "name": "CELoss",
        "kwargs": {
            "reduction": "mean"
        }
    },
    "div": {
        "name": "BKLLoss",
        "kwargs": {
            "reduction": "mean",
            "last_layer_only": "False"
        }
    },
    "optim": "Adam",
    "alpha": 1,
    "sampling_method": "smart",
    "eval_metric": {
        "name": "IoU",
        "kwargs": {
            "smooth": 1e-6, 
            "cutoff": 0.5
        }
    },
    "model": {
        "prior_mu": 0,
        "prior_sigma": 0.1,
        "bilinear": False,
        "path": "checkpoints/model_ep0.pth"
    },
    "dataset": {
        "dataset_dir": "histology_dataset/30/",
        "num_workers": 8,
        "train_batch_size": 32,
        "test_batch_size": 32,
        "classes": []
    }
}

with open("./config.json", "w") as f:
    json.dump(config, f)

print("Saved default config at ./config.json")