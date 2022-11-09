import json

config = {
    "epochs": 100,
    "batch_size": 8,
    "lr": 0.01,
    "cp_dir": "./checkpoints/",
    "device": "cpu",
    "model_config":"model.json",
    "model_path": "./model.pth",
    "kl_weight": 0.1,
    "log_dir": "./logs/",
    "loss": "dicebce",
    "KL_div": "BKL",
    "optim": "Adam",
    "dataset": {
        "dataset_dir": "./data/",
        "classes": [],
        "tranforms": ["Rotate", "Resize"]
        }
}

with open("./config.json", "w") as f:
    json.dump(config, f)

print("Saved default config at ./config.json")