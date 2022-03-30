### Make predictions
To make predictions, run predict.py with the arguments as directed in the help manual. You can see the help manual by running:
```bash
python predict,py --help
```

For example:
```bash
python predict.py -m ./Final_model.pth -i test_images/ -d cuda
```

Note: You will need to add your own pretrained model. Training script will be modified for the repository shortly.