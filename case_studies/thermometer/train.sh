PYTHONPATH=$PYTHONPATH:$(pwd) python train_classifier.py \
  -bs=256 -lr=0.1 -op=sgd -sm=0.9 -wd=0 -ne=200 -lrs --device="cuda" \
  --classifier="networks.differentiable_16_thermometer_encoding_cifar_resnet18" \
  --output=checkpoints/thermometer_16_mrn18_200_epochs.pth
