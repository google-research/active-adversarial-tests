# ML-LOO: Detecting Adversarial Examples with Feature Attribution

Code for ML-LOO on a minimal example.

## Dependencies
The code runs with Python 2.7 and requires Tensorflow 1.11.0, and Keras 2.2.4. Please `pip install` the following packages:
- `numpy`
- `scipy`
- `tensorflow` 
- `keras`
- `scikit-learn`

## Preparation stage
Generate the adversarial examples by C&W attack, to be used in the training stage of ML-LOO

```shell
###############################################
python generate_attack.py --dataset_name cifar10 --model_name resnet
###############################################
```

Extract ML-LOO features (i.e., IQR of LOO attribution maps of different layers of a neural network), and then split the data set containing original and adversarial examples into a training set and a test set. 

```shell
###############################################
python features_extract.py --dataset_name cifar10 --model_name resnet --attack cw --det ml_loo
###############################################
```

## Train ML-LOO and evaluate its performance
Train ML-LOO and evaluate it on the test set.
```shell
python train_and_evaluate.py --dataset_name cifar10 --model_name resnet --data_sample x_val200
```
The generated AUC plot can be found in cifar10resnet/figs/.














 
