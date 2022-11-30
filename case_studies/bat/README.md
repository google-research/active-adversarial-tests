# Bilateral Adversarial Training: Towards Fast Training of More Robust Models Against Adversarial Attacks

Code and models for the ICCV 2019 [paper](https://arxiv.org/abs/1811.10716).

The code is based on this [repo](https://github.com/MadryLab/cifar10_challenge) from MadryLab.

We use one-step adversarial training with targeted attack and random start, which significantly improves both the speed and accuracy.

# CIFAR10
The training time is about 12 hours on one V100 GPU with 200 epochs + CIFAR10 + WRN28-10. The models can be downloaded [here](https://drive.google.com/file/d/11uUH1iV6xENARYWnzFnEZkKLb2y_8vHJ/view?usp=sharing). 

In the table below, the rows are various robust models (differ by how to choose the target label in targeted attack and the perturbation budget used in training), and the columns are two commonly used attacks for evaluation. Specifically, PGD100-2-8 means 100 iterations of PGD with step size 2 and perturbation budget 8.

Note that the two robust models, R-MOSA-LA and R-RAND-LA, are not included in the original paper, due to space limit and possible distraction from main theme. R-MOSA-LA means Most One-Step Adversarial and R-RAND-LA means random target among non-groundtruth classes. I may upload a report providing details if time allows in future but for now you can check the corresponding [code block](https://github.com/wjyouch/bilateral-adversarial-training/blob/master/train.py#L157). The takeaway is that using a proper targeted attack in one-step adversarial training makes a huge difference.

Besides, we also use PGD100-2-8 with 100 random starts (the last column of the table below). Specifically, if any of the 100 trials causes the model to fail then we say the attack succeeds. It turns out that our models perform reasonably well against this super strong and time-consuming (100 iterations + 100 random starts) attack.

|Accuracy (%)| Clean | PGD100-2-8 | CW100-2-8 | PGD1000-2-8 | CW1000-2-8|100xPGD100-2-8|
|--------|--------|--------|--------|--------|--------|----------|
|R-MC-LA (eps=8)|91.2|55.3|53.9|54.8|53.4|52.5|
|R-MC-LA (eps=4)|92.8|62.4|60.5|61.4|59.3|58.6|
|R-MOSA-LA (eps=8)|90.7|61.3|58.3|NA|NA|57.3|
|R-MOSA-LA (eps=4)|**92.8**|**71.0**|**67.9**|NA|NA|**66.9**|
|R-RAND-LA (eps=8)|89.7|59.3|56.8|NA|NA|NA|
|R-RAND-LA (eps=4)|92.8|65.7|62.9|NA|NA|NA|
|Our implementation of [Madry's method](https://arxiv.org/abs/1706.06083)|88.0|47.2|48.1|NA|NA|NA|


FYI: CIFAR100 dataset can be downloaded [here](https://drive.google.com/file/d/1Lo32gut3G9Sg4pz-ACFdsVzsxMRmCe-1/view?usp=sharing) and SVHN dataset can be downloaded [here](https://drive.google.com/file/d/1gd3-p2_9NN6k9UshmER0fyRG3UL_D2ug/view?usp=sharing).

# Environment
Python 3.6.7

TensorFlow 1.12.0
