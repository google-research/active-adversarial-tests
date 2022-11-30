# EMPIR: Ensembles of Mixed Precision Deep Networks for Increased Robustness against Adversarial Attacks
[![Build Status](https://travis-ci.org/tensorflow/cleverhans.svg?branch=master)](https://travis-ci.org/tensorflow/cleverhans)

This repository contains the source code for the paper EMPIR: Ensembles of Mixed Precision Deep Networks for Increased Robustness against Adversarial Attacks ([Accepted at ICLR 2020](https://openreview.net/forum?id=HJem3yHKwH))

It is based on [CleverHans](https://github.com/tensorflow/cleverhans) 1.0.0, a Python library to
benchmark machine learning systems' vulnerability to
[adversarial examples](http://karpathy.github.io/2015/03/30/breaking-convnets/).
You can learn more about such vulnerabilities on the accompanying [blog](http://cleverhans.io).

## Setting up
+ Install [TensorFlow](https://www.tensorflow.org/) 
+ Install [Keras](https://keras.io/)
+ Git clone this repository
+ For ImageNet results, download ImageNet dataset and convert the data into `TFRecords` using [this](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) script. 

We tested this setup using tensorflow-gpu 1.10, keras 2.2.4, python 3.5, CUDA 9.2 and Ubuntu 18.04 on a single RTX 2080 Ti GPU. Tensorflow was installed using [anaconda](https://www.anaconda.com/). 

## Example commands
+ `python examples/mnist_attack.py --wbits=$model1_weight_prec --abits=$model1_activation_prec --wbits2=$model2_weight_prec --abits2=$model2_activation_prec --ensembleThree --model_path1=/path/to/model1/ckpt --model_path2=/path/to/model2/ckpt --model_path3=/path/to/model3/ckpt` - White-Box CW attack on MNISTconv EMPIR model
+ `python examples/mnist_attack.py --model_path=/path/to/baseline/model/ckpt` - White-Box CW attack on MNISTconv baseline model
+ `python examples/cifar10_attack.py --abits=$model1_activation_prec --wbits=$model1_weight_prec --abits2=$model2_activation_prec --wbits2=$model2_weight_prec --model_path1=/path/to/model1/ckpt --model_path2=/path/to/model2/ckpt --model_path3=/path/to/model3/ckpt --ensembleThree` - White-Box CW attack on CIFARconv EMPIR model
+ `python examples/cifar10_attack.py --model_path=/path/to/baseline/model/ckpt` - White-Box CW attack on CIFARconv baseline model
+ `python examples/alexnet_attack.py --batch_size=100 --imagenet_path=/path/to/imagenet/tf_records --ensembleThree --abits=$model1_activation_prec --wbits=$model1_weight_prec --abits2=$model2_activation_prec --wbits2=$model2_weight_prec --model_path1=/path/to/model1/ckpt --model_path2=/path/to/model2/ckpt --model_path3=/path/to/model3/ckpt` - White-Box CW attack on AlexNet EMPIR model
+ `python examples/alexnet_attack.py --batch_size=100 --imagenet_path=/path/to/imagenet/tf_records --model_path=/path/to/baseline/model/ckpt` - White-Box CW attack on AlexNet baseline model

## Results
+ EMPIR models
<table>
    <tr align="center">
        <th rowspan="2">Dataset</th>
        <th colspan=3>Precisions</th>
        <th rowspan=2>Unperturbed Accuracy (%)</th>
        <th colspan=5>Adversarial Accuracy (%)</th>
    </tr>
    <tr align="center">
        <th>Model 1</th>
        <th>Model 2</th>
        <th>Model 3</th>
        <th>CW</th>
        <th>FGSM</th>
        <th>BIM</th>
        <th>PGD</th>
        <th>Average</th>
    </tr>
    <tr align="center">
       <td>MNIST</td>
       <td> abits=4, wbits=2 <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EslYS5ShH0JBnoBLbWy8NAsBkGZebG9Z08-SiIPguPlcxA?e=5Dtb1k">Download</a> </td>
       <td> abits=4, wbits=2 <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EntWYPrfnsFMqpTdisN6egABpg8oqQjIEByA2IOXpsAOsw?e=bGV3yw">Download</a> </td>
       <td> Full-precision (32 bits) <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EumogLgncVBCm932G6fUBPgBcipIcex0GhmG0SLIZdFT2g?e=K5MCh8">Download</a> </td>
       <td> 98.89 </td>
       <td> 86.73 </td>
       <td> 67.06 </td>
       <td> 18.61 </td>
       <td> 17.51 </td>
       <td> 47.48 </td>
    </tr>
    <tr align="center">
       <td>CIFAR-10</td>
       <td> abits=2, wbits=4 <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EtnxDGIo-iBOgumA6qaAY7IBm7vCT7QVIFab5q2ZUTo4ww?e=RZd1E7">Download</a></td>
       <td> abits=2, wbits=2 <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EmU1sNJnvh5KjFNhRzU5phoBj7jVnszuE7XTOXXcVNES0g?e=E6PbCa">Download</a> </td>
       <td> Full-precision (32 bits) <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EkFITagLxGNIsqQwksNRNB0B-4FOPb-hMEHyJykvKnlFbQ?e=Dn8c1k">Download</a> </td>
       <td> 72.56 </td>
       <td> 48.51 </td>
       <td> 20.45 </td>
       <td> 24.59 </td>
       <td> 13.55 </td>
       <td> 26.78 </td>
    </tr>
    <tr align="center">
       <td>ImageNet</td>
       <td> abits=2, wbits=2 <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/Eg4hbpGyleBGtgIdVitVWK8Bt1XAu9iVGXlEqAFIsnvrPA?e=nulYxg">Download</a></td>
       <td> abits=4, wbits=4 <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/ErFX38nXblRBr6jvrhLqKNYBKwYiwqslbeGVSlH75P5XGg?e=d3rGC3">Download</a> </td>
       <td> Full-precision (32 bits) <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EmBzpnERlD1HtAbgymP2B8ABZ2DJR_tBjY0c1ho9ETNl0A?e=rlaOoP">Download</a> </td>
       <td> 55.09 </td>
       <td> 29.36 </td>
       <td> 21.65 </td>
       <td> 20.67 </td>
       <td> 11.76 </td>
       <td> 20.86 </td>
    </tr>
</table>

+ Baseline models
<table>
    <tr align="center">
        <th rowspan="2">Dataset</th>
        <th rowspan="2">Models</th>
        <th rowspan=2>Unperturbed Accuracy (%)</th>
        <th colspan=5>Adversarial Accuracy (%)</th>
    </tr>
    <tr align="center">
        <th>CW</th>
        <th>FGSM</th>
        <th>BIM</th>
        <th>PGD</th>
        <th>Average</th>
    </tr>
    <tr align="center">
       <td>MNIST</td>
       <td> MNISTconv <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EumogLgncVBCm932G6fUBPgBcipIcex0GhmG0SLIZdFT2g?e=K5MCh8">Download</a> </td>
       <td> 98.87 </td>
       <td> 3.69 </td>
       <td> 14.32 </td>
       <td> 0.9 </td>
       <td> 0.77 </td>
       <td> 4.92 </td>
    </tr>
    <tr align="center">
       <td>CIFAR-10</td>
       <td> CIFARconv <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EkFITagLxGNIsqQwksNRNB0B-4FOPb-hMEHyJykvKnlFbQ?e=Dn8c1k">Download</a> </td>
       <td> 74.54 </td>
       <td> 13.38 </td>
       <td> 10.28 </td>
       <td> 11.97 </td>
       <td> 10.69 </td>
       <td> 11.58 </td>
    </tr>
    <tr align="center">
       <td>ImageNet</td>
       <td> AlexNet <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/sen9_purdue_edu/EmBzpnERlD1HtAbgymP2B8ABZ2DJR_tBjY0c1ho9ETNl0A?e=rlaOoP">Download</a> </td>
       <td> 53.23 </td>
       <td> 9.94 </td>
       <td> 10.29 </td>
       <td> 10.81 </td>
       <td> 10.30 </td>
       <td> 10.34 </td>
    </tr>
</table>

## Citing this work

```
@inproceedings{
sen2020empir,
title={{\{}EMPIR{\}}: Ensembles of Mixed Precision Deep Networks for Increased Robustness Against Adversarial Attacks},
author={Sanchari Sen and Balaraman Ravindran and Anand Raghunathan},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJem3yHKwH}
}
```
## Copyright

Copyright 2017 - Google Inc., OpenAI and Pennsylvania State University.
