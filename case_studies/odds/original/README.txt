Note: The default parameter settings in this repository, specifically the number of samples and types of noise used to generate statistics, are meant as a quick proof-of-concept to run on minimal hardware. To achieve higher robustness to stronger attacks, it is necessary to increase these parameters.

The code of our method is in tf_robustify. It works with both pytorch and tensorflow.
Two scripts are provided: tensorflow_example.py and torch_example.py.
The torch example is the easiest, as it can just be run and will fetch both data as well as model by itself.
For the tensorflow example, use fetch_Madry_ResNet.py to fetch the models and place them in the appropriate ckpt_dir (parameter of tensorflow_example.py).

