# Deep-Learning-and-Practice Lab 7
🚀 cGAN & cNF  
🏹 The goal of this lab is to implement conditional GAN and conditional Normalizing Flow to generate object images and human faces.



## Arguments
|Argument|Description|Default|
|---|---|---|
|`'-b', '--batch_size'`|Batch size|64|
|`'-i', '--image_size'`|Image size|64|
|`'-w', '--width'`|Dimension of the hidden layers in normalizing flow|128|
|`'-d', '--depth'`|Depth of the normalizing flow|8|
|`'-n', '--num_levels'`|Number of levels in normalizing flow|3|
|`'-gv', '--grad_value_clip'`|Clip gradients at specific value|0|
|`'-gn', '--grad_norm_clip'`|Clip gradients' norm at specific value|0|
|`'-lrd', '--learning_rate_discriminator'`|Learning rate of discriminator|0.0002|
|`'-lrg', '--learning_rate_generator'`|Learning rate of generator|0.0002|
|`'-lrnf', '--learning_rate_normalizing_flow'`|Learning rate of normalizing flow|0.0005|
|`'-e', '--epochs'`|Number of epochs|300|
|`'-wu', '--warmup'`|Number of warmup epochs|10|
|`'-t', '--task'`|Task 1 or task 2|1 (1-2)|
|`'-m', '--model'`|cGAN or cNF|'dcgan'|
|`'-inf', '--inference'`|Only infer or not|False|
|`'-v', '--verbosity'`|Verbosity level|0 (0-2)|