# Deep-Learning-and-Practice Lab 7
🚀 cGAN & cNF  
🏹 The goal of this lab is to implement conditional GAN and conditional Normalizing Flow to generate object images and human faces.



## Arguments
|Argument|Description|Default|
|---|---|---|
|`'-b', '--batch_size'`|Batch size|32|
|`'-i', '--image_size'`|Image size|64|
|`'-w', '--width'`|Dimension of the hidden layers in normalizing flow|64|
|`'-d', '--depth'`|Depth of the normalizing flow|16|
|`'-n', '--num_levels'`|Number of levels in normalizing flow|3|
|`'-g', '--grad_norm_clip'`|Clip gradients during training|50|
|`'-lrd', '--learning_rate_discriminator'`|Learning rate of discriminator|0.0001|
|`'-lrg', '--learning_rate_generator'`|Learning rate of generator|0.0001|
|`'-lrnf', '--learning_rate_normalizing_flow'`|Learning rate of normalizing flow|0.001|
|`'-e', '--epochs'`|Number of epochs|
|`'-t', '--task'`|Task 1 or task 2|1|
|`'-m', '--model'`|cGAN or cNF|0|
|`'-v', '--verbosity'`|Verbosity level|0|