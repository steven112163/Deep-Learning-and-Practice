# Deep-Learning-and-Practice Lab 5
🚀 CVAE  
🏹 The goal of this lab is to implement CVAE using LSTM with Pytorch for English tense conversion.



## Arguments
|Argument|Description|Default|
|---|---|---|
|`'-hs', '--hidden_size'`|RNN hidden size|256|
|`'-ls', '--latent_size'`|Latent size|32|
|`'-c', '--condition_embedding_size'`|Condition embedding size|8|
|`'-k', '--kl_weight'`|KL weight|0.0|
|`'-kt', '--kl_weight_type'`|Fixed, monotonic or cyclical KL weight|'monotonic'|
|`'-t', '--teacher_forcing_ratio'`|Teacher forcing ratio|0.5|
|`'-tt', '--teacher_forcing_type'`|Fixed or decreasing teacher forcing ratio|'decreasing'|
|`'-lr', '--learning_rate'`|Learning rate|0.007|
|`'-e', '--epochs'`|Number of epochs|100|
|`'-l', '--load'`|Whether load the stored model and accuracies|0|
|`'-s', '--show_only'`|Whether only show the results|0|
|`'-v', '--verbosity'`|Verbosity level|0|