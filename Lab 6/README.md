# Deep-Learning-and-Practice Lab 6
🚀 DQN & DDPG  
🏹 The goal of this lab is to solve lunar landing using DQN, and solve continuous lunar landing using DDPG.



## Arguments
### DQN
|Argument|Default|
|---|---|
|`'-d', '--device'`|'cuda'|
|`'-m', '--model'`|'dqn.pth'|
|`'--logdir'`|'log/dqn'|
|`'--warmup'`|10000|
|`'--episode'`|1200|
|`'--capacity'`|10000|
|`'--batch_size'`|128|
|`'--lr'`|.0005|
|`'--eps_decay'`|.995|
|`'--eps_min'`|.01|
|`'--gamma'`|.99|
|`'--freq'`|4|
|`'--target_freq'`|100|
|`'--test_only'`|'store_true'|
|`'--render'`|'store_true'|
|`'--seed'`|20200519|
|`'--test_epsilon'`|.001|

### DDPG
|Argument|Default|
|---|---|
|'-d', '--device'`|'cuda'|
|`'-m', '--model'`|'ddpg.pth'|
|`'--logdir'`|'log/ddpg'|
|`'--warmup'`|10000|
|`'--episode'`|1200|
|`'--batch_size'`|64|
|`'--capacity'`|500000|
|`'--lra'`|1e-3|
|`'--lrc'`|1e-3|
|`'--gamma'`|.99|
|`'--tau'`|.005|
|`'--test_only'`|'store_true'|
|`'--render'`|'store_true'|
|`'--seed'`|20200519|