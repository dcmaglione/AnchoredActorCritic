# Anchored Actor Critic
## Installation
* Install nix from https://github.com/DeterminateSystems/nix-installer
* Run `nix develop`
* You are in a shell with python and the libraries required for running our custom DDPG
## Running Single Experiments

### Pendulum

#### Train single pendulum on a "simulation environment (gravity is 13 m/s^2)" while retaining replay buffer with 6 epochs and seed 1:
```bash
python -m envs.Pendulum.train_pendulum --replay_save --seed 1 --epochs 6 --gravity 13
```
#### Test the pendulum on the same gravity
```bash
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a_n:0.1,e:6,g:13,l:0.0001,s_s:1000,x:PDSR75BG7XSZ5DS/seeds/1/epochs/5 --gravity 13
```
#### Test the pendulum on the "real environment (gravity is 4 m/s^2)"
```bash
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a_n:0.1,e:6,g:13,l:0.0001,s_s:1000,x:PDSR75BG7XSZ5DS/seeds/1/epochs/5 --gravity 4
```
#### Continue training in reality without anchors (under gravity 4 m/s^2)
```bash
python -m envs.Pendulum.train_pendulum -p trained/Pendulum-custom/a_n:0.1,e:6,g:13,l:0.0001,s_s:1000,x:PDSR75BG7XSZ5DS/seeds/1/epochs/5 --gravity 4 --epochs 5 --seed 1 -a_n 0.01
```

#### Continue training in reality with anchors (under gravity 4 m/s^2)
```bash
python -m envs.Pendulum.train_pendulum -a -p trained/Pendulum-custom/a_n:0.1,e:6,g:13,l:0.0001,s_s:1000,x:PDSR75BG7XSZ5DS/seeds/1/epochs/5 --gravity 4 --epochs 5 --seed 1 -a_n 0.01
```

#### Test fine tuned pendulum on "old environment (gravity is 13 m/s^2)" without anchors
```bash
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a_n:0.01,e:6,g:4,l:0.0001,p:QMOUS4S,s_s:1000,x:PDSR75BG7XSZ5DS/seeds/1/epochs/5 --gravity 13
```

#### Test fine tuned pendulum on "old environment (gravity is 13 m/s^2)" with anchors
```bash
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a:True,a_n:0.01,e:6,g:4,l:0.0001,p:QMOUS4S,s_s:1000,x:PDSR75BG7XSZ5DS/seeds/1/epochs/5 --gravity 13
```

### Reacher
#### Train reacher on simulated env (distance to dot is within 0.2m)
```bash
python -m envs.Reacher.train_reacher --replay_save --seed 1 --epochs 70 --distance 0.2
```
#### Test reacher on 0.2m (same as training)
```bash
python -m envs.Reacher.test_reacher -lr --distance 0.2 trained/Reacher-custom/d:0.2,e:70,l:0.003,s_s:1000,x:E6ZGKFUDTWDQV5P/seeds/1/epochs/69
```
#### Test reacher on 0.1m (subset of training distance)
```bash
python -m envs.Reacher.test_reacher -lr --distance 0.1 trained/Reacher-custom/d:0.2,e:70,l:0.003,s_s:1000,x:E6ZGKFUDTWDQV5P/seeds/1/epochs/69
```
#### Fine tune reacher without anchors on distance to dot within 0.1m
We use a low action noise and enough start steps to fill the replay buffer for best fine tuning performance
```bash
python -m envs.Reacher.train_reacher -s_s 5000 -s 1 -e 15 -d 0.1 -a_n 0.01 -p trained/Reacher-custom/d:0.2,e:70,l:0.003,s_s:1000,x:E6ZGKFUDTWDQV5P/seeds/1/epochs/69
```
#### Fine tune reacher with anchors on distance to dot within 0.1m
```bash
python -m envs.Reacher.train_reacher -a -s_s 5000 -s 1 -e 15 -d 0.1 -a_n 0.01 -p trained/Reacher-custom/d:0.2,e:70,l:0.003,s_s:1000,x:E6ZGKFUDTWDQV5P/seeds/1/epochs/69
```
#### Test fine tuned reacher without anchors on old distance (0.2m)
```bash
python -m envs.Reacher.test_reacher -d 0.2 -r trained/Reacher-custom/a_n:0.01,d:0.1,e:15,l:0.003,p:KXOA45F,s_s:5000,x:UOMCKM77Z5CH7FV/seeds/1/epochs/14
```
#### Test fine tuned reacher with anchors on old distance (0.2m)
```bash
python -m envs.Reacher.test_reacher -d 0.2 -r trained/Reacher-custom/a:True,a_n:0.01,d:0.1,e:15,l:0.003,p:KXOA45F,s_s:5000,x:UOMCKM77Z5CH7FV/seeds/1/epochs/14
```

### LunarLander
#### Train lunar lander on simulated env (with wind)
```bash
python -m envs.LunarLander.train_lander --seed 1 --replay_save -w
```
#### Test lander on env with wind (same as training)
```bash
python -m envs.LunarLander.test_lander "trained/lander-custom/a_n:0.01,e:50,l:0.001,s_s:10000,w:True,x:EEUBTEYC3HCGRJC/seeds/1/epochs/49" -w
```
#### Test lander without wind ("real" environment)
```bash
python -m envs.LunarLander.test_lander "trained/lander-custom/a_n:0.01,e:50,l:0.001,s_s:10000,w:True,x:EEUBTEYC3HCGRJC/seeds/1/epochs/49"
```
#### Fine tune lander without anchors on the real environment
```bash
python -m envs.LunarLander.fine_tune_landers "trained/lander-custom/a_n:0.01,e:50,l:0.001,s_s:10000,w:True,x:EEUBTEYC3HCGRJC/seeds/1/epochs/49" --seed 1 --epochs 10
```
#### Fine tune lander with anchors on the real environment
```bash
python -m envs.LunarLander.fine_tune_landers "trained/lander-custom/a_n:0.01,e:50,l:0.001,s_s:10000,w:True,x:EEUBTEYC3HCGRJC/seeds/1/epochs/49" --seed 1 --epochs 10 -a
```
#### Test fine tuned lander without anchors on the old environment (with wind)
```bash
python -m envs.LunarLander.test_lander "trained/lander-custom/a:True,a_n:0.01,e:10,l:1e-05,p:7G5JJAX,s_s:5000,x:Z6KABHFRFQKXAOP/seeds/1/epochs/9" -w
```
#### Test fine tuned lander with anchors on the old environment (with wind)
```bash
python -m envs.LunarLander.test_lander "trained/lander-custom/a_n:0.01,e:10,l:1e-05,p:7G5JJAX,s_s:5000,x:Z6KABHFRFQKXAOP/seeds/1/epochs/9" -w
```
#### Test fine tuned lander without anchors on the new environment (without wind)
```bash
python -m envs.LunarLander.test_lander "trained/lander-custom/a:True,a_n:0.01,e:10,l:1e-05,p:7G5JJAX,s_s:5000,x:Z6KABHFRFQKXAOP/seeds/1/epochs/9"
```
#### Test fine tuned lander with anchors on the new environment (without wind)
```bash
python -m envs.LunarLander.test_lander "trained/lander-custom/a_n:0.01,e:10,l:1e-05,p:7G5JJAX,s_s:5000,x:Z6KABHFRFQKXAOP/seeds/1/epochs/9"
```


## Results

### Pendulum

#### Original Performance:

| Gravity | Reward            |
|---------|-------------------|
| 13      | 377.3409+-14.3550 |
| 4       | 352.3740+-69.8369 |

#### After finetuning to gravity 4 (without anchors):

| Gravity | Reward            |
|---------|-------------------|
| 13      | 226.9712+-39.8655 |
| 4       | 387.4662+-7.4814  |

#### With ours (with anchors):

| Gravity | Reward            |
|---------|-------------------|
| 13      | 377.6607+-13.6567 |
| 4       | 370.0494+-53.2055 |


### Reacher

Consider adding min and max reward

#### Original Performance:

| Distance | Reward            |
|----------|-------------------|
| 0.1m     | 330.3559+-39.9100 |
| 0.2m     | 350.4692+-21.3568 |

#### After finetuning to 0.1m (without anchors):

| Distance | Reward            |
|----------|-------------------|
| 0.1m     | 350.9585+-43.8958 |
| 0.2m     | 276.8328+-86.8765 |

#### With ours (with anchors):

| Distance | Reward            |
|----------|-------------------|
| 0.1m     | 336.9751+-46.2759 |
| 0.2m     | 347.0957+-26.0022 |

### Lander

#### Original Performance:

| Wind     | Reward            |
|----------|-------------------|
| False    | 138.7296+-70.4703 |
| True     | 131.1230+-74.6081 |

#### After finetuning to no wind (without anchors):

| Wind     | Reward            |
|----------|-------------------|
| False    | 97.1273+-116.2350 |
| True     | 76.9587+-103.7659 |

#### With ours (with anchors):

| Wind     | Reward            |
|----------|-------------------|
| False    | 235.7062+-84.9603 |
| True     | 161.9801+-98.4721 |
