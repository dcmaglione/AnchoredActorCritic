# Anchored Actor Critic
## Installation
* Install nix from https://github.com/DeterminateSystems/nix-installer
* Run `nix develop`
* You are in a shell with python and the libraries required for running our custom DDPG
## Running Single Experiments
### Pendulum

#### Train single pendulum on "simulation environment (gravity is 13 m/s^2)" while retaining replay buffer with 6 epochs and seed 1:
```bash
python -m envs.Pendulum.train_pendulum --replay_save --seed 1 --epochs 6 --gravity 13
```
#### Test the pendulum on same gravity
```bash
python -m envs.Pendulum.test_pendulum -lr trained/Pendulum-custom/e:6,g:13,l:0.0001,s_s:1000,x:A36BEX467PK23PZ/seeds/1/epochs/5 --gravity 13
```
#### Test the pendulum on "real environment (gravity is 9.81 m/s^2)"
```bash
python -m envs.Pendulum.test_pendulum.py -lr trained/Pendulum-custom/e:6,g:13,l:0.0001,s_s:1000,x:A36BEX467PK23PZ/seeds/1/epochs/5 --gravity 9.81
```
#### Continue training in reality without anchors (under gravity 9.81 m/s^2)
```bash
python -m envs.Pendulum.train_pendulum --replay_save --seed 1 --epochs 6 --gravity 13 --
```
### Reacher
#### Train reacher on simulated env (distance to dot is within 0.2m)
```bash
python -m envs.Reacher.train_reacher --replay_save --seed 1 --epochs 70 --distance 0.2
```
#### Test reacher on 0.2m (same as training)
```bash
python -m envs.Reacher.test_reacher -lr --distance 0.2 rained/Reacher-custom/d:0.2,e:70,l:0.003,s_s:1000,x:E6ZGKFUDTWDQV5P/seeds/1/epochs/69
```
#### Test reacher on 0.1m (subset of training distance)
```bash
python -m envs.Reacher.test_reacher -lr --distance 0.1 rained/Reacher-custom/d:0.2,e:70,l:0.003,s_s:1000,x:E6ZGKFUDTWDQV5P/seeds/1/epochs/69
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

### Original Performance:

| Distance | Reward |
|----------|--------|
| 0.1m     | 330.3559+-39.9100    |
| 0.2m     | 350.4692+-21.3568    |

### After finetuning to 0.1m (without anchors):

| Distance | Reward |
|----------|--------|
| 0.1m     | 350.9585+-43.8958  |
| 0.2m     | 276.8328+-86.8765    |

### With ours (with anchors):

| Distance | Reward |
|----------|--------|
| 0.1m     | 336.9751+-46.2759    |
| 0.2m     | 347.0957+-26.0022    |