# Anchored Actor Critic
## Installation
* Install nix from https://github.com/DeterminateSystems/nix-installer
* Run `nix develop`
* You are in a shell with python and the libraries required for running our custom DDPG
## Running Experiments
Patrick, results should be different
* Train single pendulum on "simulation environment (gravity is 13 m/s^2)" while retaining replay buffer with 6 epochs and seed 1:
`python anchored_rl/envs/Pendulum/train_pendulum.py --replay_save --seed 1 --epochs 6 --gravity 13`
* Test the pendulum on same gravity
`python anchored_rl/envs/Pendulum/test_pendulum.py -lr trained/Pendulum-custom/e:6,g:13,l:0.0001,s_s:1000,x:A36BEX467PK23PZ/seeds/1/epochs/5 --gravity 13`
* Test the pendulum on "real environment (gravity is 9.81 m/s^2)"
`python anchored_rl/envs/Pendulum/test_pendulum.py -lr trained/Pendulum-custom/e:6,g:13,l:0.0001,s_s:1000,x:A36BEX467PK23PZ/seeds/1/epochs/5 --gravity 9.81`
* Continue training in reality without anchors (under gravity 9.81 m/s^2)
`python anchored_rl/envs/Pendulum/train_pendulum.py --replay_save --seed 1 --epochs 6 --gravity 13 --`