# Anchored Actor Critic

Fork of bmabsout/AnchoredActorCritic for CPS Lab Project.

## Setup

**Nix** is recommended to initialize the environment for this project. To install Nix we recommend the [Determinate Systems Nix Installer](https://determinate.systems/posts/determinate-nix-installer/) for your respective OS. Once Nix is installed, run the following command in the root of the project directory to setup the environment.

```bash
$ nix develop
```

You should notice your shell change. You can verify the Python version you are using with `which python` and if it is coming from the `/nix` directory then everything is all set.

## Example Usage

This example is for the Pendulum which is found under `AnchoredActorCritic/envs/Pendulum`. Let's quickly go over the three files contained within the directory. You'll notice other environments will generally adhere to this structure. The primary file is `Pendulum.py`, this actually defines the Pendulum environment, initialization, step function, reward function, etc. Basically, this is the meat on the bone and what we will be experimenting with the most. 

**Side note, if you don't have experience with gymnasium I'd highly advise giving the [docs](https://gymnasium.farama.org/) a quick run through!**

The next two files of note are `train_pendulum.py` and `test_pendulum.py`. They don't warrant much of an explanation as the names are self describing, but one is a script for training the agent with configurable hyperparameters and the other lets us test/verify our agents performance. Below is some example usage.

**Training**

A quick note about this, when you train the pendulum a `trained/` directory will be created where you run the script from. Please keep that in mind as I have written these paths assuming you call the script from the root of the repository.

```bash
$ python envs/Pendulum/train_pendulum.py
```

**Testing**
```bash
$ python envs/Pendulum/test_pendulum.py -lr trained/
```
