import argparse

from . import train_lander
from pathlib import Path

def many_fine_tunes():
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs="+", type=str, help='location of training runs to fine tune')
    constant_schedule = train_lander.WithStrPolyDecay(initial_learning_rate=0.1, decay_steps=1.0, end_learning_rate=0.1)
    serializer = train_lander.lander_serializer(epochs=20, learning_rate=1e-4, act_noise=constant_schedule)
    serializer.add_serialized_args_to_parser(parser)
    cmd_args = parser.parse_args()
    for folder in cmd_args.folders:
        cmd_args.prev_folder = Path(folder)
        hp = train_lander.generate_hypers(cmd_args)
        hp.start_steps = 10000
        hp.q_importance = 0.5
        train_lander.train(cmd_args, hp, serializer)

#"trained/lander-custom/e:200,l:(0.001,1e-05),w:True,x:Y5SY3KTVQZ3OVXF/seeds/"

if __name__ == "__main__":
    many_fine_tunes()
