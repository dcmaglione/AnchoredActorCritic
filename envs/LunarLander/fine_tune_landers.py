import argparse

import keras
from . import train_lander
from pathlib import Path

def many_fine_tunes():
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs="+", type=str, help='location of training runs to fine tune')
    serializer = train_lander.lander_serializer(epochs=20, learning_rate=1e-4, act_noise=keras.optimizers.schedules.ConstantSchedule(0.1))
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
