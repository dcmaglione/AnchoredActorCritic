import argparse
import anchored_rl.envs.LunarLander.train_lander as train_lander
from anchored_rl.utils import args_utils
from pathlib import Path

def many_fine_tunes(anchored = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs="+", type=str, help='location of training runs to fine tune')
    serializer = train_lander.lander_serializer(epochs=10, learning_rate=1e-4)
    serializer.abbrev_to_args['a'] = args_utils.Serialized_Argument(name='--anchored', action='store_true', default=anchored)
    serializer.add_serialized_args_to_parser(parser)
    cmd_args = parser.parse_args()
    for folder in cmd_args.folders:
        cmd_args.prev_folder = Path(folder)
        hp = train_lander.generate_hypers(cmd_args)
        hp.start_steps = 10
        train_lander.train(cmd_args, hp, serializer)

#"trained/lander-custom/e:200,l:(0.001,1e-05),w:True,x:Y5SY3KTVQZ3OVXF/seeds/"

if __name__ == "__main__":
    many_fine_tunes(anchored = True)