import argparse
import train_pendulum
from anchored_rl.utils import args_utils
from pathlib import Path



def many_fine_tunes(anchored = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs="+", type=str, help='location of training runs to fine tune')
    serializer = train_pendulum.pendulum_serializer()
    serializer.abbrev_to_args['e'] = args_utils.Serialized_Argument(name='--epochs', type=int, default=10)
    serializer.abbrev_to_args['g'] = args_utils.Serialized_Argument(name='--gravity', type=float, default=7.0)
    serializer.abbrev_to_args['a'] = args_utils.Serialized_Argument(name='--anchored', action='store_true', default=anchored)
    serializer.add_serialized_args_to_parser(parser)
    cmd_args = parser.parse_args()
    for folder in cmd_args.folders:
        cmd_args.prev_folder = Path(folder)
        print(folder)
        train_pendulum.train(cmd_args, serializer)


def train_many(num_trainings = 6):
    for i in range(num_trainings):
        serializer = train_pendulum.pendulum_serializer()
        cmd_args = args_utils.parse_arguments(serializer, args=["-r"])
        train_pendulum.train(cmd_args, serializer)


if __name__ == "__main__":
    many_fine_tunes(anchored = False)
    # train_many()
