from .Pendulum import PendulumEnv
import argparse
from anchored_rl.utils import test_utils


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folders', nargs='+', type=str, help='location of the training run')
    parser.add_argument('-r', '--render', action="store_true", help="render the env as it evaluates")
    parser.add_argument('-n', '--num_tests', type=int, default=20)
    parser.add_argument('-g', '--gravity', type=float, default=9.81)
    parser.add_argument('-s_p', '--setpoint', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--store_results', type=str, default=None, help='path to store the test results')

    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('-l', '--use_latest', action="store_true", help="use the latest training run from the save_folder")
    return parser.parse_args(args)

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    cmd_args = parse_args()
    test_utils.run_tests(
        PendulumEnv(
            render_mode="human" if cmd_args.render else None,
            g=cmd_args.gravity,
            setpoint=cmd_args.setpoint
        ),
        cmd_args
    )
