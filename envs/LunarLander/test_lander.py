import argparse
from .lunar_lander import LunarLander
from anchored_rl.utils import test_utils

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folders', nargs='+', type=str, help='location of the training run')
    parser.add_argument('-r', '--render', action="store_true", help="render the env as it evaluates")
    parser.add_argument('-n', '--num_tests', type=int, default=20)
    parser.add_argument('-s', '--steps', type=int, default=200)
    parser.add_argument('-w', '--wind', action="store_true", help="enable wind")
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('-l', '--use_latest', action="store_true", help="use the latest training run from the save_folder")
    return parser.parse_args(args)

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    cmd_args = parse_args()
    test_utils.run_tests(LunarLander(enable_wind=cmd_args.wind, render_mode="human" if cmd_args.render else None), cmd_args)

# lander after 200 epochs with wind trained with wind
# 55.5706+-18.6731

# lander after 200 epochs without wind trained with wind
# 62.9119+-17.1236

# lander after 10 epochs of finetune without wind tested without wind
# 37.0747+-20.8988

# lander after 10 epochs of finetune without wind tested with wind
# 24.8270+-10.8566

# lander with anchors after 10 epochs of finetune without wind tested without wind
