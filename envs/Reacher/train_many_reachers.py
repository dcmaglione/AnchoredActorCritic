import argparse
import train_reacher

# options = ['DIAMETER', '']
# def parse_arguments(args=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('experiment_name', type=str, help='location to save the training run')
#     parser.add_argument('type', type=str, choices=options)
#     parser.add_argument('-a', '--anchored', action='store_true', help='whether to enable anchors or not')
#     parser.add_argument('-d', '--goal_distance', type=float, default=0.2, help='radius of points from the center')
#     parser.add_argument('-b', '--goal_bias', type=float, default=0.0, help='bias of points from the center')
#     parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
#     parser.add_argument('--save_replay', action='store_true', help='whether to save the replay buffer')
#     parser.add_argument('-s', '--seed', type=int, default=int(time.time()* 1e5) % int(1e6))
#     parser.add_argument('-r', '--learning_rate', type=float, default=3e-3)

#     cmd_args = parser.parse_args(args)
#     if cmd_args.anchored and not cmd_args.initialized_folder:
#         parser.error("cannot enable anchors without specifying --initialized_folder")
#     cmd_args.save_path = f"{get_seed_folder_path(cmd_args)}/epochs"
#     return cmd_args


def train_shrinking_circle():
    for i in range(6):
        train_reacher.train_reacher(train_reacher.parse_arguments(["reacher", "--distance", f"{0.2*((i+1.0)/6.0)}", "-r"]))

def fine_tune(anchored = False):
    for i in range(6):
        cmd_args = train_reacher.parse_arguments([
            "reacher",
            "--epochs", "10",
            "--distance", f"{0.2*((i+1.0)/6.0)}",
            "--prev_folder", 'reacher/d:0.2,e:50,l:0.003,x:PKUZRZSQ7CS6S7O/seeds/601909/epochs/49'
        ])
        cmd_args.anchored = anchored
        print(cmd_args)
        train_reacher.train_reacher(cmd_args)

def many_fine_tunes(anchored = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs="+", type=str, help='location of training runs to fine tune')
    cmd_args = parser.parse_args()
    for folder in cmd_args.folders:
        parsed_args = train_reacher.parse_arguments([
            "reacher",
            "--epochs", "10",
            "--distance", "0.1",
            "--prev_folder", folder,
        ])
        parsed_args.anchored = anchored
        print(parsed_args)
        train_reacher.train_reacher(parsed_args)



def train_many_full_circles():
    for i in range(6):
        train_reacher.parse_args_and_train(["reacher", "-r"])


if __name__ == "__main__":
    # many_fine_tunes(anchored = False)
    train_many_full_circles()
