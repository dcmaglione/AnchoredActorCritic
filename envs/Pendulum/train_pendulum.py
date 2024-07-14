from .Pendulum import PendulumEnv
from anchored_rl.rl_algs.ddpg.ddpg import ddpg, HyperParams
import anchored_rl.utils.train_utils as train_utils
import anchored_rl.utils.args_utils as args_utils

def pendulum_serializer():
    return args_utils.Arg_Serializer.join(args_utils.Arg_Serializer({
        "s_p": args_utils.Serialized_Argument("--setpoint", type=float, default=0.0),
        "g": args_utils.Serialized_Argument("--gravity", type=float, default=9.81),
    }), args_utils.default_serializer(epochs=10, learning_rate=1e-4))

def parse_args_and_train(args=None):
    serializer = pendulum_serializer()
    print(serializer.abbrev_to_args.keys())
    cmd_args = args_utils.parse_arguments(serializer)
    hp = HyperParams(epochs=cmd_args.epochs, q_lr=cmd_args.learning_rate, pi_lr=cmd_args.learning_rate, seed=cmd_args.seed, start_steps=cmd_args.start_steps, max_ep_len=200)
    generated_params = train_utils.create_train_folder_and_params("Pendulum-custom", hp, cmd_args, serializer)
    env_fn = lambda: PendulumEnv(g=cmd_args.gravity, setpoint=cmd_args.setpoint)
    ddpg(env_fn, **generated_params)

if __name__ == '__main__':
    parse_args_and_train()
