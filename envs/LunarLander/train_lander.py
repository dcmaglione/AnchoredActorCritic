from anchored_rl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from anchored_rl.utils import args_utils
from anchored_rl.utils import train_utils
from .lunar_lander import LunarLander
import tensorflow as tf

class WithStrPolyDecay(tf.optimizers.schedules.PolynomialDecay):
    def __str__(self):
        return f"({self.initial_learning_rate},{self.end_learning_rate})"

def lander_serializer(
        epochs=200,
        steps_per_epoch=5000,
        learning_rate=None
    ):
    if learning_rate is None:
        learning_rate = WithStrPolyDecay(
            1e-3,
            steps_per_epoch*epochs,
            power=2,
            end_learning_rate=1e-5
        )

    return args_utils.Arg_Serializer.join(args_utils.Arg_Serializer(
        abbrev_to_args= {
            'w': args_utils.Serialized_Argument(name='--wind', action="store_true", help='enable wind in env'),
            'steps': args_utils.Serialized_Argument(name='--steps-per-epoch', type=int, help="number of total steps in one epoch", default=steps_per_epoch)
        }), args_utils.default_serializer(epochs, learning_rate, act_noise=0.01, start_steps=10000))


def train(cmd_args, hp, serializer):
    generated_params = train_utils.create_train_folder_and_params("lander-custom", hp, cmd_args, serializer)
    env_fn = lambda: LunarLander(enable_wind=cmd_args.wind)
    ddpg(env_fn, save_freq=1, **generated_params)

def generate_hypers(cmd_args):
    return HyperParams(
        seed=cmd_args.seed,
        steps_per_epoch=cmd_args.steps_per_epoch,
        ac_kwargs={
            "actor_hidden_sizes": (64, 64),
            "critic_hidden_sizes": (128, 128, 128),
            "obs_normalizer": LunarLander().observation_space.high
        },
        start_steps=cmd_args.start_steps,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.99,
        # pi_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 50000, end_learning_rate=1e-5),
        # q_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 50000, end_learning_rate=1e-5),
        pi_lr=cmd_args.learning_rate,
        q_lr=cmd_args.learning_rate,
        batch_size=32,
        act_noise=cmd_args.act_noise,
        max_ep_len=200,
        epochs=cmd_args.epochs,
        train_every=50,
        train_steps=30,
        q_importance=0.5,
    )

if __name__ == '__main__':
    serializer = lander_serializer()
    cmd_args = args_utils.parse_arguments(serializer)
    train(cmd_args, generate_hypers(cmd_args), serializer)
