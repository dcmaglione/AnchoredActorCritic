from anchored_rl.rl_algs.ddpg.ddpg import ddpg, HyperParams
from anchored_rl.utils import args_utils, train_utils
import Pendulum


pendulum_serializer = lambda: args_utils.Arg_Serializer.join(args_utils.Arg_Serializer(
    abbrev_to_args= {
        'setpoint': args_utils.Serialized_Argument(name='--setpoint', type=float, default=0.0, help='setpoint'),
            'g': args_utils.Serialized_Argument(name='--gravity', type=float, default=15.0, help='gravity')
    }), args_utils.default_serializer(epochs=20))

def train(cmd_args, serializer):
    hp = HyperParams(
        seed=cmd_args.seed,
        steps_per_epoch=1000,
        ac_kwargs={
            "actor_hidden_sizes": (32, 32),
            "critic_hidden_sizes": (64, 64),
        },
        start_steps=1000,
        replay_size=int(1e5),
        gamma=0.9,
        polyak=0.995,
        # pi_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 50000, end_learning_rate=1e-5),
        # q_lr=tf.optimizers.schedules.PolynomialDecay(1e-3, 50000, end_learning_rate=1e-5),
        pi_lr=cmd_args.learning_rate,
        q_lr=cmd_args.learning_rate,
        batch_size=200,
        act_noise=0.05,
        max_ep_len=200,
        epochs=cmd_args.epochs,
        train_every=50,
        train_steps=30,
    )
    generated_params = train_utils.create_train_folder_and_params("Pendulum-custom", hp, cmd_args, serializer)
    env_fn = lambda: Pendulum.PendulumEnv(g=cmd_args.gravity, setpoint=cmd_args.setpoint)
    ddpg(env_fn, **generated_params)

if __name__ == '__main__':
    serializer = pendulum_serializer()
    cmd_args = args_utils.parse_arguments(serializer)
    train(cmd_args, serializer)
