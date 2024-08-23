from functools import partial
from pathlib import Path
from anchored_rl.utils import save_utils
from anchored_rl.utils.args_utils import Arg_Serializer



def create_train_folder_and_params(experiment_name, hyperparams, cmd_args, serializer: Arg_Serializer):
    """
    Sets up the folders for the experiment and trains the agent.
    """
    # Create the folders
    save_path = save_utils.save_hypers(experiment_name, hyperparams, cmd_args, serializer)
    generated_params = {
        "hp": hyperparams,
        "on_save": partial(save_utils.on_save, replay_save=cmd_args.replay_save, save_path=save_path),
        "logger_kwargs": {"output_dir": save_path}
    }
    if cmd_args.prev_folder:
        actor = lambda: save_utils.load_actor(Path(cmd_args.prev_folder, "models"))
        critic = lambda: save_utils.load_critic(Path(cmd_args.prev_folder, "models"))
        generated_params["actor_critic"] = lambda *args, **kwargs: (actor(), critic())
        generated_params["random_start"] = False
        if cmd_args.anchored:
            replay_buffer = lambda: save_utils.load_replay(cmd_args.prev_folder)
            generated_params["anchored"] = lambda *args, **kwargs: (critic(), replay_buffer())


    return generated_params