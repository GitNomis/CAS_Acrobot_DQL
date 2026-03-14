from CAS_Acrobot_DQL.train_dql import ENV, ENV_PARAMS, evaluate, initialize_mlp, rollout, train, evaluate
from CAS_Acrobot_DQL.visualise import animate_acrobot, plot_history, visualize_trajectory


import jax
import jax.random as jr
from jax.random import PRNGKey

EXPERIMENT_N=3


def main():
    key = PRNGKey(42)

    layer_sizes = (ENV.obs_shape[0], 128, 128, 128, ENV.num_actions)
    key, subkey = jr.split(key)
    params = initialize_mlp(layer_sizes, key=subkey)

    num_iters = 20000000
    K = [50,200,1000,5000]
    
    key, subkey = jr.split(key)
    params, history = train(params, subkey, num_iters=num_iters, update_interval=4, target_interval=K[EXPERIMENT_N])

    plot_history(history, save_path=f"training_history{num_iters}_{K[EXPERIMENT_N]}.png")

    jit_rollout = jax.jit(rollout, static_argnums=[2])
    key, subkey = jr.split(key)
    max_success_streak, avg_success_streak = evaluate(params, subkey)
    print("Evaluation Results:")
    print(f"Max Success Streak: {max_success_streak}")
    print(f"Avg Success Streak: {avg_success_streak}")

    outputs, _ = jit_rollout(params, key, ENV_PARAMS.max_steps_in_episode)

    visualize_trajectory(outputs, blocking=False, save_path=f"trajectory{num_iters}_{K[EXPERIMENT_N]}.png")
    animate_acrobot(outputs, save_path=f"acrobot{num_iters}_{K[EXPERIMENT_N]}.gif")


if __name__ == '__main__':
    main()
