from CAS_Acrobot_DQL.train_dql import ENV, ENV_PARAMS, evaluate, initialize_mlp, rollout, train
from CAS_Acrobot_DQL.visualise import animate_acrobot, plot_history, visualize_trajectory


import jax
import jax.random as jr
from jax.random import PRNGKey
from sympy import evaluate


def main():
    key = PRNGKey(42)

    layer_sizes = (ENV.obs_shape[0], 128, 128, 128, ENV.num_actions)
    key, subkey = jr.split(key)
    params = initialize_mlp(layer_sizes, key=subkey)

    key, subkey = jr.split(key)
    params, history = train(params, subkey, num_iters=10000000, update_interval=4, target_interval=100)

    plot_history(history)

    jit_rollout = jax.jit(rollout, static_argnums=[2])
    key, subkey = jr.split(key)
    times, success_rate = evaluate(params, key)
    print("Evaluation Results:")
    print(f"Times: {times}")
    print(f"Success Rate: {success_rate}")

    outputs = jit_rollout(params, subkey, ENV_PARAMS.max_steps_in_episode)

    visualize_trajectory(outputs, blocking=False)
    animate_acrobot(outputs)  # , save_path="acrobot.gif")

if __name__ == '__main__':
    main()
