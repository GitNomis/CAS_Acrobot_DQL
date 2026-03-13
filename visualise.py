from CAS_Acrobot_DQL.train_dql import ENV, ENV_PARAMS


import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import seaborn as sns


def animate_acrobot(output, save_path=None):
    """Animate the Acrobot's trajectory.

    Args:
        output (Transition): The output data from the environment.
        save_path (str, optional): The path to save the animation. Defaults to None.

    Returns:
        FuncAnimation: The animation object.
    """

    obs = jnp.asarray(output.obs)
    reward = jnp.asarray(output.reward)
    done = jnp.asarray(output.done)

    # Recover angles from observation
    theta1 = jnp.arctan2(obs[:, 1], obs[:, 0])
    theta2 = jnp.arctan2(obs[:, 3], obs[:, 2])

    # Optional: stop animation at first terminal step
    done_idx = jnp.where(done)[0]
    T = done_idx[0] + 1 if len(done_idx) > 0 else len(obs)

    theta1 = theta1[:T]
    theta2 = theta2[:T]
    reward = reward[:T]

    # Link lengths
    l1 = 1.0
    l2 = 1.0

    # Precompute coordinates
    x0 = jnp.zeros(T)
    y0 = jnp.zeros(T)

    x1 = l1 * jnp.sin(theta1)
    y1 = -l1 * jnp.cos(theta1)

    x2 = x1 + l2 * jnp.sin(theta1 + theta2)
    y2 = y1 - l2 * jnp.cos(theta1 + theta2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True)

    # Goal height line used in Acrobot termination
    ax.axhline(ENV_PARAMS.success_height_threshold, linestyle="--", linewidth=1)

    line, = ax.plot([], [], "o-", lw=3)
    tip_trace, = ax.plot([], [], "-", lw=1, alpha=0.5)
    title = ax.set_title("Acrobot")

    trace_x = []
    trace_y = []

    def init():
        line.set_data([], [])
        tip_trace.set_data([], [])
        title.set_text("Acrobot")
        return line, tip_trace, title

    def update(i):
        xs = [x0[i], x1[i], x2[i]]
        ys = [y0[i], y1[i], y2[i]]

        line.set_data(xs, ys)

        trace_x.append(x2[i])
        trace_y.append(y2[i])
        tip_trace.set_data(trace_x, trace_y)

        title.set_text(f"t = {i * ENV_PARAMS.dt:.2f}s, reward = {reward[i]:.2f}")
        return line, tip_trace, title

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=100 * ENV_PARAMS.dt,
        blit=True,
    )

    if save_path is not None:
        anim.save(save_path, fps=max(1, int(round(1 / ENV_PARAMS.dt))))

    plt.show()
    return anim


def visualize_trajectory(output, blocking=True):
    """Visualize the trajectory of the Acrobot.

    Args:
        output (Transition): The output data from the environment.
        blocking (bool, optional): Whether to block the execution until the plot is closed. Defaults to True.
    """
    obs = output.obs
    action = output.action
    reward = output.reward
    ts = jnp.arange(0, ENV_PARAMS.dt * len(output.obs), ENV_PARAMS.dt)

    fig, ax = plt.subplots(ENV.obs_shape[0] + 2, 1, figsize=(8, 8))
    # first three plots for the system states
    ax[0].set_title('System states over time')

    for d in range(ENV.obs_shape[0]):
        ax[d].plot(ts, obs[:, d], color='C0', label=f'State {d}')
    ax[0].set_title(r'$\cos(\theta_1)$')
    ax[1].set_title(r'$\sin(\theta_1)$')
    ax[2].set_title(r'$\cos{\theta_2}$')
    ax[3].set_title(r'$\sin(\theta_2)$')
    ax[4].set_title(r'$\dot{\theta_1}$')
    ax[5].set_title(r'$\dot{\theta_2}$')

    ax[ENV.obs_shape[0]].plot(ts, action, color='C1', label=f'Actions')
    # ax[3].set_ylim((env.action_space().low, env.action_space().high))
    ax[ENV.obs_shape[0]].set_title('u(t)')
    ax[ENV.obs_shape[0] + 1].plot(ts, reward, color='C2', label='Rewards')
    ax[ENV.obs_shape[0] + 1].set_title('r(t)')

    plt.tight_layout()
    plt.show(block=blocking)


def plot_history(history):
    iters, loss, avg_times, success_rate = history

    fig, ax = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    mask = ~jnp.isnan(loss)
    sns.scatterplot(x=iters[mask], y=loss[mask], s=10, alpha=0.3, ax=ax[0])
    ax[0].set_ylabel("Loss")

    mask = ~jnp.isnan(avg_times)
    sns.regplot(x=iters[mask], y=avg_times[mask], lowess=True,
                scatter_kws={"s": 20, "alpha": 0.3}, line_kws={"color": "orange", "lw": 2, "alpha": 0.6}, ax=ax[1])
    ax[1].set_ylabel("Avg Episode Length")

    mask = ~jnp.isnan(success_rate)
    sns.regplot(x=iters[mask], y=success_rate[mask], lowess=True,
                scatter_kws={"s": 20, "alpha": 0.3}, line_kws={"color": "orange", "lw": 2, "alpha": 0.6}, ax=ax[2])
    ax[2].set_ylabel("Success Rate")
    ax[2].set_xlabel("Training Steps")

    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
