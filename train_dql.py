from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import flashbax as fbx
import optax
from CAS_Acrobot_DQL.custom_acrobot import CustomAcrobot, EnvState, EnvParams


ENV = CustomAcrobot()
ENV_PARAMS = ENV.default_params
# ENV_PARAMS = ENV_PARAMS.replace(max_steps_in_episode=1000)


class Carry(NamedTuple):
    obs: jax.Array
    state: EnvState
    rng: jax.Array


class TState(NamedTuple):
    buffer_state: any
    params: list
    target_params: list
    opt_state: optax.OptState
    env_state: EnvState
    obs: jax.Array
    i: int


class Transition(NamedTuple):
    obs: jax.Array
    action: int
    reward: float
    next_obs: jax.Array
    done: bool


def train(params, key, num_iters, update_interval=4, target_interval=100):
    """Train the model using the specified parameters.

    Args:
        params (list): The model parameters.
        key (PRNGKey): The random key for initialization.
        num_iters (int): The number of training iterations.
        update_interval (int, optional): The interval for updating the model. Defaults to 4.
        target_interval (int, optional): The interval for updating the target network. Defaults to 100.

    Returns:
        _type_: The trained model parameters and the training history.
    """

    gamma = 0.99

    eps_start = 1.0
    eps_end = 0.02
    decay_max = num_iters * 0.8

    print_interval = num_iters // 20
    eval_interval = num_iters // 1000

    lr = 0.001
    optim = optax.adam(learning_rate=lr)
    opt_state = optim.init(params)

    buffer_size = 100000
    min_buffer_size = 20000
    batch_size = 128
    buffer = fbx.make_item_buffer(buffer_size, min_buffer_size, batch_size)
    buffer_state = buffer.init(Transition(obs=jnp.zeros(ENV.obs_shape), action=0, reward=0.0, next_obs=jnp.zeros(ENV.obs_shape), done=False))

    key, rng_reset = jr.split(key)
    obs, env_state = ENV.reset(rng_reset, ENV_PARAMS)

    def update_params(state: TState, rng: jax.Array):
        """Update the parameters of the model.

        Args:
            state (TState): The current state containing the model.
            rng (jax.Array): The random number generator key.

        Returns:
            TState: The updated state.
        """
        params = state.params
        transitions = buffer.sample(state.buffer_state, rng).experience
        loss, grads = jax.value_and_grad(loss_Q, argnums=0)(params, state.target_params, transitions, gamma)
        updates, opt_state = optim.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_state = state._replace(params=new_params, opt_state=opt_state)
        return new_state, loss

    def step(state: TState, key: jax.Array):
        """Perform a single step in the training process.

        Args:
            state (TState): The current state containing the model.
            key (jax.Array): The random number generator key.

        Returns:
            TState: The updated state.
            Tuple: A tuple containing the iteration number, loss, average times, and success rate.
        """

        rng_action, rng_step, rng_loss = jr.split(key, 3)
        # Generate transition
        eps = eps_end + (eps_start - eps_end) * jnp.maximum(0.0, 1.0 - state.i / decay_max)
        action = get_action(state.params, state.obs, rng_action, eps)
        next_obs, next_state, reward, done, _ = ENV.step(rng_step, state.env_state, action, ENV_PARAMS)
        transition = Transition(obs=state.obs, action=action, reward=reward, next_obs=next_obs, done=done)

        buffer_state = buffer.add(state.buffer_state, transition)
        state = state._replace(buffer_state=buffer_state, env_state=next_state, obs=next_obs)

        # Update parameters
        do_update = buffer.can_sample(state.buffer_state) & ((state.i % update_interval) == 0)
        state, loss = jax.lax.cond(do_update, lambda s: update_params(s, rng_loss), lambda s: (s, jnp.nan), state)

        # Evaluate policy
        avg_times, success_rate = jax.lax.cond(state.i % eval_interval == 0, lambda p: evaluate(p, rng_loss), lambda _: (jnp.nan, jnp.nan), state.params)

        # Update target network
        state = jax.lax.cond(state.i % target_interval == 0, lambda s: s._replace(target_params=s.params), lambda s: s, state)

        do_print = (state.i % print_interval) == 0
        jax.lax.cond(
            do_print,
            lambda _: jax.debug.print("iter {i}, loss={loss}", i=state.i, loss=loss),
            lambda _: None,
            operand=None,
        )

        state = state._replace(i=state.i + 1)

        return state, (state.i, loss, avg_times, success_rate)

    iter_keys = jr.split(key, num_iters)
    state = TState(buffer_state=buffer_state, params=params, target_params=params, opt_state=opt_state, env_state=env_state, obs=obs, i=0)

    state, history = jax.lax.scan(step, state, (iter_keys))
    params = state.params

    return params, history


def evaluate(params: list, rng: jax.Array):
    """Evaluate the policy by running multiple episodes.

    Args:
        params (list): The model parameters.
        rng (jax.Array): The random number generator key.

    Returns:
        Tuple: The average success times and success rate.
    """
    n_episodes = 50
    parallel_rollout = jax.vmap(rollout, in_axes=(None, 0, None))
    keys = jr.split(rng, n_episodes)
    outputs = parallel_rollout(params, keys, ENV_PARAMS.max_steps_in_episode)

    times = jnp.argmax(outputs.done, 1)
    success = times < (ENV_PARAMS.max_steps_in_episode - 1)
    avg_success_times = jnp.sum(jnp.where(success, times, 0)) / jnp.sum(success)
    success_rate = jnp.sum(success) / n_episodes

    return avg_success_times, success_rate


def get_action(params: list, obs: jax.Array, rng: jax.Array, epsilon: float = 0.0):
    """Choose an action to take given the current observation.

    Args:
        params (list): The model parameters.
        obs (jax.Array): The current observation.
        rng (jax.Array): The random number generator key.
        epsilon (float, optional): The exploration rate. Defaults to 0.0.

    Returns:
        jax.Array: The action to take.
    """

    qs = q_network(params, obs)
    rng_eps, rng_act = jr.split(rng)
    action = jnp.where(jr.uniform(rng_eps) < epsilon, ENV.action_space().sample(rng_act), jnp.argmax(qs))
    return action


def loss_Q(params: list, target_params: list, transitions: Transition, gamma: float = 0.99):
    """Compute the loss for the Q-network.

    Args:
        params (list): The model parameters.
        target_params (list): The target model parameters.
        transitions (Transition): The transition data.
        gamma (float, optional): The discount factor. Defaults to 0.99.

    Returns:
        jax.Array: The computed loss.
    """
    q_values = jax.vmap(q_network, in_axes=(None, 0))(params, transitions.obs)
    predicted = q_values[jnp.arange(q_values.shape[0]), transitions.action]

    next_q_values = jax.vmap(q_network, in_axes=(None, 0))(target_params, transitions.next_obs)
    max_next_q = jnp.max(next_q_values, axis=1)

    target = jnp.where(transitions.done, transitions.reward, transitions.reward + gamma * max_next_q)
    target = jax.lax.stop_gradient(target)
    loss = jnp.mean((predicted - target) ** 2)

    return loss


def initialize_mlp(layer_sizes: tuple, key: PRNGKey, scale: float = 1e-2):
    """ Initialize the parameters of a multi-layer perceptron (MLP).
    Inputs:
        layer_sizes (tuple) Tuple of shapes of the neural network layers. Includes the input shape, hidden layer shape, and output layer shape.
        key (PRNGKey) Random key for parameter initialization.
        scale (float) standard deviation of initial weights and biases

    Return: 
        params (List) Tuple of weights and biases - [ (weights_1, biases_1), ..., (weights_n, biases_n) ]
    """
    keys = jr.split(key, 2 * len(layer_sizes))
    params = []

    for i in range(len(layer_sizes[:-1])):
        input, output = layer_sizes[i], layer_sizes[i + 1]
        W = jr.normal(keys[i], (input, output)) * scale
        b = jr.normal(keys[i + 1], (output,)) * scale
        params.append((W, b))

    return params


def q_network(params: list, x: jax.Array):
    """
    MLP that predicts Q-values for each action.

    Inputs:
        params (PyTree): network parameters
        x (D,): observation
    Returns:
        q_values (num_actions): predicted Q-values for each action
    """

    for (W, b) in params[:-1]:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)

    W, b = params[-1]
    return jnp.dot(x, W) + b


def rollout(params: list, rng_input: jax.Array, steps_in_episode: int):
    """Rollout a jitted gymnax episode with lax.scan.

    Args:
        params (list): The model parameters.
        rng_input (jax.Array): The random number generator key.
        steps_in_episode (int): The number of steps in the episode.

    Returns:
        Transition: The output data.
    """
    # Reset the environment
    rng_reset, rng_episode = jr.split(rng_input)
    obs, state = ENV.reset(rng_reset, ENV_PARAMS)

    def step(state_input: Carry, _):
        """Step function for the environment.

        Args:
            state_input (Carry): The current state of the environment.
            _ (NoneType): Placeholder for the scan loop.

        Returns:
            Carry: The updated state of the environment.
            Transition: The output data of the current step.
        """
        rng, rng_action, rng_step = jr.split(state_input.rng, 3)
        action = get_action(params, state_input.obs, rng_action)
        next_obs, next_state, reward, done, _ = ENV.step(
            rng_step, state_input.state, action, ENV_PARAMS
        )
        carry = Carry(obs=next_obs, state=next_state, rng=rng)
        output = Transition(obs=state_input.obs, action=action, reward=reward, next_obs=next_obs, done=done)
        return carry, output

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        step,
        Carry(obs=obs, state=state, rng=rng_episode),
        (),
        length=steps_in_episode,
    )

    return scan_out


