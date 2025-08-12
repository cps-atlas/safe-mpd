import os
import sys
import jax
import jax.numpy as jnp

# Ensure mbd is importable when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mbd
from mbd.envs import get_env


def run_random_rollout(num_trailers: int = 3, steps: int = 50, seed: int = 0):
    env = get_env(
        "n_trailer2d",
        case="parking",
        dt=0.2,
        H=steps,
        num_trailers=num_trailers,
        enable_gated_rollout_collision=False,
        enable_gated_rollout_hitch=False,
        enable_projection=False,
        enable_guidance=False,
        reward_threshold=25.0,
    )
    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    thetas_over_time = []
    xs = []
    for t in range(steps):
        key, k1 = jax.random.split(key)
        u = jax.random.uniform(k1, shape=(2,), minval=-0.5, maxval=0.5)
        state = env.step(state, u)
        x = state.pipeline_state
        xs.append(x)
        thetas = x[3: 3 + num_trailers]
        thetas_over_time.append(thetas)
    thetas_over_time = jnp.stack(thetas_over_time)  # [T, N]

    # Assert: each trailer angle varies over time
    var_time = jnp.var(thetas_over_time, axis=0)
    if not jnp.all(var_time > 1e-6):
        raise RuntimeError(f"Some trailer angles did not vary over time: var {var_time}")

    # Assert: final trailer angles are not all equal
    final_thetas = thetas_over_time[-1]
    if not (jnp.std(final_thetas) > 1e-6):
        raise RuntimeError(f"Final trailer angles are nearly identical: {final_thetas}")

    out_dir = os.path.join(mbd.__path__[0], "..", "results", "n_trailer2d", "debug")
    os.makedirs(out_dir, exist_ok=True)
    jnp.save(os.path.join(out_dir, "random_rollout_states.npy"), jnp.stack(xs))
    print("OK: angles vary and are not identical; saved rollout to:", out_dir)


if __name__ == "__main__":
    run_random_rollout(num_trailers=3, steps=60, seed=42)
