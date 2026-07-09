"""Collect a small Push-T dataset locally for training the world model.

Uses gym-pusht at 64x64. Actions are a mix of uniform-random targets and
"push toward / through the block" targets, so trajectories contain lots of
contact/pushing dynamics for the world model to learn. Also renders a canonical
goal image (block placed at the goal pose).

Output: pusht_data.npz  with
    pixels  (N, 64, 64, 3) uint8      per-step frames
    actions (N, 2) float32            action taken at that step (target pos)
    states  (N, 5) float32            [agent_xy, block_xy, block_angle]
    ep_idx  (N,) int32                episode id
    coverage(N,) float32
    goal_image (64, 64, 3) uint8
"""

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gym_pusht  # noqa: F401 (registers the env)
import gymnasium as gym
import numpy as np


def make_env(size):
    return gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", observation_width=size, observation_height=size)


def scripted_push(agent, block, goal, rng):
    """Heuristic: get behind the block (opposite the goal), then push it toward
    the goal. Generates trajectories that actually approach the goal, so the
    world model sees near-goal states."""
    vec = goal[:2] - block[:2]
    dist = np.linalg.norm(vec)
    if dist < 1.0:
        return np.clip(block[:2] + rng.normal(0, 40, 2), 0, 512).astype(np.float32)
    u = vec / dist
    behind = block[:2] - u * 45.0
    if np.linalg.norm(agent - behind) > 45.0:
        target = behind                      # first, get behind the block
    else:
        target = block[:2] + u * 70.0        # then push through toward the goal
    return np.clip(target + rng.normal(0, 15, 2), 0, 512).astype(np.float32)


def goal_image(env, size, rng):
    # place the block exactly at the goal pose; agent parked away from it
    gx, gy, ga = 256, 256, np.pi / 4
    state = [rng.uniform(100, 400), rng.uniform(100, 400), gx, gy, ga]
    obs, _ = env.reset(options={"reset_to_state": state})
    return obs["pixels"].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pusht_data.npz")
    args = ap.parse_args()

    env = make_env(args.size)
    rng = np.random.default_rng(args.seed)

    gimg = goal_image(env, args.size, rng)

    px, ac, st, ep, cov = [], [], [], [], []
    for e in range(args.episodes):
        obs, info = env.reset(seed=args.seed + e)
        goal_pose = np.array([256.0, 256.0, np.pi / 4])
        for t in range(args.steps):
            block_xy = info["block_pose"][:2]
            agent_xy = info["pos_agent"]
            u = rng.random()
            if u < 0.55:                                  # scripted goal-directed push
                a = scripted_push(agent_xy, info["block_pose"], goal_pose, rng)
            elif u < 0.8:                                 # push toward/through block
                a = np.clip(block_xy + rng.normal(0, 60, size=2), 0, 512).astype(np.float32)
            else:                                         # uniform explore
                a = rng.uniform(0, 512, size=2).astype(np.float32)

            px.append(obs["pixels"].copy())
            ac.append(a)
            st.append(np.concatenate([info["pos_agent"], info["block_pose"]]).astype(np.float32))
            ep.append(e)
            cov.append(float(info.get("coverage", 0.0)))

            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        if (e + 1) % 25 == 0:
            print(f"  collected {e + 1}/{args.episodes} episodes, {len(px)} steps")
    env.close()

    np.savez_compressed(
        args.out,
        pixels=np.asarray(px, np.uint8), actions=np.asarray(ac, np.float32),
        states=np.asarray(st, np.float32), ep_idx=np.asarray(ep, np.int32),
        coverage=np.asarray(cov, np.float32), goal_image=gimg.astype(np.uint8),
    )
    print(f"saved {len(px)} transitions -> {args.out} "
          f"(mean coverage {np.mean(cov):.3f}, max {np.max(cov):.3f})")


if __name__ == "__main__":
    main()
