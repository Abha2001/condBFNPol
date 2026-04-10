"""
ExtendedHybridLunarLander – Large Discrete Action Space
=========================================================

Extended version of HybridLunarLander with 16 discrete actions instead of 4.
This creates a larger discrete action space to better showcase BFN's advantage
over diffusion methods (which struggle with large one-hot encodings).

Discrete actions (16 total):
  k=0   COAST           - No thrust (drift)
  k=1   MAIN_FULL       - Full upward thrust
  k=2   MAIN_HALF       - Half upward thrust
  k=3   MAIN_LOW        - Low upward thrust
  k=4   LEFT_STRONG     - Strong left thrust
  k=5   LEFT_MEDIUM     - Medium left thrust
  k=6   LEFT_WEAK       - Weak left thrust
  k=7   RIGHT_STRONG    - Strong right thrust
  k=8   RIGHT_MEDIUM    - Medium right thrust
  k=9   RIGHT_WEAK      - Weak right thrust
  k=10  UP_LEFT_STRONG  - Combined up + left (strong)
  k=11  UP_LEFT_WEAK    - Combined up + left (weak)
  k=12  UP_RIGHT_STRONG - Combined up + right (strong)
  k=13  UP_RIGHT_WEAK   - Combined up + right (weak)
  k=14  PULSE_LEFT      - Brief left pulse
  k=15  PULSE_RIGHT     - Brief right pulse

Each action has 1 continuous parameter for fine-tuning intensity.

For diffusion: 16 one-hot + 1 continuous = 17D action
For BFN: 1 discrete class + 1 continuous = much more efficient
"""

import importlib.metadata as _metadata

_original_entry_points = _metadata.entry_points


def _safe_entry_points(*args, **kwargs):
    eps = _original_entry_points(*args, **kwargs)
    if kwargs.get("group") != "gymnasium.envs":
        return eps
    filtered = [
        ep for ep in eps if getattr(ep, "module", "") != "gymnasium_robotics.__init__"
    ]
    try:
        return type(eps)(filtered)
    except Exception:
        return filtered


_metadata.entry_points = _safe_entry_points

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Tuple


# ─── Action constants ────────────────────────────────────────────────────────

COAST = 0
MAIN_FULL = 1
MAIN_HALF = 2
MAIN_LOW = 3
LEFT_STRONG = 4
LEFT_MEDIUM = 5
LEFT_WEAK = 6
RIGHT_STRONG = 7
RIGHT_MEDIUM = 8
RIGHT_WEAK = 9
UP_LEFT_STRONG = 10
UP_LEFT_WEAK = 11
UP_RIGHT_STRONG = 12
UP_RIGHT_WEAK = 13
PULSE_LEFT = 14
PULSE_RIGHT = 15

NUM_DISCRETE = 16

# All actions have 1 continuous parameter for intensity fine-tuning
MAX_PARAM_DIM = 1

# Action names for logging
ACTION_NAMES = {
    COAST: "coast",
    MAIN_FULL: "main_full",
    MAIN_HALF: "main_half",
    MAIN_LOW: "main_low",
    LEFT_STRONG: "left_strong",
    LEFT_MEDIUM: "left_medium",
    LEFT_WEAK: "left_weak",
    RIGHT_STRONG: "right_strong",
    RIGHT_MEDIUM: "right_medium",
    RIGHT_WEAK: "right_weak",
    UP_LEFT_STRONG: "up_left_strong",
    UP_LEFT_WEAK: "up_left_weak",
    UP_RIGHT_STRONG: "up_right_strong",
    UP_RIGHT_WEAK: "up_right_weak",
    PULSE_LEFT: "pulse_left",
    PULSE_RIGHT: "pulse_right",
}

# Base thrust configurations for each action
# Format: (base_main, base_side, main_range, side_range)
# The continuous param modulates within the range
ACTION_CONFIG = {
    COAST:           (0.0, 0.0, 0.0, 0.0),      # No thrust
    MAIN_FULL:       (1.0, 0.0, 0.2, 0.0),      # Full up, modulate 0.8-1.0
    MAIN_HALF:       (0.5, 0.0, 0.2, 0.0),      # Half up, modulate 0.3-0.7
    MAIN_LOW:        (0.2, 0.0, 0.15, 0.0),     # Low up, modulate 0.05-0.35
    LEFT_STRONG:     (0.0, -1.0, 0.0, 0.2),     # Strong left
    LEFT_MEDIUM:     (0.0, -0.7, 0.0, 0.15),    # Medium left
    LEFT_WEAK:       (0.0, -0.55, 0.0, 0.05),   # Weak left (just above threshold)
    RIGHT_STRONG:    (0.0, 1.0, 0.0, 0.2),      # Strong right
    RIGHT_MEDIUM:    (0.0, 0.7, 0.0, 0.15),     # Medium right
    RIGHT_WEAK:      (0.0, 0.55, 0.0, 0.05),    # Weak right
    UP_LEFT_STRONG:  (0.7, -0.8, 0.15, 0.1),    # Up + left strong
    UP_LEFT_WEAK:    (0.3, -0.6, 0.1, 0.1),     # Up + left weak
    UP_RIGHT_STRONG: (0.7, 0.8, 0.15, 0.1),     # Up + right strong
    UP_RIGHT_WEAK:   (0.3, 0.6, 0.1, 0.1),      # Up + right weak
    PULSE_LEFT:      (0.0, -0.6, 0.0, 0.1),     # Left pulse
    PULSE_RIGHT:     (0.0, 0.6, 0.0, 0.1),      # Right pulse
}


class ExtendedHybridLunarLander(gym.Env):
    """
    Extended parameterized-action-space wrapper with 16 discrete actions.

    Action format:
        action = {
            "k":   int in {0, 1, ..., 15},
            "x_k": ndarray of shape (1,) in [-1, 1]
        }
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        use_image_obs: bool = False,
        img_size: tuple = (84, 84),
    ):
        super().__init__()

        self.use_image_obs = use_image_obs
        self.img_size = img_size

        self._env = self._make_env(render_mode="rgb_array")
        self.render_mode = "rgb_array"

        # Observation space
        state_space = self._env.observation_space  # Box(8,)

        if self.use_image_obs:
            H, W = self.img_size
            self.observation_space = spaces.Dict({
                "state": state_space,
                "image": spaces.Box(0, 255, shape=(H, W, 3), dtype=np.uint8),
            })
        else:
            self.observation_space = state_space

        # Parameterized action space
        self.action_space = spaces.Dict({
            "k":   spaces.Discrete(NUM_DISCRETE),
            "x_k": spaces.Box(-1.0, 1.0, shape=(MAX_PARAM_DIM,), dtype=np.float32),
        })

        # For compatibility with HyAR/BFN
        self.param_dims = {k: 1 for k in range(NUM_DISCRETE)}
        self.param_dims[COAST] = 0  # COAST has no params
        self.num_discrete = NUM_DISCRETE
        self.max_param_dim = MAX_PARAM_DIM

    @staticmethod
    def _make_env(render_mode: str) -> gym.Env:
        candidates = ["LunarLanderContinuous-v3", "LunarLanderContinuous-v2"]
        last_err = None
        for env_id in candidates:
            try:
                return gym.make(env_id, render_mode=render_mode)
            except gym.error.DependencyNotInstalled as e:
                raise RuntimeError(
                    "Box2D missing. Install: pip install 'gymnasium[box2d]'"
                ) from e
            except Exception as e:
                last_err = e
        raise RuntimeError(
            f"Could not create LunarLanderContinuous. Tried {candidates}. "
            f"Last error: {last_err}"
        )

    def _convert_action(self, k: int, x_k: np.ndarray) -> np.ndarray:
        """
        Map parameterized action (k, x_k) → [main_throttle, side_throttle]
        """
        base_main, base_side, main_range, side_range = ACTION_CONFIG[k]

        if k == COAST:
            return np.array([0.0, 0.0], dtype=np.float32)

        # x_k[0] ∈ [-1, 1] modulates the intensity
        mod = float(np.clip(x_k[0], -1.0, 1.0))

        # Apply modulation to base values
        main = base_main + main_range * mod
        side = base_side + side_range * mod * np.sign(base_side) if base_side != 0 else 0.0

        # Clip to valid ranges
        main = float(np.clip(main, 0.0, 1.0))
        side = float(np.clip(side, -1.0, 1.0))

        return np.array([main, side], dtype=np.float32)

    def _get_image(self) -> np.ndarray:
        frame = self._env.render()
        H, W = self.img_size
        nH, nW = frame.shape[:2]
        if (nH, nW) != (H, W):
            row_idx = (np.arange(H) * nH / H).astype(int)
            col_idx = (np.arange(W) * nW / W).astype(int)
            frame = frame[np.ix_(row_idx, col_idx)]
        return frame.astype(np.uint8)

    def _build_obs(self, state_obs: np.ndarray):
        if self.use_image_obs:
            return {"state": state_obs, "image": self._get_image()}
        return state_obs

    def reset(self, *, seed=None, options=None):
        state_obs, info = self._env.reset(seed=seed, options=options)
        return self._build_obs(state_obs), info

    def step(self, action):
        k = int(action["k"])
        x_k = np.asarray(action["x_k"], dtype=np.float32)

        cont_action = self._convert_action(k, x_k)
        state_obs, reward, terminated, truncated, info = self._env.step(cont_action)

        info["discrete_action"] = k
        info["action_name"] = ACTION_NAMES[k]

        return self._build_obs(state_obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    def get_action_spec(self) -> dict:
        return {
            "num_discrete": self.num_discrete,
            "param_dims": dict(self.param_dims),
            "max_param_dim": self.max_param_dim,
            "action_names": dict(ACTION_NAMES),
        }


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ExtendedHybridLunarLander – 16 Discrete Actions")
    print("=" * 60)

    env = ExtendedHybridLunarLander(use_image_obs=False)

    spec = env.get_action_spec()
    print(f"\nAction spec:")
    print(f"  Discrete actions : {spec['num_discrete']}")
    print(f"  Max param dim    : {spec['max_param_dim']}")
    print(f"\n  Actions:")
    for k, name in spec["action_names"].items():
        config = ACTION_CONFIG[k]
        print(f"    k={k:2d}  {name:18s}  base=(main={config[0]:.1f}, side={config[1]:+.1f})")

    print(f"\nAction space : {env.action_space}")
    print(f"Obs space    : {env.observation_space}")

    # Test episode
    print("\nRunning random episode...")
    obs, info = env.reset(seed=42)

    action_counts = {k: 0 for k in range(NUM_DISCRETE)}
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_counts[info["discrete_action"]] += 1
        total_reward += reward
        steps += 1

    print(f"  Steps: {steps}  Total reward: {total_reward:+.1f}")
    print(f"\n  Action distribution:")
    for k, name in spec["action_names"].items():
        pct = 100 * action_counts[k] / steps if steps > 0 else 0
        print(f"    {name:18s}: {action_counts[k]:3d} ({pct:4.1f}%)")

    env.close()
    print("\nAll tests passed.")
