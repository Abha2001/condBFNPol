"""
Lunar Lander Dataset for BFN/Diffusion Policy training.

This dataset handles the hybrid action space of the Lunar Lander environment:
- Discrete action (k): Integer in {0, 1, 2, 3}
  - 0 = COAST, 1 = MAIN_ENGINE, 2 = LEFT_BOOST, 3 = RIGHT_BOOST
- Continuous parameters (x_k): Vector in [-1, 1]² (padded)

Two action encoding modes are supported:
1. "hybrid" - Native hybrid: discrete as class index, continuous as values
2. "continuous" - All actions as continuous: one-hot discrete + continuous params

Supports two zarr formats:
1. Zhaoyang's format: data/{state, action, hybrid_action_discrete, reward}
2. Converted format: data/{state, discrete_action, continuous_params, reward}
"""

from typing import Dict, Optional
import torch
import numpy as np
import copy
import zarr
from pathlib import Path

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


# Action space constants (must match environments/lunar_lander.py)
NUM_DISCRETE = 4
MAX_PARAM_DIM = 2


class LunarLanderReplayBuffer:
    """Simple replay buffer for Lunar Lander data stored in zarr format.

    Supports both:
    - Zhaoyang's format: action (continuous), hybrid_action_discrete, state
    - Converted format: discrete_action, continuous_params, state
    """

    def __init__(self, zarr_path: str):
        self.root = zarr.open(str(zarr_path), mode='r')
        self.data = self.root['data']

        # Check for meta/episode_ends
        if 'meta' in self.root and 'episode_ends' in self.root['meta']:
            self.meta = self.root['meta']
            self.episode_ends = np.array(self.meta['episode_ends'])
        elif 'episode_ends' in self.data:
            # Alternative location
            self.episode_ends = np.array(self.data['episode_ends'])
        else:
            # Infer from data if not available (assume single long episode)
            total_len = len(np.array(self.data[list(self.data.keys())[0]]))
            self.episode_ends = np.array([total_len])

        self.n_episodes = len(self.episode_ends)

        # Compute episode starts
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        # Detect format
        self.keys = list(self.data.keys())
        self.zhaoyang_format = 'hybrid_action_discrete' in self.keys

        print(f"LunarLanderReplayBuffer: detected {'Zhaoyang' if self.zhaoyang_format else 'converted'} format")
        print(f"  Keys: {self.keys}")
        print(f"  Episodes: {self.n_episodes}")

    def __len__(self):
        return self.episode_ends[-1] if len(self.episode_ends) > 0 else 0

    def __getitem__(self, key):
        # Handle key mapping for compatibility
        if self.zhaoyang_format:
            if key == 'discrete_action':
                key = 'hybrid_action_discrete'
            elif key == 'continuous_params':
                # Zhaoyang's format uses hybrid_action_params for continuous params
                key = 'hybrid_action_params' if 'hybrid_action_params' in self.keys else 'action'
        return np.array(self.data[key])

    def get_episode(self, idx: int, include_images: bool = False) -> Dict[str, np.ndarray]:
        """Get data for a single episode.

        Args:
            idx: Episode index
            include_images: If True, include 'img' data (slow for large images)
        """
        start = self.episode_starts[idx]
        end = self.episode_ends[idx]

        result = {}
        for key in self.data.keys():
            if key == 'episode_ends':
                continue
            if key == 'img' and not include_images:
                continue
            result[key] = np.array(self.data[key][start:end])

        # Map keys for compatibility with our expected format
        if self.zhaoyang_format:
            if 'hybrid_action_discrete' in result:
                result['discrete_action'] = result['hybrid_action_discrete']
            # Continuous params: prefer hybrid_action_params, fallback to action
            if 'hybrid_action_params' in result:
                result['continuous_params'] = result['hybrid_action_params']
            elif 'action' in result and 'continuous_params' not in result:
                result['continuous_params'] = result['action']

        return result

    def get_episode_length(self, idx: int) -> int:
        """Get length of an episode."""
        return self.episode_ends[idx] - self.episode_starts[idx]


class LunarLanderSequenceSampler:
    """Sequence sampler for Lunar Lander data."""

    def __init__(
        self,
        replay_buffer: LunarLanderReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        episode_mask: Optional[np.ndarray] = None,
        include_images: bool = False,
    ):
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.include_images = include_images

        if episode_mask is None:
            episode_mask = np.ones(replay_buffer.n_episodes, dtype=bool)
        self.episode_mask = episode_mask

        # Build index map: sample_idx -> (episode_idx, start_idx)
        self.indices = []
        for ep_idx in range(replay_buffer.n_episodes):
            if not episode_mask[ep_idx]:
                continue

            ep_len = replay_buffer.get_episode_length(ep_idx)
            # Allow starting positions that can form valid sequences
            # with padding
            for start in range(ep_len):
                self.indices.append((ep_idx, start))

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        """Sample a sequence of length sequence_length."""
        ep_idx, start_idx = self.indices[idx]
        episode = self.replay_buffer.get_episode(ep_idx, include_images=self.include_images)
        ep_len = len(episode['state'])

        result = {}
        for key, data in episode.items():
            # Handle padding
            seq = []
            for i in range(start_idx - self.pad_before,
                          start_idx + self.sequence_length - self.pad_before):
                if i < 0:
                    seq.append(data[0])  # Pad with first element
                elif i >= ep_len:
                    seq.append(data[-1])  # Pad with last element
                else:
                    seq.append(data[i])
            result[key] = np.stack(seq, axis=0)

        return result


class LunarLanderDataset(BaseLowdimDataset):
    """
    Lunar Lander dataset for BFN/Diffusion Policy training.

    Args:
        zarr_path: Path to zarr dataset
        horizon: Sequence length for training
        pad_before: Number of steps to pad before sequence
        pad_after: Number of steps to pad after sequence
        action_mode: "hybrid" for native discrete+continuous,
                     "continuous" for all continuous encoding
        seed: Random seed for train/val split
        val_ratio: Fraction of episodes for validation
        max_train_episodes: Maximum number of training episodes
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        action_mode: str = "hybrid",  # "hybrid" or "continuous"
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()

        self.zarr_path = zarr_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.action_mode = action_mode

        # Load replay buffer
        self.replay_buffer = LunarLanderReplayBuffer(zarr_path)

        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.train_mask = train_mask
        self.sampler = LunarLanderSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        # Compute action dimensions based on mode
        if action_mode == "hybrid":
            # Discrete: 1 (class index), Continuous: 2
            self.discrete_dim = 1
            self.continuous_dim = MAX_PARAM_DIM
            self.action_dim = self.discrete_dim + self.continuous_dim  # 3
        else:  # continuous
            # One-hot discrete (4) + continuous params (2) = 6
            self.action_dim = NUM_DISCRETE + MAX_PARAM_DIM  # 6

        print(f"LunarLanderDataset initialized:")
        print(f"  Episodes: {self.replay_buffer.n_episodes}")
        print(f"  Train episodes: {train_mask.sum()}")
        print(f"  Action mode: {action_mode}")
        print(f"  Action dim: {self.action_dim}")

    def get_validation_dataset(self):
        """Get validation dataset."""
        val_set = copy.copy(self)
        val_set.sampler = LunarLanderSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """Get normalizer for the dataset."""
        # Sample all data for normalization
        all_data = self._get_all_data()

        normalizer = LinearNormalizer()
        normalizer.fit(data=all_data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def _get_all_data(self) -> Dict[str, torch.Tensor]:
        """Get all data for normalization.

        Returns dict with 'state' and 'action' keys to match the structure
        used in batch['obs'] = {'state': tensor} during training.
        """
        all_obs = []
        all_actions = []

        for ep_idx in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(ep_idx)
            obs, action = self._process_episode(episode)
            all_obs.append(obs)
            all_actions.append(action)

        # Return structure matching batch['obs'] = {'state': ...}
        # Normalizer will be fit with 'state' and 'action' keys
        return {
            'state': torch.from_numpy(np.concatenate(all_obs, axis=0)),
            'action': torch.from_numpy(np.concatenate(all_actions, axis=0))
        }

    def _process_episode(self, episode: Dict[str, np.ndarray]):
        """Process episode data into obs and action tensors."""
        # State observation: 8D
        state = episode['state'].astype(np.float32)  # [T, 8]

        # Action processing depends on mode
        discrete_k = episode['discrete_action'].astype(np.int64)  # [T]
        continuous_x = episode['continuous_params'].astype(np.float32)  # [T, 2]

        if self.action_mode == "hybrid":
            # [discrete_class, continuous_param_0, continuous_param_1]
            action = np.concatenate([
                discrete_k[:, None].astype(np.float32),
                continuous_x
            ], axis=-1)  # [T, 3]
        else:
            # One-hot encode discrete + continuous params
            one_hot = np.zeros((len(discrete_k), NUM_DISCRETE), dtype=np.float32)
            one_hot[np.arange(len(discrete_k)), discrete_k] = 1.0
            action = np.concatenate([one_hot, continuous_x], axis=-1)  # [T, 6]

        return state, action

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # Process the sample
        state = sample['state'].astype(np.float32)  # [T, 8]
        discrete_k = sample['discrete_action'].astype(np.int64)  # [T]
        continuous_x = sample['continuous_params'].astype(np.float32)  # [T, 2]

        if self.action_mode == "hybrid":
            action = np.concatenate([
                discrete_k[:, None].astype(np.float32),
                continuous_x
            ], axis=-1)  # [T, 3]
        else:
            one_hot = np.zeros((len(discrete_k), NUM_DISCRETE), dtype=np.float32)
            one_hot[np.arange(len(discrete_k)), discrete_k] = 1.0
            action = np.concatenate([one_hot, continuous_x], axis=-1)  # [T, 6]

        data = {
            'obs': {
                'state': state  # [T, 8]
            },
            'action': action  # [T, action_dim]
        }

        return dict_apply(data, torch.from_numpy)

    def get_all_actions(self) -> torch.Tensor:
        """Get all actions in the dataset."""
        all_actions = []
        for ep_idx in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(ep_idx)
            _, action = self._process_episode(episode)
            all_actions.append(action)
        return torch.from_numpy(np.concatenate(all_actions, axis=0))


class LunarLanderHybridDataset(LunarLanderDataset):
    """Convenience class for hybrid action mode."""

    def __init__(self, zarr_path: str, **kwargs):
        kwargs['action_mode'] = 'hybrid'
        super().__init__(zarr_path, **kwargs)


class LunarLanderContinuousDataset(LunarLanderDataset):
    """Convenience class for continuous action mode."""

    def __init__(self, zarr_path: str, **kwargs):
        kwargs['action_mode'] = 'continuous'
        super().__init__(zarr_path, **kwargs)


class LunarLanderImageDataset(BaseLowdimDataset):
    """
    Lunar Lander dataset with IMAGE observations.

    Supports different observation and action configurations:
    - obs_keys: ['img'] or ['img', 'state']
    - action_mode: 'hybrid' for [discrete_class, params] (3D) - BFN native discrete
                   'continuous' for hybrid_action_flat (5D) - DDPM continuous
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        obs_keys: list = ['img'],
        action_mode: str = 'continuous',  # 'hybrid' or 'continuous'
        action_key: str = None,  # Deprecated, use action_mode instead
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()

        self.zarr_path = zarr_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_keys = obs_keys
        self.action_mode = action_mode

        # Handle legacy action_key parameter
        if action_key is not None:
            if action_key == 'hybrid_action_flat':
                self.action_mode = 'continuous'
            else:
                self.action_mode = action_mode

        # Load replay buffer
        self.replay_buffer = LunarLanderReplayBuffer(zarr_path)

        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.train_mask = train_mask
        self.sampler = LunarLanderSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            include_images=('img' in obs_keys)
        )

        # Set action dimension based on mode
        if self.action_mode == 'hybrid':
            # [discrete_class, param_0, param_1] = 3D
            self.action_dim = 1 + MAX_PARAM_DIM  # 3
        else:  # continuous
            # hybrid_action_flat = 5D
            self.action_dim = 5

        print(f"LunarLanderImageDataset initialized:")
        print(f"  Episodes: {self.replay_buffer.n_episodes}")
        print(f"  Train episodes: {train_mask.sum()}")
        print(f"  Obs keys: {obs_keys}")
        print(f"  Action mode: {self.action_mode} (dim={self.action_dim})")

    def get_validation_dataset(self):
        """Get validation dataset."""
        val_set = copy.copy(self)
        val_set.sampler = LunarLanderSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            include_images=('img' in self.obs_keys)
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """Get normalizer for the dataset."""
        all_data = self._get_all_data()
        normalizer = LinearNormalizer()
        normalizer.fit(data=all_data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def _get_all_data(self) -> Dict[str, torch.Tensor]:
        """Get all data for normalization."""
        all_actions = []

        for ep_idx in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(ep_idx)
            action = self._process_action(episode)
            all_actions.append(action)

        # For images, we typically don't normalize (already in [0,1])
        # Just return action for normalization
        return {
            'action': torch.from_numpy(np.concatenate(all_actions, axis=0))
        }

    def _process_action(self, episode: Dict[str, np.ndarray]) -> np.ndarray:
        """Process action based on action_mode.

        Uses hybrid_action_discrete and hybrid_action_flat as specified:
        - hybrid mode: [discrete_class, last 2 dims of hybrid_action_flat] = 3D
        - continuous mode: hybrid_action_flat = 5D
        """
        if self.action_mode == 'hybrid':
            # [discrete_class, param_0, param_1] = 3D
            # Discrete from hybrid_action_discrete (mapped to discrete_action)
            discrete_k = episode['discrete_action'].astype(np.float32)
            # Continuous params from last 2 dims of hybrid_action_flat
            hybrid_flat = episode['hybrid_action_flat'].astype(np.float32)
            continuous_x = hybrid_flat[:, -2:]  # Last 2 dims are the continuous params
            action = np.concatenate([
                discrete_k[:, None],
                continuous_x
            ], axis=-1)
        else:  # continuous
            # Use hybrid_action_flat (5D) directly
            action = episode['hybrid_action_flat'].astype(np.float32)
        return action

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # Build observation dict
        obs_dict = {}
        for key in self.obs_keys:
            if key == 'img':
                # Image: [T, H, W, C] -> [T, C, H, W]
                img = sample['img'].astype(np.float32)
                img = np.transpose(img, (0, 3, 1, 2))  # NHWC -> NCHW
                obs_dict['img'] = img
            elif key == 'state':
                obs_dict['state'] = sample['state'].astype(np.float32)

        # Get action based on action_mode
        action = self._process_action(sample)

        data = {
            'obs': obs_dict,
            'action': action
        }

        return dict_apply(data, torch.from_numpy)

    def get_all_actions(self) -> torch.Tensor:
        """Get all actions in the dataset."""
        all_actions = []
        for ep_idx in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(ep_idx)
            action = self._process_action(episode)
            all_actions.append(action)
        return torch.from_numpy(np.concatenate(all_actions, axis=0))
