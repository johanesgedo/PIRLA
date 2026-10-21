# Execution for Physics-Informed Reinforcement Learning Agent (PIRLA)
# Copyright (C) 2026 Johanes Gedo Sea
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations

import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

import pirla

try:
    import cupy as cp
except Exception:
    cp = None


NUM_ACTIONS = 10
STEPS = 1
LOSS_VAL = 0.0
EPOCH = 8
BATCH_SIZE = 128
TRAIN_EVERY = 1
SEED = 42
HIDDEN_DIM = 256
BUFFER_CAPACITY = 50000
EVAL_EPISODES = 100
N_SAMPLES = 25
OUTPUT_DIR = './outputs_pirl'


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _as_numpy(x):
    if cp is not None:
        try:
            return cp.asnumpy(x)
        except Exception:
            pass
    return np.asarray(x)

def _as_scalar(x) -> float:
    arr = _as_numpy(x).astype(np.float64, copy=False)
    return float(arr.reshape(-1)[0])

def _to_backend(x):
    if cp is not None:
        try:
            return cp.asarray(x)
        except Exception:
            pass
    return np.asarray(x)

def _replay_size(replay) -> int:
    if replay is None:
        return 0
    for name in ('Valid_Size', 'size', '__len__'):
        try:
            fn = getattr(replay, name)
            return int(fn()) if callable(fn) else int(fn)
        except Exception:
            pass
    return 0

"""
def _replay_size(replay) -> int:
    if replay is None:
        return 0
    if hasattr(replay, "Valid_Size") and callable(replay.Valid_Size):
        try:
            return int(replay.Valid_Size())
        except Exception:
            pass
    if hasattr(replay, "size"):
        try:
            return int(replay.size)
        except Exception:
            pass
    if hasattr(replay, "__len__"):
        try:
            return int(len(replay))
        except Exception:
            pass
    return 0
"""

def _sample_to_batch(sample):
    if isinstance(sample, dict):
        return sample
    if isinstance(sample, (tuple, list)) and len(sample) >= 6:
        states, actions, rewards, next_states, next_actions, indices = sample[:6]
        return {
            "states": _to_backend(states),
            "actions": _to_backend(actions),
            "rewards": _to_backend(rewards),
            "next_states": _to_backend(next_states),
            "next_actions": _to_backend(next_actions),
            "indices": _to_backend(indices),
            "idx": _to_backend(indices),
        }
    raise TypeError(f"Unsupported replay sample format: {type(sample)!r}")


class OneStepImageClassificationEnv:
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        seed: int = 42,
        download: bool = True,
    ):
        self.root = root
        self.train = bool(train)
        self.seed = int(seed)
        self.download = bool(download)
        self.rng = np.random.default_rng(seed)
        self.dataset_name = "MNIST"
        self.num_actions = NUM_ACTIONS
        self.state_dim = None
        self.observation_shape = None
        self._load_dataset()
        self._build_order()
        self.cursor = 0

    def _load_dataset(self) -> None:
        transform = transforms.Compose([transforms.ToTensor()])
        try:
            self.dataset = datasets.MNIST(
                root=self.root,
                train=self.train,
                download=self.download,
                transform=transform,
            )
            self.observation_shape = (28, 28)
            self.num_actions = NUM_ACTIONS
            self.state_dim = 28 * 28
            self.dataset_name = "MNIST"
        except Exception as e:
            print(f"[INFO] MNIST unavailable, fallback to sklearn digits: {e}")
            X, y = load_digits(return_X_y=True)
            X = (X / 16.0).astype(np.float32).reshape(-1, 8, 8)
            self.dataset = list(zip(X, y.astype(np.int64)))
            self.observation_shape = (8, 8)
            self.num_actions = NUM_ACTIONS
            self.state_dim = 8 * 8
            self.dataset_name = "sklearn-digits"

    """
    def _build_order(self) -> None:
        n = len(self.dataset)
        if self.train:
            sef.order = self.rng.permutation(n)
        else:
            self.order = np.arange(n, dtype=np.int64)
    """

    def _build_order(self) -> None:
        n = len(self.dataset)
        self.order = self.rng.permutation(n) if self.train else np.arange(n, dtype=np.int64)

    def _get_item(self, idx: int):
        item = self.dataset[int(idx)]
        if self.dataset_name == "MNIST":
            image, label = item
            x = image.squeeze(0).numpy().astype(np.float32, copy=False)
            return x, int(label)
        x, label = item
        return np.asarray(x, dtype=np.float32), int(label)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
            self._build_order()
        self.cursor = int(self.cursor % len(self.order))
        idx = int(self.order[self.cursor])
        obs, label = self._get_item(idx)
        info = {"index": idx, "label": label, "dataset": self.dataset_name}
        return obs, info

    def step(self, action):
        idx = int(self.order[self.cursor])
        _, label = self._get_item(idx)
        if hasattr(action, "item"):
            action = action.item()
        action = int(np.asarray(action).reshape(-1)[0])
        reward = 1.0 if action == label else 0.0
        terminated = True
        truncated = False
        info = {
            'index': idx,
            'label': label,
            'action': action,
            'correct': bool(action == label),
            'dataset': self.dataset_name,
        }
        self.cursor = (self.cursor + 1) % len(self.order)
        next_idx = int(self.order[self.cursor])
        next_obs, next_label = self._get_item(next_idx)
        info["next_index"] = next_idx
        info["next_label"] = next_label
        return next_obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None


def patch_engine_api(engine):
    aliases = {
        "configure_from_env": "Configure_From_Environment",
        "sync_dimension": "Sync_Dimensions",
        "reset_env": "Environment_Reset",
        "step_env": "Environment_Step",
        "train": "Training",
        "state_dict": "State_Dict",
        "load_state_dict": "Load_State_Dict",
        "save": "Save",
        "load": "Load",
    }
    for alias, target in aliases.items():
        if not hasattr(engine, alias) and hasattr(engine, target):
            setattr(engine, alias, getattr(engine, target))
    return engine

def build_engine(
    env,
    seed: int,
    hidden_dim: int,
    buffer_capacity: int
):
    engine = pirla.Physics_Informed_Reinforcement_Learning(
        env=env,
        vec_env=None,
        backbone=None,
        autodiff=None,
        pinn=None,
        rl=None,
        num_actions=NUM_ACTIONS,
        num_states=None,
        state_dim=env.state_dim,
        hidden_dim=hidden_dim,
        buffer_capacity=buffer_capacity,
        seed=seed,
        normalize_states=False,
        normalize_rewards=False,
    )
    patch_engine_api(engine)
    if hasattr(engine, 'Configure_From_Environment'):
        try:
            engine.Configure_From_Environment(env)
        except Exception as e:
            print(f"[WARN] Configure_From_Environment failed; continuing with manual dimensions: {e}")
    return engine

def _reset_env(env):
    out = env.reset()
    return out if isinstance(out, tuple) and len(out) == 2 else (out, {})

def _step_env(
    env,
    action
):
    out = env.step(action)
    if len(out) == 5:
        return out
    if len(out) == 4:
        next_state, reward, done, info = out
        return next_state, reward, done, False, info
    raise ValueError('Unexpected environment step output.')

def evaluate_policy(
    engine,
    env,
    episodes: int = EVAL_EPISODES
) -> Dict[str, float]:
    correct = 0
    total = 0
    total_reward = 0.0
    for _ in range(int(episodes)):
        obs, _info = _reset_env(env)
        action = engine.Select_Action(obs, strategy='auto', deterministic=True)
        _next_obs, reward, terminated, truncated, _step_info = _step_env(env, action)
        total_reward += float(reward)
        total += 1
        correct += int(float(reward) > 0.5)
    return {
        "accuracy": correct / max(total,1),
        "mean_reward": total_reward / max(total, 1),
    }

def moving_averange(
    values: List[float],
    window: int = 10
) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) < window:
        return np.cumsum(arr) / np.arange(1, len(arr) + 1)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode='valid')

def plot_metric(
    values: List[float],
    title: str,
    ylabel: str,
    out_path: str,
    smooth_window: int = 10
):
    plt.figure(figsize=(8, 4.5))
    x = np.arange(1, len(values) + 1)
    plt.plot(x, values, label='raw')
    if len(values) >= 2:
        sm = moving_averange(values, window=min(smooth_window, max(2, len(values))))
        xs = np.arange(len(sm)) + (len(values) - len(sm) + 1)
        plt.plot(xs, sm, label=f'moving_avg_{min(smooth_window, len(values))}')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def visualize_prediction(
    engine,
    env,
    n_samples=N_SAMPLES,
    out_path="outputs_pirl/predictions.png"
):
    fig, axes = plt.subplots(5, 5, figsize=(10,10))
    flat_axes = axes.flat
    for i, ax in enumerate(flat_axes):
        if i >= n_samples:
            ax.axis("off")
            continue
        obs, info = env.reset()
        action = engine.Select_Action(obs, strategy="auto", deterministic=True)
        if hasattr(action, "item"):
            action = action.item()
        ax.imshow(obs, cmap="gray")
        ax.set_title(
            f"T:{info['label']} P:{action}",
            fontsize=9,
            color="green" if action == info["label"] else "red",
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_confusion_matrix(
    engine,
    env,
    out_path: str
):
    y_true = []
    y_pred = []
    for _ in range(len(env.order)):
        obs, info = env.reset()
        pred = engine.Select_Action(
            obs,
            strategy="auto",
            deterministic=True
        )
        if hasattr(pred, "item"):
            pred = pred.item()
        y_true.append(info["label"])
        y_pred.append(pred)
    cm=confusion_matrix(y_tru, y_pred)
    plt.figure(figsize=(8,8))
    plt.imshow(cm)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(out_path, dpi=400)
    plt.close()

"""
def Show_MNIST_Samples(
    dataset,
    n=25
):
    plt.figure(figsize=(8,8))
    n = min(n, len(dataset))
    for i in range(n):
        img, label = dataset[i]
        plt.subplot(5, 5, i + 1)
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.title(str(label))
        plt.axis("off")
    plt.tight_layout
    plt.show()
"""

def Show_MNIST_Samples(
    dataset,
    n_samples: int = 25,
    random_sample: bool = True,
    figsize=(10,10),
    cmap="gray",
    save_path=None,
):
    total = len(dataset)
    n_samples = min(n_samples, total)
    if random_sample:
        indices = np.random.choice(total, n_samples, replace=False)
    else:
        indices = np.arange(n_samples)
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    plt.figure(figsize=figsize)
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        # MNIST Tensor (1, H, W)
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        plt.subplot(rows, cols, i + 1)
        if image.ndim == 2:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.title(f"Label : {label}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400)
    plt.show()

def Show_Prediction_Samples(
    engine,
    dataset,
    n_samples: int = 25,
    random_sample: bool = True,
    figsize=(10, 10),
    save_path: str | None = None,
):
    total = len(dataset)
    n_samples = min(int(n_samples), total)
    if random_sample:
        indices = np.random.choice(total, n_samples, replace=False)
    else:
        indices = np.arange(n_samples)
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    plt.figure(figsize=figsize)
    for i, idx in enumerate(indices):
        image, label = dataset[int(idx)]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        pred = engine.Select_Action(image, strategy="auto", deterministic=True)
        pred = _scalar_action(pred)
        is_correct = int(pred) == int(label)
        color = "green" if is_correct else "red"
        plt.subplot(rows, cols, i+1)
        if image.ndim == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        plt.title(f"T:{int(label)} P:{int(pred)}", fontsize=9, color=color)
        plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=500)
    plt.show()

"""
def run_launcher(
    epoch: int = EPOCH,
    batch_size: int = BATCH_SIZE,
    train_every: int = TRAIN_EVERY,
    seed: int = SEED,
    hidden_dim: int = HIDDEN_DIM,
    buffer_capacity: int = BUFFER_CAPACITY,
    checkpoint_dir: str | None = None,
    output_dir: str = OUTPUT_DIR
):
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    train_env = OneStepImageClassificationEnv(train=True, seed=seed, download=True)
    Show_MNIST_Samples(
        train_env.dataset,
        n_samples=49,
        random_sample=False
    )
    eval_env = OneStepImageClassificationEnv(train=False, seed=seed, download=True)
    engine = build_engine(
        env=train_env,
        seed=seed,
        hidden_dim=hidden_dim,
        buffer_capacity=buffer_capacity,
    )
    print(f"[INFO] Dataset: {train_env.dataset_name}")
    print(f"[INFO] Input shape: {train_env.observation_shape}")
    print(f"[INFO] Actions: {train_env.num_actions}")
    print(f"[INFO] State dim: {train_env.state_dim}")
    print(f"[INFO] Engine: {engine.__class__.__name__}")
    print("[INFO] Training_PIRL.py acts only as a launcher; the PIRLA engine performs the learning.")

    history: Dict[str, List[float]] = {
        'episode_reward': [],
        'episode_length': [],
        'train_loss': [],
        'physics_loss': [],
        'policy_loss': [],
        'entropy': [],
        'eval_accuracy': [],
        'eval_reward': [],
        'replay_size': [],
        'mean_q': [],
        'temperature': [],
        'epsilon': [],
    }

    for ep in range(int(epoch)):
        obs, _info = _reset_env(train_env)
        ep_reward = 0.0
        ep_steps = 0
        last_loss = np.nan
        for _t in range(int(STEPS)):
            action = engine.Select_Action(obs, strategy='auto')
            next_obs, reward, terminated, truncated, _step_info = _step_env(train_env, action)
            done = bool(terminated or truncated)
            next_action = engine.Select_Action(next_obs, strategy='auto')
            engine.Store_Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                next_action=next_action,
                done=done,
            )
            ep_reward += float(reward)
            ep_steps += 1
            obs = next_obs
            if done:
                break
        replay = getattr(engine, 'replay', None)
        replay_size = _replay_size(replay)
        if replay_size >= int(batch_size) and ((ep + 1) % int(train_every) == 0):
            sampled = engine.Sample_Batch(batch_size, prioritized=getattr(engine, 'use_prioritized_replay', True))
            batch = _sample_to_batch(sampled)
            metrics = engine.Training_Step(batch=batch, optimizer='adam', tabular_method='sarsa')
            if isinstance(metrics, dict):
                pinn_update = metrics.get('pinn_update')
                if isinstance(pinn_update, dict) and 'total_loss' in pinn_update:
                    last_loss = _as_scalar(pinn_update['total_loss'])
                elif 'rl_proxy_loss' in metrics:
                    last_loss = float(metrics['rl_proxy_loss'])
                history["physics_loss"].append(float(metrics.get("physics_loss", np.nan)))
                history["policy_loss"].append(float(metrics.get("policy_loss", np.nan)))
                history["entropy"].append(float(metrics.get("entropy", np.nan)))
                history["mean_q"].append(float(metrics.get("mean_q", np.nan)))
            else:
                history["physics_loss"].append(np.nan)
                history["policy_loss"].append(np.nan)
                history["entropy"].append(np.nan)
                history["mean_q"].append(np.nan)
        else:
            history["physics_loss"].append(np.nan)
            history["policy_loss"].append(np.nan)
            history["entropy"].append(np.nan)
            history["mean_q"].append(np.nan)
        eval_metrics = evaluate_policy(engine, eval_env, episodes=min(EVAL_EPISODES, len(eval_env.order)))
        history['episode_reward'].append(ep_reward)
        history['episode_length'].append(ep_steps)
        history['train_loss'].append(float(last_loss)) if np.isfinite(last_loss) else np.nan
        history['eval_accuracy'].append(float(eval_metrics['accuracy']))
        history['eval_reward'].append(float(eval_metrics['mean_reward']))
        history['replay_size'].append(float(replay_size))
        print(
            f'Episode {ep+1:03d} | '
            f'train_reward={ep_reward:.4f} | '
            f'eval_acc={eval_metrics["accuracy"]:.4f} | '
            f'loss={last_loss:.6f} | '
            f'replay={replay_size}'
        )
        if checkpoint_dir is not None and hasattr(engine, 'save'):
            try:
                engine.Save(os.path.join(checkpoint_dir, f'pirl_ep_{ep+1:03d}.pkl'))
            except Exception as e:
                print(f'[WARN] checkpoint failed to save on episode {ep+1}: {e}')
    plot_metric(history['episode_reward'], 'Training Reward per Episode', 'Reward', os.path.join(output_dir, 'training_reward.png'))
    plot_metric(history['eval_accuracy'], 'Evaluation Accuracy per Episode', 'Accuracy', os.path.join(output_dir, 'evaluation_accuracy.png'), smooth_window=5)
    plot_metric([x if np.isfinite(x) else 0.0 for x in history['train_loss']], 'Training Loss per Episode', 'Loss', os.path.join(output_dir, 'training_loss.png'))
    plot_metric(history['replay_size'], 'Replay Buffer Growth', 'Replay Size', os.path.join(output_dir, 'replay_size.png'))
    print(f'[INFO] Plot saved on: {os.path.abspath(output_dir)}')
    print(f"[INFO] Accuracy of Last Evaluation: {history['eval_accuracy'][-1]:.4f}")
    print(f"[INFO] Reward of Last Evaluation: {history['eval_reward'][-1]:.4f}")
    return engine, history
"""

"""
def run_launcher(
    epoch: int = EPOCH,
    batch_size: int = BATCH_SIZE,
    train_every: int = TRAIN_EVERY,
    seed: int = SEED,
    hidden_dim: int = HIDDEN_DIM,
    buffer_capacity: int = BUFFER_CAPACITY,
    checkpoint_dir: str | None = None,
    output_dir: str = OUTPUT_DIR
):
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    train_env = OneStepImageClassificationEnv(train=True, seed=seed, download=True)
    Show_MNIST_Samples(
        train_env.dataset,
        n_samples=49,
        random_sample=False
    )
    eval_env = OneStepImageClassificationEnv(train=False, seed=seed, download=True)
    engine = build_engine(
        env=train_env,
        seed=seed,
        hidden_dim=hidden_dim,
        buffer_capacity=buffer_capacity,
    )
    print(f"[INFO] Dataset: {train_env.dataset_name}")
    print(f"[INFO] Input shape: {train_env.observation_shape}")
    print(f"[INFO] Actions: {train_env.num_actions}")
    print(f"[INFO] State dim: {train_env.state_dim}")
    print(f"[INFO] Engine: {engine.__class__.__name__}")
    print("[INFO] Training_PIRL.py acts only as a launcher; the PIRLA engine performs the learning.")

    history: Dict[str, List[float]] = {
        'episode_reward': [],
        'episode_length': [],
        'train_loss': [],
        'physics_loss': [],
        'policy_loss': [],
        'entropy': [],
        'eval_accuracy': [],
        'eval_reward': [],
        'replay_size': [],
        'mean_q': [],
        'temperature': [],
        'epsilon': [],
    }

    for ep in range(int(epoch)):
        # obs, _info = _reset_env(train_env)
        ep_reward = 0.0
        ep_steps = 0
        last_loss = np.nan

        # for _t in range(int(STEPS)):
        #     action = engine.Select_Action(obs, strategy='auto')
        #     next_obs, reward, terminated, truncated, _step_info = _step_env(train_env, action)
        #     done = bool(terminated or truncated)
        #     next_action = engine.Select_Action(next_obs, strategy='auto')
        #     engine.Store_Transition(
        #         state=obs,
        #         action=action,
        #         reward=reward,
        #         next_state=next_obs,
        #         next_action=next_action,
        #         done=done,
        #     )
        #     ep_reward += float(reward)
        #     ep_steps += 1
        #     obs = next_obs
        #     if done:
        #         break

        for ep in range(epoch):
            ep_reward = 0.0
            ep_correct = 0
            ep_total = 0
            ep_loss = []
            if train_env.train:
                train_env._build_order()
            for idx in train_env.order:
                obs, label = train_env._get_item(idx)
                action = engine.Select_Action(obs, strategy="auto")
                reward = float(action == label)
                next_obs = obs
                engine.Store_Transition(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    next_action=action,
                    done=True,
                )
                ep_reward += reward
                ep_correct += int(reward)
                ep_total += 1

        replay = getattr(engine, 'replay', None)
        replay_size = _replay_size(replay)

        if replay_size >= batch_size:
            updates = max(1, replay_size // batch_size)
            for _ in range(updates):
                sampled = engine.Sample_Batch(
                    batch_size=batch_size,
                    prioritized=getattr(engine, "use_prioritized_replay", True,),
                )
                batch = _sample_to_batch(sampled)
                metrics = engine.Training_Step(
                    batch=batch,
                    optimizer="adam",
                    tabular_method="sarsa",
                )
                if isinstance(metrics, dict):
                    pinn = metrics.get("pinn_update")
                    if pinn is not None:
                        if "total_loss" in pinn:
                            ep_loss.append(_as_scalar(pinn["total_loss"]))
                    history["physics_loss"].append(float(metrics.get("physics_loss", np.nan,)))
                    history["policy_loss"].append(float(metrics.get("policy_loss", np.nan,)))
                    history["entropy"].append(float(metrics.get("entropy", np.nan,)))
                    history["mean_q"].append(float(metrics.get("mean_q", np.nan,)))
        if replay_size >= int(batch_size) and ((ep + 1) % int(train_every) == 0):
            sampled = engine.Sample_Batch(batch_size, prioritized=getattr(engine, 'use_prioritized_replay', True))
            batch = _sample_to_batch(sampled)
            metrics = engine.Training_Step(batch=batch, optimizer='adam', tabular_method='sarsa')
            if isinstance(metrics, dict):
                pinn_update = metrics.get('pinn_update')
                if isinstance(pinn_update, dict) and 'total_loss' in pinn_update:
                    last_loss = _as_scalar(pinn_update['total_loss'])
                elif 'rl_proxy_loss' in metrics:
                    last_loss = float(metrics['rl_proxy_loss'])
                history["physics_loss"].append(float(metrics.get("physics_loss", np.nan)))
                history["policy_loss"].append(float(metrics.get("policy_loss", np.nan)))
                history["entropy"].append(float(metrics.get("entropy", np.nan)))
                history["mean_q"].append(float(metrics.get("mean_q", np.nan)))
            else:
                history["physics_loss"].append(np.nan)
                history["policy_loss"].append(np.nan)
                history["entropy"].append(np.nan)
                history["mean_q"].append(np.nan)
        else:
            history["physics_loss"].append(np.nan)
            history["policy_loss"].append(np.nan)
            history["entropy"].append(np.nan)
            history["mean_q"].append(np.nan)
        eval_metrics = evaluate_policy(engine, eval_env, episodes=min(EVAL_EPISODES, len(eval_env.order)))
        history['episode_reward'].append(ep_reward)
        history['episode_length'].append(ep_steps)
        history['train_loss'].append(float(last_loss)) if np.isfinite(last_loss) else np.nan
        history['eval_accuracy'].append(float(eval_metrics['accuracy']))
        history['eval_reward'].append(float(eval_metrics['mean_reward']))
        history['replay_size'].append(float(replay_size))
        print(
            f'Episode {ep+1:03d} | '
            f'train_reward={ep_reward:.4f} | '
            f'eval_acc={eval_metrics["accuracy"]:.4f} | '
            f'loss={last_loss:.6f} | '
            f'replay={replay_size}'
        )
        if checkpoint_dir is not None and hasattr(engine, 'save'):
            try:
                engine.Save(os.path.join(checkpoint_dir, f'pirl_ep_{ep+1:03d}.pkl'))
            except Exception as e:
                print(f'[WARN] checkpoint failed to save on episode {ep+1}: {e}')
    plot_metric(history['episode_reward'], 'Training Reward per Episode', 'Reward', os.path.join(output_dir, 'training_reward.png'))
    plot_metric(history['eval_accuracy'], 'Evaluation Accuracy per Episode', 'Accuracy', os.path.join(output_dir, 'evaluation_accuracy.png'), smooth_window=5)
    plot_metric([x if np.isfinite(x) else 0.0 for x in history['train_loss']], 'Training Loss per Episode', 'Loss', os.path.join(output_dir, 'training_loss.png'))
    plot_metric(history['replay_size'], 'Replay Buffer Growth', 'Replay Size', os.path.join(output_dir, 'replay_size.png'))
    print(f'[INFO] Plot saved on: {os.path.abspath(output_dir)}')
    print(f"[INFO] Accuracy of Last Evaluation: {history['eval_accuracy'][-1]:.4f}")
    print(f"[INFO] Reward of Last Evaluation: {history['eval_reward'][-1]:.4f}")
    return engine, history
"""

def _scalar_action(action) -> int:
    if hasattr(action, "item"):
        try:
            return int(action.item())
        except Exception:
            pass
    if cp is not None:
        try:
            if isinstance(action, cp.ndarray):
                action = cp.asnumpy(action)
        except Exception:
            pass
    arr = np.asarray(action)
    if arr.ndim > 0 and arr.size > 1:
        return int(np.argmax(arr))
    return int(arr.reshape(-1)[0])

def run_launcher(
    epoch: int = EPOCH,
    batch_size: int = BATCH_SIZE,
    train_every: int = TRAIN_EVERY,
    seed: int = SEED,
    hidden_dim: int = HIDDEN_DIM,
    buffer_capacity: int = BUFFER_CAPACITY,
    checkpoint_dir: str | None = None,
    output_dir: str = OUTPUT_DIR
):
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    train_env = OneStepImageClassificationEnv(train=True, seed=seed, download=True)
    Show_MNIST_Samples(
        train_env.dataset,
        n_samples=49,
        random_sample=False
    )
    eval_env = OneStepImageClassificationEnv(train=False, seed=seed, download=True)
    engine = build_engine(
        env=train_env,
        seed=seed,
        hidden_dim=hidden_dim,
        buffer_capacity=buffer_capacity,
    )
    print(f"[INFO] Dataset: {train_env.dataset_name}")
    print(f"[INFO] Input shape: {train_env.observation_shape}")
    print(f"[INFO] Actions: {train_env.num_actions}")
    print(f"[INFO] State dim: {train_env.state_dim}")
    print(f"[INFO] Engine: {engine.__class__.__name__}")
    print("[INFO] Training_PIRL.py acts only as a launcher; the PIRLA engine performs the learning.")

    history: Dict[str, List[float]] = {
        'episode_reward': [],
        'episode_length': [],
        'train_loss': [],
        'physics_loss': [],
        'policy_loss': [],
        'entropy': [],
        'eval_accuracy': [],
        'eval_reward': [],
        'replay_size': [],
        'mean_q': [],
        'temperature': [],
        'epsilon': [],
        'train_accuracy': [],
    }

    best_accuracy = -1.0
    for ep in range(epoch):
        if train_env.train:
            train_env._build_order()
        ep_reward = 0.0
        ep_correct = 0
        ep_total = 0
        epoch_losses = []
        physics_vals = []
        policy_vals = []
        entropy_vals = []
        mean_q_vals = []
        for idx in train_env.order:
            obs, label = train_env._get_item(idx)
            action = _scalar_action(engine.Select_Action(obs, strategy="auto",))
            label = int(label)
            reward = float(action == label)
            engine.Store_Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=obs,
                next_action=action,
                done=True,
            )
            ep_reward += reward
            ep_correct += int(reward)
            ep_total += 1
        replay = getattr(engine, "replay", None)
        replay_size = _replay_size(replay)
        if replay_size >= batch_size:
            updates = max(1, replay_size // batch_size)
            for _ in range(updates):
                sampled = engine.Sample_Batch(
                    batch_size=batch_size,
                    prioritized=getattr(engine, "use_prioritized_replay", True,),
                )
                batch = _sample_to_batch(sampled)
                metrics = engine.Training_Step(
                    batch=batch,
                    optimizer="adam",
                    tabular_method="sarsa",
                )
                if isinstance(metrics, dict):
                    pinn = metrics.get("pinn_update")
                    if isinstance(pinn, dict) and "total_loss" in pinn:
                            epoch_losses.append(_as_scalar(pinn["total_loss"]))
                    physics_vals.append(float(metrics.get("physics_loss", np.nan)))
                    policy_vals.append(float(metrics.get("policy_loss", np.nan)))
                    entropy_vals.append(float(metrics.get("entropy", np.nan)))
                    mean_q_vals.append(float(metrics.get("mean_q", np.nan)))
        train_accuracy = ep_correct / max(ep_total, 1)
        train_reward = ep_reward / max(ep_total, 1)
        train_loss = float(np.mean(epoch_losses)) if len(epoch_losses) else np.nan
        physics_loss = float(np.nanmean(physics_vals)) if len(physics_vals) else np.nan
        policy_loss = float(np.nanmean(policy_vals)) if len(policy_vals) else np.nan
        entropy = float(np.nanmean(entropy_vals)) if len(entropy_vals) else np.nan
        mean_q = float(np.nanmean(mean_q_vals)) if len(mean_q_vals) else np.nan
        eval_metrics = evaluate_policy(
            engine,
            eval_env,
            episodes=min(
                EVAL_EPISODES,
                len(eval_env.order),
            ),
        )
        history["episode_reward"].append(train_reward)
        history["episode_length"].append(ep_total)
        history["train_loss"].append(train_loss)
        history["physics_loss"].append(physics_loss)
        history["policy_loss"].append(policy_loss)
        history["entropy"].append(entropy)
        history["mean_q"].append(mean_q)
        history["eval_accuracy"].append(eval_metrics["accuracy"])
        history["eval_reward"].append(eval_metrics["mean_reward"])
        history["replay_size"].append(replay_size)
        # history.setdefault("train_accuracy", [],).append(train_accuracy)
        history["train_accuracy"].append(train_accuracy)
        print(
            f"[Epoch {ep+1:03d}] "
            f"TrainAcc={train_accuracy:.4f} "
            f"TrainReward={train_reward:.4f} "
            f"EvalAcc={eval_metrics['accuracy']:.4f} "
            f"Loss={train_loss:.6f} "
            f"Replay={replay_size}"
        )
        if (
            checkpoint_dir is not None
            and hasattr(engine, "Save")
            and eval_metrics["accuracy"] > best_accuracy
        ):
            try:
                best_accuracy = eval_metrics["accuracy"]
                engine.Save(os.path.join(checkpoint_dir, "best_model.pkl"))
            except Exception as e:
                print(f"[WARN] failed to save best model: {e}")

    plot_metric(
        history['episode_reward'],
        'Training Reward per Episode',
        'Reward',
        os.path.join(output_dir, 'training_reward.png')
    )
    plot_metric(
        history['eval_accuracy'],
        'Evaluation Accuracy per Episode',
        'Accuracy',
        os.path.join(output_dir, 'evaluation_accuracy.png'),
        smooth_window=5
    )
    plot_metric(
        [x if np.isfinite(x) else 0.0 for x in history['train_loss']],
        'Training Loss per Episode',
        'Loss',
        os.path.join(output_dir, 'training_loss.png')
    )
    plot_metric(
        history['replay_size'],
        'Replay Buffer Growth',
        'Replay Size',
        os.path.join(output_dir, 'replay_size.png')
    )
    plot_metric(
        history["train_accuracy"],
        "Training Accuracy",
        "Accuracy",
        os.path.join(output_dir, "training_accuracy.png",),
    )
    print(f'[INFO] Plot saved on: {os.path.abspath(output_dir)}')
    print(f"[INFO] Accuracy of Last Evaluation: {history['eval_accuracy'][-1]:.4f}")
    print(f"[INFO] Reward of Last Evaluation: {history['eval_reward'][-1]:.4f}")
    return engine, history


def main():
    parser = argparse.ArgumentParser(description='Launcher for PIRLA + Training Visualization.')
    parser.add_argument('--epoch', type=int, default=EPOCH)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--train_every', type=int, default=TRAIN_EVERY)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM)
    parser.add_argument('--buffer_capacity', type=int, default=BUFFER_CAPACITY)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_pirl')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    engine, history = run_launcher(
        epoch=args.epoch,
        batch_size=args.batch_size,
        train_every=args.train_every,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        buffer_capacity=args.buffer_capacity,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )

    output_dir = args.output_dir

    plot_metric(
        history["physics_loss"],
        "Physics Loss",
        "Loss",
        os.path.join(output_dir, "physics_loss.png")
    )

    plot_metric(
        history["policy_loss"],
        "Policy Loss",
        "Loss",
        os.path.join(output_dir, "policy_loss.png")
    )

    plot_metric(
        history["entropy"],
        "Entropy",
        "Entropy",
        os.path.join(output_dir, "entropy.png")
    )

    plot_metric(
        history["mean_q"],
        "Mean Q Value",
        "Q",
        os.path.join(output_dir, "mean_q.png")
    )

    if hasattr(engine, "env") and getattr(engine, "env", None) is not None:
        try:
            Show_Prediction_Samples(
                engine=engine,
                dataset=engine.env.dataset,
                n_samples=25,
                random_sample=True,
                save_path=os.path.join(output_dir, "prediction_samples.png")
            )
        except Exception as e:
            print(f"[WARN] Could not display prediction samples: {e}")


if __name__ == '__main__':
    main()





















