# PIRLA - Physics-Informed Reinforcement Learning Agent
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



import cupy as cp
import numpy as np
import pickle
import traceback


EPSILON = 0.1
TEMPERATURE = 1.0
K_VALUE = 5
REWARD_VAR = 0.0
ENTROPY = 0.0
ENTROPY_THRESHOLD = 1.0
VAR_THRESHOLD = 1.0
GLOBAL_TEMPERATURE = 1.0

GLOBAL_TEMPERATURE_MIN = 0.01
GLOBAL_TEMPERATURE_MAX = 5.0
LOCAL_TEMPERATURE_MIN = 0.01
LOCAL_TEMPERATURE_MAX = 10.0
LOCAL_TEMPERATURE_MAX = 10.0

ALPHA = 0.1
GAMMA = 0.99
LAMBDA = 0.95
TAU = 0.03
BETA1 = 0.9
BETA2 = 0.999
SIGMA = 0.01
LEARNING_RATE = 0.01

RANK = 4

ENTROPY_EPS = 1e-8
SPARSE_EPS = 1e-6

NUM_ACTIONS = 10
NUM_OPTIONS = 4
MAX_STEPS_PER_OPTION = 20
OPTION_EXPLORATION = 0.10
EPS = 1e-6
EPS2 = 1e-8
EPS3 = 1e-3

KULLBACK_LEIBLER_COEFF = 0.0

TILE_SIZE = 16
PAD_VALUE = 0.0

NUM_ENSEMBLE = 2
RESIDUAL_WEIGHT = 0.0

MUNCHAUSEN_COEF = 0.9
LOG_CLIP_LOWER = -1.0

BUFFER_CAPACITY = 100000

POLICY_LR = 1e-3
VALUE_LR = 1e-3
CONVERGENCE_THRESHOLD = 1e-4

NUMBER_OF_AGENTS = 1
MOMENTUM = 0.99

TAU_TEMPERATURE = 1.0

PHYSICS_WEIGHT = 1.0
DATA_WEIGHT = 1.0
BOUNDARY_WEIGHT = 1.0

EPOCHS = 1000

REWARD_BONUS_SCALE = 1.0

STEPS = 1
EPISODES = 100
MAX_STEPS = 1000
TRAIN_EVERY = 1

RL_WEIGHT = 1.0
CURIOSITY_WEIGHT = 0.01
ENTROPY_WEIGHT = 0.001

LEARNING_RATE_GLOBAL = 0.01
LEARNING_RATE_LOCAL = 0.01
DECAY_RATE = 0.99
MIN_EPSILON = 0.01

BATCH_SIZE = 128

NUMBER_OF_LAYERS = 2
MAXIMUM_SEQUENCE_LEN = 1024
LAYER_NORM_EPS = 1e-5

COURANT_NUMBER = 1.0
NU = 0.01
X_DIM = 0
TIME_DIM=-1

NUMBER_OF_HEADS = 8
NUMBER_OF_ENVIRONMENT = 1

FF_DIM = 1024
STATE_DIM = 784


class Action_MultiStrategy:
    def __init__(
        self,
        num_actions: int,
        seed: int | None = None
    ):
        self.num_actions = int(num_actions)
        self.rng = cp.random.default_rng(seed)

    def Prepare_Q_Rows(
        self,
        Q,
        states
    ):
        Q = cp.asarray(Q)
        states = cp.asarray(states, dtype=cp.int32).ravel()
        if Q.ndim == 1:
            if Q.size % self.num_actions != 0:
                raise ValueError("Length of Q 1D must be a multiple of number of action")
            q_rows = Q.reshape(-1, self.num_actions)[states]
        elif Q.ndim == 2:
            if Q.shape[1] != self.num_actions:
                raise ValueError(f"Q.shape[1] must have the same length as number of action = {self.num_actions}")
            q_rows = Q[states]
        else:
            raise ValueError("Q must be 1D or 2D")
        return q_rows, states

    def Select_Action_Epsilon_Greedy(
        self,
        Q,
        states,
        epsilon,
        action_out=None
    ):
        q_rows, states = self.Prepare_Q_Rows(Q, states)
        q_rows = q_rows.astype(cp.float32, copy=False)
        greedy_actions = cp.argmax(q_rows, axis=1).astype(cp.int32)
        explore_mask = self.rng.random(states.size) < float(epsilon)
        random_actions = self.rng.integers(0, self.num_actions, size=states.size, dtype=cp.int32)
        selected_actions = cp.where(explore_mask, random_actions, greedy_actions)
        if action_out is not None:
            actions_out = cp.asarray(action_out, dtype=cp.int32)
            actions_out[:states.size] = selected_actions
            return actions_out
        return selected_actions

    def Softmax_Policy_Action_Select(
        self,
        Q,
        states,
        temperature,
        actions_out=None
    ):
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError("Temperature must be > 0.")
        q_rows, states = self.Prepare_Q_Rows(Q, states)
        q_rows = q_rows.astype(cp.float32, copy=False)
        q_rows = q_rows - cp.max(q_rows, axis=1, keepdims=True)
        logits = q_rows / temperature
        probs = cp.exp(logits)
        probs = probs / cp.sum(probs, axis=1, keepdims=True)
        cdf = cp.cumsum(probs, axis=1)
        u = self.rng.random(states.size, dtype=cp.float32)
        selected = cp.argmax(cdf >= u[:,None], axis=1).astype(cp.int32)
        if actions_out is not None:
            actions_out = cp.asarray(actions_out, dtype=cp.int32)
            actions_out[:states.size] = selected
            return actions_out
        return selected

    def Select_Action_TopK_Sampling(
        self,
        Q,
        states,
        K,
        actions_out=None
    ):
        K = int(K)
        if K <= 0:
            raise ValueError("K must be > 0.")
        q_rows, states = self.Prepare_Q_Rows(Q, states)
        q_rows = q_rows.astype(cp.float32, copy=False)
        K = min(K, self.num_actions)
        topk_idx = cp.argpartition(q_rows, -K, axis=1)[:, -K:]
        topk_vals = cp.take_along_axis(q_rows, topk_idx, axis=1)
        weights = topk_vals - cp.min(topk_vals, axis=1, keepdims=True) + 1e-6
        cdf = cp.cumsum(weights, axis=1)
        total = cdf[:, -1]
        u = self.rng.randim(states.size, dtype=cp.float32) * total
        selected_pos = cp.argmax(cdf >= u[:,None], axis=1)
        selected_actions = topk_idx[cp.arange(states.size), selected_pos].astype(cp.int32)
        if actions_out is not None:
            actions_out = cp.asarray(actions_out, dtype=cp.int32)
            actions_out[:states.size] = selected_actions
            return actions_out
        return selected_actions

    def Auto_Select_Action_MultiStrategy(
        self,
        Q,
        states,
        epsilon,
        temperature,
        K,
        reward_var,
        entropy,
        entropy_threshold,
        var_threshold,
        actions_out=None,
    ):
        if reward_var > var_threshold:
            return self.Select_Action_Epsilon_Greedy(
                Q=Q,
                states=states,
                epsilon=epsilon,
                actions_out=actions_out,
            )
        elif entropy > entropy_threshold:
            return self.Select_Action_TopK_Sampling(
                Q=Q,
                states=states,
                K=K,
                actions_out=actions_out,
            )
        else:
            return self.Softmax_Policy_Action_Select(
                Q=Q,
                states=states,
                temperature=temperature,
                actions_out=actions_out,
            )

    def Running_Action_MultiStrategy(
        self,
        Q,
        states,
        epsilon=EPSILON,
        temperature=TEMPERATURE,
        K=K_VALUE,
        reward_var=REWARD_VAR,
        entropy=ENTROPY,
        entropy_threshold=ENTROPY_THRESHOLD,
        var_threshold=VAR_THRESHOLD,
        actions_out=None,
    ):
        return self.Auto_Select_Action_MultiStrategy(
            Q=Q,
            states=states,
            epsilon=epsilon,
            temperature=temperature,
            K=K,
            reward_var=reward_var,
            entropy=entropy,
            entropy_threshold=entropy_threshold,
            var_threshold=var_threshold,
            actions_out=actions_out
        )


def Actor_Critic(
    self,
    actor_params,
    critic_params,
    actor_grads,
    critic_grads,
    lr_actor,
    lr_critic,
):
    actor_params = cp.asarray(actor_params)
    critic_params = cp.asarray(critic_params)
    actor_grads = cp.asarray(actor_grads)
    critic_grads = cp.asarray(critic_grads)
    if actor_params.shape != actor_grads.shape:
        raise ValueError("Actor Parameters and Actor Gradients must have the same shape.")
    if critic_params.shape != critic_grads.shape:
        raise ValueError("Critic Parameters and Critic Gradients must have the same shape.")
    lr_actor = cp.float32(lr_actor)
    lr_critic = cp.float32(lr_critic)
    actor_params[...] =(
        actor_params.astype(cp.float32, copy=False) -
        lr_actor * actor_grads.astype(cp.float32, copy=False)
    ).astype(actor_params.dtype, copy=False)
    critic_params[...] = (
        critic_params.astype(cp.float32, copy=False) -
        lr_critic * critic_grads.astype(cp.float32, copy=False)
    ).astype(critic_params.dtype, copy=False)


class Adaptive_MetaLearning:
    def __init__(
        self,
        num_agents: int,
        temperature_init: float = TEMPERATURE,
        global_temperature_init: float = GLOBAL_TEMPERATURE,
        epsilon_init: float = EPSILON,
        dtype=cp.float32,
    ):
        self.num_agents = int(num_agents)
        self.dtype = dtype
        self.temperature_array = cp.full(self.num_agents, temperature_init, dtype=dtype)
        self.global_temperature = cp.array(global_temperature_init, dtype=dtype)
        self.epsilon = cp.full(self.num_agents, epsilon_init, dtype=dtype)

    def Adaptive_Temperature_Scheduler(
        self,
        mean_entropy_local,
        mean_entropy_global,
        target_entropy,
        lr_global,
        lr_local,
        temperature_array=None,
        global_temperature=None,
        global_temperature_min: float = GLOBAL_TEMPERATURE_MIN,
        global_temperature_max: float = GLOBAL_TEMPERATURE_MAX,
        local_temperature_min: float = LOCAL_TEMPERATURE_MIN,
        local_temperature_max: float = LOCAL_TEMPERATURE_MAX,
    ):
        temperature_array = self.temperature_array if temperature_array is None else cp.asarray(temperature_array, dtype=self.dtype)
        global_temperature = self.global_temperature if global_temperature is None else cp.asarray(global_temperature, dtype=self.dtype)
        mean_entropy_local = cp.asarray(mean_entropy_local, dtype=self.dtype).ravel()
        if mean_entropy_local.size != self.num_agents:
            raise ValueError("Mean Entropy Local must have the same length as number of agents")
        mean_entropy_global = cp.asarray(mean_entropy_global, dtype=self.dtype)
        target_entropy = cp.asarray(target_entropy, dtype=self.dtype)
        lr_global = cp.asarray(lr_global, dtype=self.dtype)
        lr_local = cp.asarray(lr_local, dtype=self.dtype)
        delta_global = mean_entropy_global - target_entropy
        global_temperature[...] = cp.clip(
            global_temperature + lr_global * delta_global,
            global_temperature_min,
            global_temperature_max
        )
        temp_g = cp.asarray(global_temperature, dtype=self.dtype)
        correction = lr_local * (mean_entropy_local - target_entropy) * temp_g
        temperature_array[...] = cp.clip(
            temperature_array + correction,
            local_temperature_min,
            local_temperature_max,
        )
        self.temperature_array = temperature_array
        self.global_temperature = global_temperature
        return temperature_array, global_temperature

    def Epsilon_Decay(
        self,
        mean_reward,
        var_reward,
        decay_rate,
        min_epsilon,
        epsilon_array=None,
    ):
        epsilon_array = self.epsilon if epsilon_array is None else cp.asarray(epsilon_array, dtype=self.dtype)
        _ = mean_rewad
        var_reward = cp.asarray(var_reward, dtype=self.dtype)
        decay_rate = cp.asarray(decay_rate, dtype=self.dtype)
        min_epsilon = cp.asarray(min_epsilon, dtype=self.dtype)
        confidence = 1.0 / (1.0 + var_reward)
        confidence = cp.clip(confidence, 0.01, 1.0)
        decay_conf = decay_rate * confidence
        new_eps = epsilon_array * decay_conf + min_epsilon
        epsilon_array[...] = cp.maximum(new_eps, min_epsilon)
        self.epsilon = epsilon_array
        return epsilon_array

    def Adaptive_MetaLearning_Step(
        self,
        mean_entropy_local,
        mean_entropy_global,
        target_entropy,
        lr_global,
        lr_local,
        mean_reward,
        var_reward,
        decay_rate,
        min_epsilon,
        temperature_array=None,
        global_temperature=None,
        epsilon_array=None,
    ):
        temp_array, global_temp = self.adaptive_temperature_scheduler(
            mean_entropy_local=mean_entropy_local,
            mean_entropy_global=mean_entropy_global,
            target_entropy=target_entropy,
            lr_global=lr_global,
            lr_local=lr_local,
            temperature_array=temperatre_array,
            global_temperature=global_temperature
        )

        eps_array = self.Epsilon_Decay(
            mean_reward=mean_reward,
            var_reward=var_reward,
            decay_rate=decay_rate,
            min_epsilon=min_epsilon,
            epsilon_array=epsilon_array
        )

        return temp_array, global_temp, eps_array


    def get_state(self):
        return {
            "temperature_array": self.temperature_array,
            "global_temperature": self.global_temperature,
            "epsilon": self.epsilon
        }


class Credit_Assignment:
    def __init__(
        self,
        num_actions: int,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        lambda_: float = LAMBDA,
        learning_rate: float = LEARNING_RATE,
        dtype=cp.float16,
        compute_dtype=cp.float32,
    ):
        self.num_actions = int(num_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.learning_rate = float(learning_rate)
        self.dtype = dtype
        self.compute_dtype = compute_dtype

    def Prepare_Q(
        self,
        Q
    ):
        Q = cp.asarray(Q)
        if Q.ndim == 1:
            if Q.size % self.num_actions != 0:
                raise ValueError("Length of 1D Q must have a multiple of number of actions.")
            Q_view = Q.reshape(-1, self.num_actions)
        elif Q.ndim == 2:
            if Q.shape[1] != self.num_actions:
                raise ValueError("Q.shape[1] must have the same as number of actions")
            Q_view = Q
        else:
            raise ValueError("Q must be 1D or 2D")
        return Q_view

    def TD_Lambda(
        self,
        transitions,
        Q,
        alpha=None,
        gamma=None,
        lambda_=None,
    ):
        Q_view = self.Prepare_Q(Q)
        Q_flat = Q_view.reshape(-1)
        transitions = cp.asarray(transitions)
        if transitions.ndim == 2:
            transitions = transitions[None, ...]
        if transitions.ndim != 3 or transitions.shape[-1] < 5:
            raise ValueError("Transitions must have the shape of [E, T, 5] OR [T, 5].")
        alpha = cp.asarray(self.alpha if alpha is None else alpha, dtype=self.compute_dtype)
        gamma = cp.asarray(self.gamma if gamma is None else gamma, dtype=self.compute_dtype)
        lambda_ = cp.asarray(self.lambda_ if lambda_ is None else lambda_, dtype=self.compute_dtype)
        num_episodes, T, _ = transitions.shape
        for ep in range(num_episodes):
            episode = transitions[ep]
            e = cp.zeros(T, dtype=self.compute_dtype)
            for t in range(T):
                s = int(episode[t,0].item())
                a = int(episode[t,1].item())
                r = cp.asarray(episode[t,2], dtype=self.compute_dtype)
                s_next = int(episode[t,3].item())
                a_next = int(episode[t,4].item())
                idx_sa = s * self.num_actions + a
                idx_snext_anext = s_next * self.num_actions + a_next
                Q_sa = cp.asarray(Q_flat[idx_sa], dtype=self.compute_dtype)
                Q_snext_anext = cp.asarray(Q_flat[idx_snext_anext], dtype=self.compute_dtype)
                delta = r + gamma * Q_snext_anext - Q_sa
                if t > 0:
                    e[:t] *= gamma * lambda_
                e[t] += 1.0
                for i in range(t+1):
                    s_i = int(episode[i,0].item())
                    a_i = int(episode[i,1].item())
                    idx_i = s_i * self.num_actions + a_i
                    q_old = cp.asarray(Q_flat[idx_i], dtype=self.compute_dtype)
                    update = alpha * delta * e[i]
                    Q_flat[idx_i] = (q_old + update).astype(Q_flat.dtype)
        return Q_view

    def Fused_Eligibility_Traces(
        self,
        rewards,
        errors,
        weights,
        discount=None,
        lambda_decay=None,
        learning_rate=None
    ):
        rewards = cp.asarray(rewards, dtype=self.compute_dtype).ravel()
        errors = cp.asarray(errors, dtype=self.compute_dtype).ravel()
        weights = cp.asarray(weights)
        T = rewards.size
        if errors.size != T or weights.size != T:
            raise ValueError("rewards, errors, and weights must have the same length")
        discount = cp.asarray(0.99 if discount is None else discount, dtype=self.compute_dtype)
        lambda_decay = cp.asarray(0.95 if lambda_decay is None else lambda_decay, dtype=self.compute_dtype)
        learning_rate = cp.asarray(self.learning_rate if learning_rate is None else learning_rate, dtype=self.compute_dtype)
        forward_trace = cp.empty(T, dtype=self.compute_dtype)
        G = cp.asarray(0.0, dtype=self.compute_dtype)
        for t in range(T-1, -1, -1):
            G = rewards[t] + discount * G
            forward_trace[t] = G
        eligibility_trace = cp.empty(T, dtype=self.compute_dtype)
        e_trace = cp.asarray(0.0, dtype=self.compute_dtype)
        for t in range(T):
            e_trace = lambda_decay * e_trace + errors[t]
            eligibility_trace[t] = e_trace
        weights[...] = (
            weights.astype(self.compute_dtype, copy=False) +
            learning_rate * forward_trace * eligibility_trace
        ).astype(weights.dtype, copy=False)
        return forward_trace.astype(self.dtype), eligibility_trace.astype(self.dtype), weights

    def Real_Time_Recurrent_Learning(
        self,
        history_Q,
        rank = RANK,
        lambda_decay=None,
        learning_rate=None,
        U=None,
        V=None
    ):
        history_Q = cp.asarray(history_Q, dtype=self.compute_dtype)
        if history_Q.ndim != 2:
            raise ValueError("History_Q must be a shape of [T, state_dim].")
        lambda_decay = cp.asarray(0.95 if lambda_decay is None else lambda_decay, dtype=self.compute_dtype)
        learning_rate = cp.asarray(self.learning_rate if learning_rate is None else learning_rate, dtype=self.compute_dtype)
        if U is None:
            U = cp.full((state_dim, rank), 0.01, dtype=self.compute_dtype)
        else:
            U = cp.asarray(U, dtype=self.compute_dtype)
            if U.shape != (state_dim, rank):
                raise ValueError("Shape U must be [state_dim, rank].")
        if V is None:
            V = cp.full((state_dim, rank), 0.01, dtype=self.compute_dtype)
        else:
            V = cp.asarray(V, dtype=self.compute_dtype)
            if V.shape != (state_dim, rank):
                raise ValueError("Shape V must be [state_dim, rank].")
            meta_base = cp.sum(U*V, axis=1)
            eligibility = cp.zeros(state_dim, dtype=self.compute_dtype)
            for t in range(1,T):
                grad_t = history_Q[t] - history_Q[t-1]
                eligibility = lambda_decay * eligibility + grad_t * meta_base
            eligibility_trace = (eligibility * learning_rate).astype(self.dtype, copy=False)
        return eligibility_trace

    def Credit_Assignment_Step(
        self,
        mode: str,
        **kwargs,
    ):
        mode = mode.lower().strip()
        if mode == "TD_Lambda":
            return self.TD_Lambda(**kwargs)
        if mode == "Fused_Eligibility":
            return self.Fused_Eligibility_Traces(**kwargs)
        if mode == "Real_Time_Recurrent_Learning":
            return self.Real_Time_Recurrent_Learning(**kwargs)
        raise ValueError("Mode must be one of the: TD_Lambda, Fused_Eligibilitu, Real_Time_Recurrent_Learning")


class Curiosity_And_Regulation:
    def __init__(
        self,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        entropy_eps=ENTROPY_EPS,
        sparse_eps=SPARSE_EPS,
    ):
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.sparse_eps = sparse_eps

    def As_2D(
        self,
        x
    ):
        x = cp.asarray(x)
        if x.ndim == 1:
            return x[None,:]
        if x.ndim == 2:
            return x
        raise ValueError("Input must bs 1D or 2D")

    def Curiosity_Reward(
        self,
        state_embedddings,
        predicted_embeddings,
        reduce="l2"
    ):
        s = self.As_2D(state_embedddings).astype(self.compute_dtype, copy=False)
        p = self.As_2D(predicted_embeddings).astype(self.compute_dtype, copy=False)
        if s.shape != p.shape:
            raise ValueError("State Embeddings and Predicted Embeddings must have the same shape")
        diff = s - p
        if reduce.lower() == "l2":
            intrinsic = cp.sum(diff * diff, axis=1)
        elif reduce.lower() == "l1":
            intrinsic = cp.sum(cp.abs(diff), axis=1)
        elif reduce.lower() == "rmse":
            intrinsic = cp.sqrt(cp.mean(diff*diff, axis=1) + self, entropy_eps)
        else:
            raise ValueError("reduce must be one of the: 'l2', 'l1', 'rmse'")
        return intrinsic.astype(self.dtype, copy=False)

    def Q_Sparse_Regularization(
        self,
        Q,
        sparsity_lambda,
        in_place=True
    ):
        Q = cp.asarray(Q)
        lam = cp.asarray(sparsity_lambda, dtype=self.compute_dtype)
        qf = Q.astype(self.compute_dtype, copy=False)
        shrunk = cp.maximum(cp.abs(qf) - lam, 0.0)
        regulated = cp.sign(qf) * shrunk
        if in_place:
            Q[...] = regulated.astype(Q.dtype, copy=False)
            return Q
        return regulated.astype(Q.dtype, copy=False)

    def Policy_Entropy(
        self,
        log_probs,
        reduce="mean"
    ):
        lp = self.As_2D(log_probs).astype(self.compute_dtype, copy=False)
        lp = lp - cp.max(lp, axis=1, keepdims=True)
        p = cp.exp(lp)
        entropy = -cp.sum(p*lp, axis=1)
        if reduce == "none":
            return entropy.astype(self.dtype, copy=False)
        if reduce == "mean":
            return cp.mean(entropy).astype(self.dtype, copy=False)
        if reduce == "sum":
            return cp.sum(entropy).astype(self.dtype, copy=False)
        raise ValueError("reduce must be one of the: 'none', 'mean', 'sum'.")

    def Regulate_Policy_And_Q(
        self,
        state_embeddings,
        predicted_embeddings,
        log_probs,
        Q,
        sparsity_lambda,
        entropy_reduce="mean",
        in_place_q=True,
    ):
        curiosity_rewards = self.Curiosity_Reward(state_embeddings, predicted_embeddings)
        policy_ent = self.Policy_Entropy(log_probs, reduce=entropy_reduce)
        Q_reg = self.q_sparse_regularization(Q, sparsity_lambda, in_place=in_place_q)
        return {
            "Curiosity_Rewards": curiosity_rewards,
            "Policy_Entropy": policy_ent,
            "Q_regularized": Q_reg,
        }

    def Curiosity_And_Regulation_Step(
        self,
        state_embeddings,
        predicted_embeddings,
        log_probs,
        Q,
        sparsity_lambda,
        entropy_reduce="mean",
        in_place_q=True,
    ):
        return self.Regulate_Policy_And_Q(
            state_embeddings=state_embeddings,
            predicted_embeddings=predicted_embeddings,
            log_probs=log_probs,
            Q=Q,
            sparsity_lambda=sparsity_lambda,
            entropy_reduce=entropy_reduce,
            in_place_q=in_place_q
        )


class Hierarchical_Temporal_Abstraction:
    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        num_options: int = NUM_OPTIONS,
        state_embed_dim: int | None = None,
        goal_embed_dim: int | None = None,
        max_steps_per_option: int = MAX_STEPS_PER_OPTION,
        option_exploration: float = OPTION_EXPLORATION,
        seed: int | None = None,
        dtype = cp.float32,
        eps: float = EPS,
    ):
        self.num_actions = int(num_actions)
        self.num_actions = int(num_options)
        self.state_embed_dim = None if state_embed_dim is None else int(state_embed_dim)
        self.goal_embed_dim = None if goal_embed_dim is None else int(goal_embed_dim)
        self.max_steps_per_option = int(max_steps_per_option)
        self.option_exploration = float(option_exploration)
        self.dtype = dtype
        self.eps = float(eps)
        self.rng = cp.random.default_rng(seed)

    def As_2D(
        self,
        x
    ):
        x = cp.asarray(x, dtype=self.dtype)
        if x.ndim == 1:
            return x[None,:]
        if x.ndim == 2:
            return x
        raise ValueError("Iput must be 1D or 2D")

    def Cosine_Similarity(
        self,
        vec1,
        vec2
    ):
        v1 = self.As_2D(vec1)
        v2 = self.As_2D(vec2)
        if v1.shape != v2.shape:
            raise ValueError("vec1 and vec2 must have the same shape.")
        dot = cp.sum(v1*v2, axis=1)
        n1 = cp.sqrt(cp.sum(v1*v1, axis=1))
        n2 = cp.sqrt(cp.sum(v2*v2, axis=1))
        return dot / (n1*n2 + self.eps)

    def Termination_Condition(
        self,
        states,
        option_ids
    ):
        states = cp.asarray(states, dtype=cp.int32).ravel()
        option_ids = cp.asarray(option_ids, dtype=cp.int32).ravel()
        if states.size != option_ids.size:
            raise ValueError("states and option_ids must have the same length.")
        return (states % (option_ids+2)) == 0

    def Intra_Option_Policy(
        self,
        states,
        option_ids,
        num_actions=None,
        exploration_rate=None,
    ):
        states = cp.asarray(states, dtype=cp.int32).ravel()
        option_ids = cp.asarray(option_ids, dtype=cp.int32).ravel()
        if states.size != option_ids.size:
            raise ValueError("states and option_ids must have the same length")
        num_actions = self.num_actions if num_actions is None else int(num_actions)
        exploration_rate = self.option_exploration if exploration_rate is None else float(exploration_rate)
        base_action = (states + option_ids) % num_actions
        explore = self.rng.random(states.size) < exploration_rate
        random_action = self.rng.integers(0, num_actions, size_states.size, dtype=cp.int32)
        return cp.where(explore, random_action, base_action).astype(cp.int32)

    def Options_Framework(
        self,
        states,
        option_ids=None,
        num_actions=None,
        exploration_rate=None,
        max_steps_per_option=None,
    ):
        states = cp.asarray(states, dtype=cp.int32).ravel()
        num_agents = states.size
        num_actions = self.num_actions if num_actions is None else int(num_actions)
        exploration_rate = self.option_exploration if exploration_rate is None else float(exploration_rate)
        max_steps_per_option = self.max_steps_per_option if max_steps_per_option is None else int(max_steps_per_option)
        if option_ids is None:
            option_ids = cp.arange(self.num_options, dtype=cp.int32)
        else:
            option_ids = cp.asarray(option_ids, dtype=cp.int32).ravel()
        if option_ids.size == 0:
            raise ValueError("option_ids must not be empty.")
        cond = (states[:,None] % (option_ids[None,:]+2)) != 0
        has_any = cp.any(cond, axis=1)
        selected_idx = cp.where(has_any, cp.argmax(cond, axis=1), 0).astype(cp.int32)
        high_policy_option = selected_idx
        actions = cp.empty(num_agents, dtype=cp.int32)
        next_states = states.copy()
        for i in range(num_agents):
            s = int(next_states[i].item())
            opt_idx = int(selected_idx[i].item())
            opt_id = int(option_ids[opt_idx].item())
            t = 0
            last_action = 0
            while t < max_steps_per_option:
                if (s % (opt_id+2)) == 0:
                    break
                base_action = (s + opt_id) % num_actions
                if self.rng.random() < exploration_rate:
                    a = int(self.rng.integers(0,num_actions))
                else:
                    a = int(base_action)
                s = s + a
                last_action = a
                t += 1
            actions[i] = last_action
            next_states[i] = s
        return high_policy_option, actions, next_states

    def Manager_Generate_Goal(
        self,
        state_embeddings,
        weight_manager,
        bias_manager
    ):
        state_embeds = self.As_2D(state_embeddings)
        weight_manager = cp.asarray(weight_manager, dtype=self.dtype)
        bias_manager = cp.asarray(bias_manager, dtype=self.dtype).ravel()
        if self.state_embed_dim is not None and state_embeds.shape[1] != self.state_embed_dim:
            raise ValueError("Dimension of state_embeddings does not match with the state_embed_dim.")
        if bias_manager.ndim != 1:
            raise ValueError("bias_manager must be 1D.")
        goal = state_embeds @ weight_manager.T + bias_manager[None,:]
        return cp.maximum(goal,0.0).astype(self.dtype, copy=False)

    def Worker_Policy(
        self,
        state_embeddings,
        goal_embeddings,
        weight_worker,
        bias_worker
    ):
        state_embeds = self.As_2D(state_embeddings)
        goal_embeds = self.As_2D(goal_embeddings)
        if state_embeds.shape[0] != goal_embeds.shape[0]:
            raise ValueError("State Embeddings and Goal Embeddings must have the same number of arrays.")
        combined = cp.concatenate([state_embeds, goal_embeds], axis=1)
        weight_worker = cp.asarray(weight_worker, dtype=self.dtype)
        bias_worker = cp.asarray(bias_worker, dtype=self.dtype).ravel()
        if self.goal_embed_dim is not None and goal_embeds.shape[1] != self.goal_embed_dim:
            raise ValueError("Dimension of Goal Embeddings does not match with the Goal Embedding Dim.")
        logits = combined @ weight_worker.T + bias_worker[None,:]
        return logits.astype(self.dtype, copy=False)

    def Softmax(
        self,
        logits
    ):
        logits = cp.asarray(logits, dtype=self.dtype)
        logits = logits - cp.max(logits, axis=1, keepdims=True)
        exp_logits = cp.exp(logits)
        return exp_logits / (cp.sum(exp_logits, axis=1, keepdims=True) + self.eps)

    def Sample_Actions(
        self,
        probs
    ):
        probs = cp.asarray(probs, dtype=self.dtype)
        cdf = cp.cumsum(probs, axis=1)
        u = self.rng.random((probs.shape[0], 1), dtype=self.dtype)
        return cp.argmax(cdf >= u, axis=1).astype(cp.int32)

    def Feudal_RL(
        self,
        d_state_embeddings,
        weight_manager,
        bias_manager,
        weigth_worker,
        bias_worker,
        return_probs=False,
    ):
        d_state_embeds = self.As_2D(d_state_embeddings)
        goal_embeds = self.Manager_Generate_Goal(d_state_embeds, weight_manager, bias_manager)
        action_logits = self.Worker_Policy(d_state_embeds, goal_embeds, weight_worker, bias_worker)
        action_probs = self.Softmax(action_logits)
        actions = self.sample_actions(action_probs)
        if return_probs:
            return goal_embeds, action_logits, action_probs, actions
        return goal_embeds, actions

    def Running_Hierarchical_Temporal_Abstraction(
        self,
        mode: str,
        **kwargs
    ):
        mode = mode.lower().strip()
        if mode == "options":
            return self.Options_Framework(**kwargs)
        if mode == "feudal":
            return self.Feudal_RL(**kwargs)
        if mode == "hybrid":
            out = {}
            if "states" in kwargs:
                out["options"] = self.Options_Framework(
                    states=kwargs["states"],
                    option_ids=kwargs.get("option_ids", None),
                    num_actions=kwargs.get("num_actions", None),
                    exploration_rate=kwargs.get("exploration_rate", None),
                    max_steps_per_option=kwargs.get("max_steps_per_option", None),
                )
            if "d_state_embeds" in kwargs:
                out["Feudal"] = self.Feudal_RL(
                    d_state_embeddings=kwargs["d_state_embeds"],
                    weight_manager=kwargs["weight_manager"],
                    bias_manager=kwargs["bias_manager"],
                    weight_worker=kwargs["weight_worker"],
                    bias_worker=kwargs["bias_worker"],
                    return_probs=kwargs.get("return_probs", False),
                )
            return out
        raise ValueError("Mode must be one of the: 'options', 'feudal', or 'hybrid'.")


def Logging_Q_Snapshot(
    self,
    Q,
    Q_log=None,
    snapshot_id: int = 0,
    max_snapshots: int | None = None,
    dtype=cp.float16,
):
    Q = cp.asarray(Q)
    if Q.ndim == 1:
        if not hasattr(self, "num_actions"):
            raise ValueError("Number of actions is not be found in the class")
        if Q.size % self.num_actions != 0:
            raise ValueError("Shape of Q is not compatible with number of actions.")
        num_states = Q.size // self.num_actions
        Q = Q.reshape(num_states, self.num_actions)
    elif Q.ndim == 2:
        num_states, num_actions = Q.shape
    else:
        raise ValueError("Q must be 1D or 2D array")
    if Q_log is None:
        if max_snapshots is None:
            max_snapshots = max(snapshot_id + 1, 1)
        Q_log = cp.zeros(
            (max_snapshots, num_states, num_actions),
            dtype=dtype
        )
    elif snapshot_id >= Q_log.shape[0]:
        new_size = max(snapshot_id+1, 2*Q_log.shape[0])
        num_buffer = cp.zeros(
            (new_size, num_states, num_actions),
            dtype=Q_log.dtype,
        )
        new_buffer[:Q_log.shape[0]] = Q_log
        Q_log = new_buffer
    Q_log[snapshot_id,:,:] = Q.astype(Q_log.dtype, copy=False,)
    return Q_log


class Network_And_Noise:
    def __init__(
        self,
        seed: int | None = None,
        dtype=cp.float32
    ):
        self.dtype = dtype
        self.rng = cp.random.default_rng(seed)

    def Apply_Parameter_Noise_Single(
        self,
        params,
        sigma: float,
        out_params=None
    ):
        params = cp.asarray(params, dtype=self.dtype)
        sigma = float(sigma)
        noise = self.rng.standard_normal(params.shape, dtype=self.dtype) * sigma
        out = params + noise
        if out_params is not None:
            out_params = cp.asarray(out_params, dtype=self.dtype)
            out_params[...] = out
            return out_params
        return out

    def Update_Target_Network(
        self,
        online_q,
        target_q,
        tau: float
    ):
        online_q = cp.asarray(online_q)
        target_q = cp.asarray(target_q)
        if online_q.shape != target_q.shape:
            raise ValueError("online_q and target_q must have the same shape.")
        tau = cp.asarray(tau, dtype=self.dtype)
        target_q[...] = tau * online_q.astype(self.dtype, copy=False) + (1.0 - tau) * target_q.astype(self.dtype, copy=False)
        return target_q

    def Add_Noise_And_Update_Target(
        self,
        params,
        sigma: float,
        online_q,
        target_q,
        tau: float,
    ):
        noisy_params = self.Apply_Parameter_Noise_Single(params, sigma=sigma)
        updated_target = self.Update_Target_Network(online_q=online_q, target_q=target_q, tau=tau)
        return noisy_params, updated_target

    def Running_Network_And_Noise(
        self,
        mode: str,
        **kwargs
    ):
        mode = mode.lower().strip()
        if mode == "noise":
            return self.Apply_Parameter_Noise_Single(
                params=kwargs["params"],
                sigma=kwargs["sigma"],
                out_params=kwargs.get("out_params", None),
            )
        if mode == "update_target":
            return self.Update_Target_Network(
                online_q=kwargs["online_q"],
                target_q=kwargs["target_q"],
                tau=kwargs["tau"],
            )
        if mode == "hybrid":
            return self.Add_Noise_And_Update_Target(
                params=lwargs["params"],
                sigma=kwargs["sigma"],
                online_q=kwargs["online_q"],
                target_q=kwargs["target_q"],
                tau=kwargs["tau"],
            )
        raise ValueError("Mode must be one of the: 'noise', 'update_target', or 'hybrid'.")


class Normalization:
    def __init__(
        self,
        eps: float = EPS,
        dtype = cp.float16,
        compute_dtype = cp.float32,
    ):
        self.eps = float(eps)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.running_reward_mean = None
        self.running_reward_var = None

    def Normalize_Q(
        self,
        Q,
        inplace: bool = True
    ):
        Q = cp.asarray(Q)
        if Q.ndim == 1:
            raise ValueError("Q must be a 2D shape: [Number of States, Number of Actions].")
        work = Q if inplace else Q.copy()
        qf = work.astype(self.compute_dtype, copy=False)
        q_min = cp.min(qf, axis=1, keepdims=True)
        q_max = cp.max(qf, axis=1, keepdims=True)
        q_range = q_max - q_min
        inv_range = cp.where(q_range > 0, 1.0 / q_range, 0.0)
        q_norm = (qf - q_min) * inv_range
        work[...] = q_norm.astype(work.dtype, copy=False)
        return

    def Normalize_Rewards(
        self,
        rewards,
        return_stats: bool = True
    ):
        rewards = cp.asarray(rewards, dtype=self.compute_dtype).ravel()
        mean = cp.mean(rewards)
        var = cp.var(rewards)
        std = cp.sqrt(cp.maximum(var,0.0) + self.eps)
        rewards_norm = ((rewards-mean)/std).astype(self.dtype, copy=False)
        if return_stats:
            return rewards_norm, mean, std
        return rewards_norm

    def Normalize_Rewards_Block(
        self,
        rewards,
        running: bool = False,
        momentum: float = 0.99,
        return_stats: bool = True,
    ):
        rewards = cp.asarray(rewards, dtype=self.compute_dtype).ravel()
        batch_mean = cp.mean(rewards)
        batch_var = cp.var(rewards)
        if running:
            if self.running_reward_mean is None:
                self.running_reward_mean = batch_mean
            if self.running_reward_var is None:
                self.running_reward_var = batch_var
            self.running_reward_mean = (momentum * self.running_reward_mean + (1.0 - momentum) * batch_mean)
            self.running_reward_var = (momentum * self.running_reward_var + (1.0 - momentum) * batch_var)
            mean = self.running_reward_mean
            var = self.running_reward_var
        else:
            mean = batch_mean
            var = batch_var
        std = cp.sqrt(cp.maximum(var,0.0) + self.eps)
        rewards_norm = ((rewards-mean)/std).astype(self.dtype, copy=False)
        if return_stats:
            return rewards_norm, mean, std
        return rewards_norm

    def Update_Baseline(
        self,
        rewards,
        baseline,
        alpha: float = 0.01,
        inplace: bool = True
    ):
        rewards = cp.asarray(rewards, dtype=self.compute_dtype).ravel()
        baseline = cp.asarray(baseline, dtype=self.compute_dtype).ravel()
        if rewards.shape != baseline.shape:
            raise ValueError("reward and baseline must have the same shape")
        updated = baseline + float(alpha) * (rewards - baseline)
        if inplace:
            baseline[...] = updated.astype(baseline.dtype, copy=False)
            return baseline.astype(self.dtype, copy=False)
        return updated.astype(self.dtype, copy=False)

    def Compute_Advantage(
        self,
        rewards,
        baseline
    ):
        rewards = cp.asarray(rewards, dtype=self.compute_dtype).ravel()
        baseline = cp.asarray(baseline, dtype=self.compute_dtype).ravel()
        if rewards.shape != baseline.shape:
            raise ValueError("reward and baseline must have the same shape!")
        return (rewards-baseline).astype(self.dtype, copy=False)

    def Step(
        self,
        Q=None,
        rewards=None,
        baseline=None,
        alpha: float = 0.01,
        running_rewards: bool = False,
        momentum: float = 0.99,
    ):
        out = {}
        if Q is not None:
            out["Q_normalized"] = self.normalize_q(Q,inplace=False)
        if rewards is not None:
            if running_rewards:
                r_norm, mean, std = self.normalize_rewards_block(rewards, running=True, momentum=momentum, return_stats=True,)
            else:
                r_norm, mean, std = self.normalize_rewards(rewards, return_stats=True,)
            out["rewards_normalized"] = r_norm
            out["reward_mean"] = mean
            out["reward_std"] = std
            if baseline is not None:
                new_baseline = self.update_baseline(rewards=rewards, baseline=baseline, alpha=alpha, inplace=False,)
                out["baseline"] = new_baseline
                out["advantage"] = self.compute_advantage(rewards, new_baseline)
        return out


class Policy_Loss_Option:
    def __init__(
        self,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        eps: float = EPS2,
    ):
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.eps = float(eps)

    def As_1D(
        self,
        X
    ):
        x = cp.asarray(x)
        return x.ravel()

    def Proximal_Policy_Optimalization_Clip_Advantage(
        self,
        old_log_probs,
        new_log_probs,
        advantages,
        epsilon: float = 0.2,
        reduce: str = "none",
    ):
        old_log_probs = self.As_1D(old_log_probs).astype(self.compute_dtype, copy=False)
        new_log_probs = self.As_1D(new_log_probs).astype(self.compute_dtype, copy=False)
        advantages = self.As_1D(advantages).astyype(self.compute_dtype, copy=False)
        if not (old_log_probs.shape == new_log_probs.shape == advantages.shape):
            raise ValueError("old_log_probs, new_log_probs, and advantages must have the same shape.")
        eps = cp.asarray(epsilon, dtype=self.compute_dtype)
        ratio = cp.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped_ratio = cp.clip(ratio, 1.0 - eps, 1.0 + eps)
        clipped = clipped_ratio * advantages
        loss = -cp.minimum(unclipped, clipped)
        if reduce == "none":
            return loss.astype(self.dtype, copy=False)
        if reduce == "mean":
            return cp.mean(loss).astype(self.compute_dtype)
        if reduce == "sum":
            return cp.sum(loss).astype(self.compute_dtype)
        raise ValueError("reduce must be one of the: 'none', 'mean', 'sum'.")

    def Kullback_Leibler_Divergence_Policy_Regularization(
        self,
        old_log_probs,
        new_log_probs,
        reduce: str = "none",
    ):
        old_log_probs = self.As_1D(old_log_probs).astype(self.compute_dtype, copy=False)
        new_log_probs = self.As_1D(new_log_probs).astype(self.compute_dtype, copy=False)
        if old_log_probs.shape != new_log_probs.shape:
            raise ValueError("old_log_probs and new_log_probs must have the same shape.")
        kl = old_log_probs - new_log_probs
        if reduce == "none":
            return kl.astype(self.dtype, copy=False)
        if reduce == "mean":
            return cp.mean(kl).astype(self.compute_dtype)
        if reduce == "sum":
            return cp.sum(kl).astype(self.compute_dtype)
        raise ValueError("reduce must be one of the: 'none', 'mean', 'sum'.")

    def Enforce_Trust_Region(
        self,
        old_policy,
        new_policy,
        max_kl: float,
        inplace: bool = True,
    ):
        old_policy = self.As_1D(old_policy).astype(self.compute_dtype, copy=False)
        new_policy = self.As_1D(new_policy).astype(self.compute_dtype, copy=False)
        if old_policy.shape != new_policy.shape:
            raise ValueError("Old Policy and New Policy must have the same shape!")
        diff = new_policy - old_policy
        kl_surrogate = cp.sum(diff*diff)
        max_kl = cp.asarray(max_kl, dtype=self.compute_dtype)
        scale = cp.where(kl_surrogate >  max_kl, cp.sqrt(max_kl/(kl_surrogate+self.eps)), 1.0)
        update = old_policy + scale * diff
        if inplace:
            new_policy[...] = updated.astype(new_policy.dtype, copy=False)
            return new_policy
        return updated.astype(self.dtype, copy=False)

    def Policy_Loss(
        self,
        old_log_probs,
        new_log_probs,
        advantages,
        epsilon: float = EPSILON,
        kl_coeff: float = KULLBACK_LEIBLER_COEFF,
        reduce: str = "mean",
    ):
        ppo_loss = self.ppo_clip_advantage(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            advantages=advantages,
            epsilon=epsilon,
            reduce="none",
        ).astype(self.compute_dtype, copy=False)
        if kl_coeff != 0.0:
            kl = self.Kullback_Leibler_Divergence_Policy_Regularization(
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                reduce="none",
            ).astype(self.compute_dtype, copy=False)
            ppo_loss = ppo_loss + float(kl_coeff) * kl
        if reduce == "none":
            return ppo_loss.astype(self.dtype, copy=False)
        if reduce == "mean":
            return cp.mean(ppo_loss).astype(self.compute_dtype)
        if reduce == "sum":
            return cp.sum(ppo_loss).astype(self.compute_dtype)
        raise ValueError("reduce must be one of the: 'none', 'mean', 'sum'!")

    def Policy_Loss_Option_Step(
        self,
        old_log_probs,
        new_log_probs,
        advantages,
        epsilon: float = EPSILON,
        kl_coeff: float = KULLBACK_LEIBLER_COEFF,
        old_policy = None,
        new_policy = None,
        max_kl: float | None = None,
        inplace_trust_region: bool = True,
        loss_reduce: str = "mean",
    ):
        out = {}
        out["ppo_loss"] = self.policy_loss(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            advantages=advantages,
            epsilon=epsilon,
            kl_coeff=kl_coeff,
            reduce=loss_reduce,
        )
        out["kl_penalty"] = self.Kullback_Leibler_Divergence_Policy_Regularization(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            reduce=loss_reduce,
        )
        if old_policy is not None and new_policy is not None and max_kl is not None:
            out["trusted_policy"] = self.Enforce_Trust_Region(
                old_policy=old_policy,
                new_policy=new_policy,
                max_kl=max_kl,
                inplace=inplace_trust_region,
            )
        return out
    

class Reply_Buffer:
    def __init__(
        self,
        capacity: int,
        state_shape = None,
        action_dtype = cp.int32,
        reward_dtype = cp.float16,
        seed: int | None = None,
    ):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.action_dtype = action_dtype
        self.reward_dtype = reward_dtype
        self.rng = cp.random.default_rng(seed)
        self.size = 0
        self.pos = 0
        self.states_buf = None
        self.actions_buf = None
        self.rewards_dtype = None
        self.next_states_buf = None
        self.next_actions_buf = None
        self.priorities = cp.zeros(self.capacity, dtyype=cp.float32)
        self.alias = None
        self.prob = None
        self.alias_dirty = True

    def Infer_State_Shape(
        self,
        state
    ):
        if self.state_shape is not None:
            return tuple(self.state_shape)
        state = cp.asarray(state)
        return tuple(state.shape)

    def Allocate(
        self,
        state,
        next_state
    ):
        if self.states_buf is not None:
            return
        s_shape = self.Infer_State_Shape(state)
        ns_shape = self.Infer_State_Shape(next_state)
        self.states_buf = cp.empty((self.capacity, *s_shape), dtype=cp.asarray(state).dtype)
        self.next_states_buf = cp.empty((self.capacity, *ns_shape), dtype=cp.asarray(next_state).dtype)
        self.actions_buf = cp.empty(self.capacity, dtype=self.action_dtype)
        self.rewards_buf = cp.empty(self.capacity, dtype=self.reward_dtype)
        self.next_actions_buf = cp.empty(self.capacity, dtype=self.action_dtype)

    def Add(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        priority: float | None = None
    ):
        state = cp.asarray(next_state)
        next_state = cp.asarray(next_state)
        action = cp.asarray(action, dtype=self.action_dtype)
        reward = cp.asarray(reward, dtype=self.reward_dtype)
        if next_action is None:
            next_action = action
        next_action = cp.asarray(next_action, dtype=self.action_dtype)
        done = cp.asarray(done, dtype=cp.bool_)
        self.Allocate(state, next_state)
        idx = self.pos
        self.states_buf[idx] = state
        self.actions_buf[idx] = action
        self.rewards_buf[idx] = reward
        self.next_states_buf[idx] = next_state
        self.next_actions_buf[idx] = next_action
        if hasattr(self, "done_buf"):
            self.done_buf[idx] = done
        if priority is None:
            priority = 1.0 if self.size == 0 else float(cp.max(self.priorities[:self.size]).item())
        self.priorities[idx] = cp.asarray(priority, dtype=cp.float32)
        self.alias_dirty = True
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def Add_Batch(
        self,
        states,
        actions,
        rewards,
        next_states,
        next_actions,
        priorities=None
    ):
        states = cp.asarray(states)
        actions = cp.asarray(actions)
        rewards = cp.asarray(rewards)
        next_states = cp.asarray(next_states)
        next_actions = cp.asarray(next_actions)
        n = int(states.shape[0])
        if not (actions.shape[0] == rewards.shape[0] == next_states.shape[0] == next_actions.shape[0] == n):
            raise ValueError("All input batch must have the same number of sample!")
        self.Allocate(states[0], next_states[0])
        if priorities is None:
            priorities = cp.ones(n, dtype=cp.float32)
        else:
            priorities = cp.asarray(priorities, dtype=cp.float32).ravel()
            if priorities.size != n:
                raise ValueError("Priorities must have the same length as batch!")
        for i in range(n):
            self.Add(
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                next_actions[i],
                priority=float(priorities[i].item()),
            )

    def Valid_Size(self):
        if self.size <= 0:
            raise ValueError("Reply buffer is still empty!")
        return self.size

    def Sample_Uniform(
        self,
        batch_size: int,
        replace: bool = True
    ):
        n = self.Valid_Size()
        batch_size = int(batch_size)
        if not replace and batch_size > n:
            raise ValueError("batch_size cannot be larger than buffer size if replace=False.")
        idx = self.rng.integers(0, n, size=batch_size, dtype=cp.int32) if replace else self.rng.permutation(n)[:batch_size]
        return (
            self.states_buf[idx],
            self.actions_buf[idx],
            self.rewards_buf[idx],
            self.next_states_buf[idx],
            self.next_actions_buf[idx],
            idx.astype(cp.int32),
        )

    def Build_Alias_Table(self):
        n = self.Valid_Size()
        p = cp.asarray(self.priorities[:n], dtype=cp.float32)
        total = cp.sum(p)
        if total <= 0:
            p = cp.ones(n, dtype=cp.float32) / n
        else:
            p = p / total
        prob = cp.zeros(n, dtype=cp.float32)
        alias = cp.zeros(n, dtype=cp.int32)
        p_np = cp.asnumpy(p)
        prob_np = np.zeros(n, dtype=np.float32)
        alias_np = np.zeros(n, dtype=np.int32)
        scaled = p_np * n
        small = [i for i, x in enumerate(scaled) if x < 1.0]
        large = [i for i, x in enumerate(scaled) if x >= 1.0]
        while small and large:
            s = small.pop()
            l = large.pop()
            prob_np[s] = scaled[s]
            alias_np[s] = l
            scaled[l] = scaled[l] - (1.0-scaled[s])
            if scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        for i in large:
            prob_np[i] = 1.0
            alias_np[i] = i
        for i in small:
            prob_np[i] = 1.0
            alias_np[i] = i
        self.prob = cp.asarray(prob_np, dtype=cp.float32)
        self.alias = cp.asarray(alias_np, dtype=cp.int32)
        self.alias_dirty = False

    def Sample_Prioritized_Indices(
        self,
        batch_size: int
    ):
        n = self.Valid_Size()
        batch_size = int(batch_size)
        if self.alias_dirty or self.alias is None or self.prob is None:
            self.Build_Alias_Table()
        u = self.rng.random(batch_size, dtype=cp.float32)
        v = self.rng.random(batch_size, dtype=cp.float32)
        i = cp.asarray((u*u).astype(cp.int32))
        i = cp.clip(i, 0, n-1)
        selected = cp.where(v < self.prob[i], i, self.alias[i])
        return selected.astype(cp.int32)

    def Sample_Prioritized(
        self,
        batch_size: int
    ):
        idx = self.Sample_Prioritized_Indices(batch_size)
        return(
            self.states_buf[idx],
            self.actions_buf[idx],
            self.rewards_buf[idx],
            self.next_states_buf[idx],
            self.next_actions_buf[idx],
            idx,
        )

    def Update_Priorities(
        self,
        indices,
        new_priorities
    ):
        indices = cp.asarray(indices, dtype=cp.int32).ravel()
        new_priorities = cp.asarray(new_priorities, dtype=cp.float32).ravel()
        if indices.size != new_priorities.size:
            raise ValueError("indices and new priorities must have the same length")
        self.priorities[indices] = cp.maximum(new_priorities, EPS2)
        self.alias_dirty = True

    def Get_All(self):
        n = self.Valid_Size()
        return {
            "states": self.states_buf[:n],
            "actions": self.actions_buf[:n],
            "rewards": self.rewards_buf[:n],
            "next_states": self.next_states_buf[:n],
            "next_actions": self.next_actions_buf[:n],
            "priorities": self.priorities[:n],
        }

    def Clear(self):
        self.size = 0
        self.pos = 0
        self.alias_dirty = True
        self.priorities[:] = 0

    def __len__(self):
        ret. self.size

    def run(
        self,
        mode: str,
        **kwargs
    ):
        mode = mode.lower().strip()
        if mode == "add":
            return self.add(**kwargs)
        if mode == "add_batch":
            return self.add_batch(**kwargs)
        if mode == "sample":
            return self.sample_uniform(**kwargs)
        if mode == "sample_prioritized":
            return self.sample_prioritized(**kwargs)
        if mode == "update_priorities":
            return self.update_priorities(**kwargs)
        if mode == "clear":
            return self.clear()
        raise ValueError("mode must be one of the: add, add_batch_sample, sample_prioritized, update_priorities, clear.")


class Reward_Aggregation:
    def __init__(
        self,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        eps: float = EPS
    ):
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.eps = float(eps)

    def As_2D(
        self,
        rewards
    ):
        rewards = cp.asarray(rewards, dtype=self.compute_dtype)
        if rewards.ndim == 1:
            return rewards[None,:]
        if rewards.ndim == 2:
            return rewards
        raise ValueError("rewards must be 1D or 2D!")

    def Tile_Sum_Stochasticity(
        self,
        rewards,
        tile_size: int = TILE_SIZE,
        pad_value: float = PAD_VALUE
    ):
        r = self.As_2D(rewards)
        num_eps, steps = r.shape
        tile_size = int(tile_size)
        if tile_size <= 0:
            raise ValueError("TILE_SIZE must be > 0..!")
        remainder = steps % tile_size
        if remainder != 0:
            pad = tile_size - remainder
            r = cp.pad(r, ((0,0),(0,pad)), mode="constant", constant_values=pad_value)
        num_tiles = r.shape[1] // tile_size
        tiled = r.reshape(num_eps, num_tiles, tile_size)
        tile_sums = cp.sum(tiled, axis=2)
        return tile_sums.astype(self.dtype, copy=False)

    def Episode_Sum_Stochasticity(
        self,
        tile_sums
    ):
        tile_sums = cp.asarray(tile_sums, dtype=self.compute_dtype)
        if tile_sums.ndim == 1:
            return cp.sum(tile_sums).astype(self.dtype, copy=False)
        if tile_sums.ndim != 2:
            raise ValueError("tile_sums must be 1D or 2D!")
        episode_sums = cp.sum(tile_sums, axis=1)
        return episode_sums.astype(self.dtype, copy=False)

    def Stats_Stochasticity(
        self,
        episode_sums
    ):
        episode_sums = cp.asarray(episode_sums, dtype=self.compute_dtype).ravel()
        if episode_sums.size == 0:
            raise ValueError("episode_sums must not be empty!")
        mean = cp.mean(episode_sums)
        var = cp.var(episode_sums)
        sum_ = cp.sum(episode_sums)
        sumsq = cp.sum(episode_sums*episode_sums)
        return {
            "mean": mean.astype(self.compute_dtype, copy=False),
            "var": var.astype(self.compute_dtype, copy=False),
            "sum": sum_.astype(self.compute_dtype, copy=False),
            "sumsq": sumsq.astype(self.compute_dtype, copy=False),
        }

    def Log_Episode_Reward(
        self,
        rewards
    ):
        r = self.As_2D(rewards)
        episode_rewards = cp.sum(r, axis=1)
        return episode_rewards.astype(self.dtype, copy=False)

    def Aggregate(
        self,
        rewards,
        tile_size: int = TILE_SIZE,
        pad_value: float = PAD_VALUE,
    ):
        tile_sums = self.Tile_Sum_Stochasticity(
            rewards=rewards,
            tile_size=tile_size,
            pad_value=pad_value,
        )
        episode_sums = self.Episode_Sum_Stochasticity(tile_sums)
        stats = self.Stats_Stochasticity(episode_sums)
        episode_rewards = self.Log_Episode_Reward(rewards)
        return {
            "tile_sums": tile_sums,
            "episode_sums": episode_sums,
            "stats": stats,
            "episode_rewards": episode_rewards,
        }

    def Reward_Aggregation_Step(
        self,
        rewards,
        tile_size: int = TILE_SIZE,
        pad_value: float = PAD_VALUE
    ):
        return self.Aggregate(
            rewards=rewards,
            tile_size=tile_size,
            pad_value=pad_value,
        )


class State_Action_Reward_State_Action:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        epsilon: float = EPSILON,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        seed: int | None = None,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.rng = cp.random.default_rng(seed)
        self.Q = cp.zeros((self.num_states, self.num_actions), dtype=self.dtype)

    def Select_Action(
        self,
        state
    ):
        state = cp.asarray(state, dtype=cp.int32).ravel()
        q_row = self.Q[state].astype(self.compute_dtype, copy=False)
        greedy = cp.argmax(q_row, axis=1).astype(cp.int32)
        explore = self.rng.random(state.size) < self.epsilon
        random_action = self.rng.integers(0, self.num_actions, size=state.size, dtype=cp.int32)
        return cp.where(explore, random_action, greedy).astype(cp.int32)

    def Update(
        self,
        states,
        actions,
        rewards,
        next_states,
        next_actions,
        alpha: float | None = None,
        gamma: float | None = None,
    ):
        states = cp.asarray(states, dtype=cp.int32).ravel()
        actions = cp.asarray(actions, dtype=cp.int32).ravel()
        rewards = cp.asarray(rewards, dtype=self.compute_dtype).ravel()
        next_states = cp.asarray(next_states, dtype=cp.int32).ravel()
        next_actions = cp.asarray(next_actions, dtype=cp.int32).ravel()
        if not(states.size == actions.size == rewards.size == next_states.size == next_actions.size):
            raise ValueError("All input batch must have the same length.")
        alpha = float(self.alpha if alpha is None else alpha)
        gamma = float(self.gamma if gamma is None else gamma)
        q_sa = self.Q[states,actions].astype(self.compute_dtype, copy=False)
        q_next = self.Q[next_states,next_actions].astype(self.compute_dtype, copy=False)
        td_target = rewards + gamma * q_next
        td_error = td_target - q_sa
        new_q = q_sa + alpha * td_error
        if states.size == cp.unique(states * self.num_actions + actions).size:
            self.Q[states,actions] = new_q.astype(self.dtype, copy=False)
        else:
            for i in range(states.size):
                s = int(states[i].item())
                a = int(actions[i].item())
                self.Q[s,a] = cp.asarray(new_q[i], dtype=self.dtype)
        return td_error.astype(self.dtype, copy=False)

    def Update_Single(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        alpha: float | None = None,
        gamma: float | None = None,
    ):
        s = int(cp.asarray(state, dtype=cp.int32).item())
        a = int(cp.asarray(action, dtype=cp.int32).item())
        r = cp.asarray(reward, dtype=self.compute_dtype).item()
        s2 = int(cp.asarray(next_state, dtype=cp.int32).item())
        a2 = int(cp.asarray(next_action, dtype=cp.int32).item())
        alpha = float(self.alpha if alpha is None else alpha)
        gamma = float(self.gamma if gamma is None else gamma)
        q_sa = float(cp.asarray(self.Q[s,a], dtype=self.compute_dtype).item())
        q_next = float(cp.asarray(self.Q[s,a], dtype=self.compute_dtype).item())
        td_error = r + gamma * q_next - q_sa
        self.Q[s,a] = cp.asarray(q_sa + alpha * td_error, dtype=self.dtype)
        return cp.asarray(td_error, dtype=self.dtype)

    def Train_Episode(
        self,
        states,
        actions,
        rewards,
        next_states,
        next_actions,
        alpha: float | None = None,
        gamma: float | None = None,
    ):
        return self.update(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            next_actions=next_actions,
            alpha=alpha,
            gamma=gamma,
        )

    def Get_Q(self):
        return self.Q

    def Set_Q(self, Q):
        Q = cp.asarray(Q)
        if Q.shape != (self.num_states, self.num_actions):
            raise ValueError("Shape Q doesn't match with the number of states and number of actions.")
        self.Q[...] = Q.astype(self.dtype, copy=False)
        return self.Q

    def Reset(self):
        self.Q.fill(0)
        return self.Q

    def Running_SARSA(
        self,
        mode: str,
        **kwargs
    ):
        mode = mode.lower().strip()
        if mode == "select_action":
            return self.select_action(kwargs["state"])
        if mode == "update":
            return self.update(
                kwargs["states"],
                kwargs["actions"],
                kwargs["rewards"],
                kwargs["next_states"],
                kwargs["next_actions"],
                kwargs.get("alpha", None),
                kwargs.get("gamma", None),
            )
        if mode == "single":
            return self.update_single(
                kwargs["state"],
                kwargs["action"],
                kwargs["reward"],
                kwargs["next_state"],
                kwargs["next_action"],
                kwargs.get("alpha", None),
                kwargs.get("gamma", None),
            )
        if mode == "reset":
            return self.reset()
        raise ValueError("mode must be one of the: select_action, update, single, reset!")


class Temporal_Difference:
    def __init__(
        self,
        num_states: int,
        num_ensemble: int = NUM_ENSEMBLE,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        seed: int | None = None,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.num_ensemble = int(num_ensemble)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.rng = cp.random.default_rng(seed)
        self.Q1 = cp.zeros((self.num_states, self.num_actions), dtype=self.dtype)
        self.Q2 = cp.zeros((self.num_states, self.num_actions), dtype=self.dtype)
        self.Q_ensemble = cp.zeros((self.num_ensemble, self.num_states, self.num_actions), dtype=self.dtype,)

    def As_1D_Int(
        self,
        x
    ):
        return cp.asarray(x, dtype=cp.int32).ravel()

    def As_1D_Reward(
        self,
        x
    ):
        return cp.asarray(x, dtype=self.compute_dtype).ravel()

    def As_2D_Q(
        self,
        Q
    ):
        Q = cp.asarray(Q)
        if Q.ndim != 2:
            raise ValueError("Q must be 2D shape: [number of states, number of actions]!")
        if Q.shape != (self.num_states, self.num_actions):
            raise ValueError("Shape Q doesn't match witj number of states and number of actions!")
        return Q

    def Set_Q1(
        self,
        Q1
    ):
        self.Q1[...] = self.As_2D_Q(Q1).astype(self.dtype, copy=False)
        return self.Q1

    def Set_Q2(
        self,
        Q2
    ):
        self.Q2[...] = self.As_2D_Q(Q2).astype(self.dtype, copy=False)
        return self.Q2

    def Set_Ensemble(
        self,
        Q_Ensemble
    ):
        Q_ensemble = cp.asarray(Q_Ensemble)
        expected = (self.num_ensemble, self.num_states, self.num_actions)
        if Q_ensemble.shape != expected:
            raise ValueError(f"Shape Q_Ensemble must be {expected}!")
        self.Q_ensemble[...] = Q_ensemble.astype(self.dtype, copy=False)
        return self.Q_ensemble

    def Get_Q1(self):
        return self.Q1

    def Get_Q2(self):
        return self.Q2

    def Get_Ensemble(self):
        return self.Q_ensemble

    def Double_Q_Update(
        self,
        states,
        actions,
        rewards,
        next_states,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        seed: int | None = None,
        inplace: bool = True,
    ):
        states = self.As_1D_Int(states)
        actions = self.As_1D_Int(actions)
        rewards = self.As_1D_Reward(rewards)
        next_states = self.As_1D_Int(next_states)
        n = states.size
        if not (actions.size == rewards.size == next_states.size == n):
            raise ValueError("All input batch must have the same length!")
        alpha = cp.asarray(alpha, dtype=self.compute_dtype)
        gamma = cp.asarray(gamma, dtype=self.compute_dtype)
        rng = self.rng if seed is None else cp.random.default_rng(seed)
        update1 = rng.random(n) < 0.5
        q1 = self.Q1
        q2 = self.Q2
        a_star = cp.argmax(q1[next_states].astype(self.compute_dtype, copy=False), axis=1).astype(cp.int32)
        idx = cp.arange(n, dtype=cp.int32)
        q_eval = cp.where(
            update1,
            q1[states,actions],
            q2[states,actions],
        ).astype(self.compute_dtype, copy=False)
        td = rewards + gamma * q_eval - q_cur
        new_q = q_cur + alpha * td
        if inplace:
            for i in range(n):
                s = int(states[i].item())
                a = int(actions[i].item())
                if bool(update1[i].item()):
                    q1[s,a] = cp.asarray(new_q[i], dtype=self.dtype)
                else:
                    q2[s,a] = cp.asarray(new_q[i], dtype=self.dtype)
            return td.astype(self.dtype, copy=False)
        q1_new = q1.copy()
        q2_new = q2.copy()
        for i in range(n):
            s = int(states[i].item())
            a = int(actions[i].item())
            if bool(update1[i].item()):
                q1_new[s,a] = cp.asarray(new_q[i], dtype=self.dtype)
            else:
                q2_new[s,a] = cp.asarray(new_q[i], dtype=self.dtype)
        return (
            q1_new.astype(self.dtype, copy=False),
            q2_new.astype(self.dtype, copy=False),
            td.astype(self.dtype, copy=False),
        )

    def Ensemble_Q_Update(
        self,
        states,
        actions,
        rewards,
        next_states,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        inplace: bool = True,
    ):
        states = self.As_1D_Int(states)
        actions = self.As_1D_Int(actions)
        rewards = self.As_1D_Reward(rewards)
        next_states = self.As_1D_Int(next_states)
        n = states.size
        if not (actions.size == rewards.size == next_states.size == n):
            raise ValueError("All input batch must have same length!")
        alpha = cp.asarray(alpha, dtype=self.compute_dtype)
        gamma = cp.asarray(gamma, dtype=self.compute_dtype)
        Q = self.Q_ensemble.astype(self.compute_dtype, copy=False)
        next_q = Q[:,next_states,:]
        max_vals = cp.max(next_q, axis=2)
        avg_maxQ = cp.mean(max_vals, axis=0)
        target = rewards + gamma * avg_maxQ
        oldQ = Q[:,states,actions]
        newQ = oldQ + alpha * (target[None,:] - oldQ)
        if inplace:
            for e in range(self.num_ensemble):
                self.Q_ensemble[e, states, actions] = newQ[e].astype(self.dtype, copy=False)
            return (target.astype(self.dtype, copy=False),)
        Q_new = self.Q_ensemble.copy()
        for e in range(self.num_ensemble):
            Q_new[e, states, actions] = newQ[e].astype(self.dtype, copy=False)
        return Q_new.astype(self.dtype, copy=False), target.astype(self.dtype, copy=False)

    def Munchausen_Deep_Q_Network(
        self,
        states,
        actions,
        rewards,
        next_states,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        tau: float = TAU,
        munchausen_coef: float = MUNCHAUSEN_COEF,
        log_clip_lower: float = LOG_CLIP_LOWER,
        inplace: bool = True,
    ):
        states = self.As_1D_Int(states)
        actions = self.As_1D_Int(actions)
        rewards = self.As_1D_Reward(rewards)
        next_states = self.As_1D_Int(next_states)
        n = states.size
        if not (actions.size == rewards.size == next_states.size == n):
            raise ValueError("All input batch must have same length!")
        alpha = cp.asarray(alpha, dtype=self.compute_dtype)
        gamma = cp.asarray(gamma, dtype=self.compute_dtype)
        tau = cp.asarray(tau, dtype=self.compute_dtype)
        Q0 = self.Q_ensemble[0].astype(self.compute_dtype, copy=False)
        q_curr_all = Q0[states]
        maxQ = cp.max(q_curr_all, axis=1, keepdims=True)
        exp_buffer = cp.exp((q_curr_all - maxQ) / tau)
        sum_exp = cp.sum(exp_buffer, axis=1)
        pi = exp_buffer[cp.arange(n), actions] / (sum_exp + 1e-12)
        log_pi = cp.log(cp.maximum(pi, 1e-12))
        log_pi = cp.maximum(log_pi, log_clip_lower)
        r_prime = rewards + munchausen_coef * log_pi
        Q = self.Q_ensemble.astype(self.compute_dtype, copy=False)
        next_q = Q[:,next_states,:]
        max_vals = cp.max(next_q, axis=2)
        avg_maxQ = cp.mean(max_vals, axis=0)
        target = r_prime + gamma * avg_maxQ
        oldQ = Q[:, states, actions]
        newQ = oldQ + alpha * (target[None,:] - oldQ)
        if inplace:
            for e in range(self.num_ensemble):
                self.Q_ensemble[e, states, actions] = newQ[e].astype(self.dtype, copy=False)
            return target.astype(self.dtype, copy=False)
        Q_new = self.Q_ensemble.copy()
        for e in range(self.num_ensemble):
            Q_new[e, states, actions] = newQ[e].astype(self.dtype, copy=False)
        return Q_new.astype(self.dtype, copy=False), target.astype(self.dtype, copy=False)

    def Running_Temporal_Difference(
        self,
        mode: str,
        **kwargs
    ):
        mode = mode.lower().strip()
        if mode == "double_q":
            return self.double_q_update(**kwargs)
        if mode == "ensemble_q":
            return self.Ensemble_Q_Update(**kwargs)
        if mode == "munchausen":
            return self.Munchausen_Deep_Q_Network(**kwargs)
        raise ValueError("mode must be one of the: double_q, ensemble_q, or munchausen!")


class Unified_Advantages_Signal:
    def __init__(
        self,
        num_states: int | None = None,
        num_actions: int | None = None,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        eps: float = EPS,
    ):
        self.num_states = None if num_states is None else int(num_states)
        self.num_actions = None if num_actions is None else int (num_actions)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.eps = float(eps)

    def As_1D_Int(
        self,
        x
    ):
        return cp.asarray(x, dtype=cp.int32).ravel()

    def As_1D_Float(
        self,
        x
    ):
        return cp.asarray(x, dtype=self.compute_dtype).ravel()

    def As_2D_Float(
        self,
        x
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        if x.ndim == 1:
            return x[None,:]
        if x.ndim == 2:
            return x
        raise ValueError("Input must be 1D or 2D!")

    def Compute_Advantage_Signal(
        self,
        Q,
        states,
        actions,
        rewards,
        next_states,
        next_actions,
        gamma: float,
    ):
        Q = cp.asarray(Q)
        if Q.ndim != 2:
            raise ValueError("Q must be [number of states, number of actions]!")
        states = self.As_1D_Int(states)
        actions = self.As_1D_Int(actions)
        rewards = self.As_1D_Float(rewards)
        next_states = self.As_1D_Int(next_states)
        next_actions = self.As_1D_Int(next_actions)
        if not (states.size == actions.size == rewards.size == next_states.size == next_actions.size):
            raise ValueError("All input batch must have same length!")
        gamma = cp.asarray(gamma, dtype=self.compute_dtype)
        q_sa = Q[states,actions].astype(self.compute_dtype,copy=False)
        q_s2a2 = Q[next_states,next_actions].astype(self.compute_dtype,copy=False)
        advantage = rewards + gamma * q_s2a2 - q_sa
        return advantage.astype(self.dtype,copy=False)

    def Generalized_Advantage_Estimation(
        self,
        rewards,
        values,
        gamma: float,
        lambda_gae: float,
    ):
        rewards = self.As_1D_Float(rewards)
        values = self.As_1D_Float(values)
        if values.size != rewards.size + 1:
            raise ValueError("values must have the length of T+1 if rewards=T!")
        gamma = float(gamma)
        lambda_gae = float(lambda_gae)
        T = rewards.size
        deltas = rewards + gamma * values[1:] - values[:-1]
        advantages = cp.empty(T,dtype=self.compute_dtype)
        gae = cp.asarray(0.0, dtype=self.compute_dtype)
        for t in range(T-1, -1, -1):
            gae = deltas[t] + gamma * lambda_gae * gae
            advantages[t] = gae
        return advantages.astype(self.dtype, copy=False)

    def Entropy_Bonus_And_Add_Entropy_To_Advantage(
        self,
        log_probs,
        advantages,
        entropy_coef: float,
    ):
        log_probs = self.As_2D_Float(log_probs)
        advantages = self.As_1D_Float(advantages)
        if log_probs.shape[0] != advantages.size:
            raise ValueError("Number of sample log_probs and advantages must be same!")
        entropy_coef = cp.asarray(entropy_coef, dtype=self.compute_dtype)
        lp = log_probs - cp.max(log_probs, axis=1, keepdims=True)
        p = cp.exp(lp)
        entropy = -cp.sum(p*lp, axis=1)
        adv_out = advantages + entropy_coef * entropy
        return adv_out.astype(self.dtype, copy=False), entropy.astype(self.dtype, copy=False)

    def Compute_Unified(
        self,
        Q=None,
        states=None,
        actions=None,
        rewards=None,
        next_states=None,
        next_actions=None,
        gamma: float = GAMMA,
        values=None,
        lambda_gae: float = LAMBDA,
        log_probs=None,
        entropy_coef: float = ENTROPY,
        mode: str = "td",
    ):
        mode = mode.lower().strip()
        if mode == "td":
            return self.Compute_Advantage_Signal(
                Q=Q,
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                next_actions=next_actions,
                gamma=gamma,
            )
        if mode == "gae":
            if values is None:
                raise ValueError("values must be given to Generalize Advantage Estimation mode!")
            return self.Generalized_Advantage_Estimation(
                rewards=rewards,
                values=values,
                gamma=gamma,
                lambda_gae=lambda_gae,
            )
        if mode == "entropy":
            if log_probs is None or rewards is None:
                raise ValueError("log_probs and advantages must be given to Entropy mode!")
            return self.Entropy_Bonus_And_Add_Entropy_To_Advantage(
                log_probs=log_probs,
                advantages=rewards,
                entropy_coef=entropy_coef,
            )
        if mode == "full":
            if values is not None:
                adv = self.Generalized_Advantage_Estimation(
                    rewards=rewards,
                    values=values,
                    gamma=gamma,
                    lambda_gae=lambda_gae,
                )
            else:
                adv = self.Compute_Advantage_Signal(
                    Q=Q,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    next_actions=next_actions,
                    gamma=gamma,
                )
            if log_probs is not None and entropy_coef != 0.0:
                adv, entropy = self.Entropy_Bonus_And_Add_Entropy_To_Advantage(
                    log_probs=log_probs,
                    advantages=adv,
                    entropy_coef=entropy_coef,
                )
                return adv, entropy
            return adv
        raise Valueerror("mode must be one of the: td, gae, entropy, full!")


class Safe_Reply_Buffer:
    def __init__(
        self,
        capacity: int,
        seed: int | None = None
    ):
        self.capacity = int(capacity)
        self.rng = cp.random.default_rng(seed)
        self.size = 0
        self.pos = 0
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.next_actions = None
        self.dones = None
        self.priorities = cp.zeros(self.capacity, dtype=cp.float32)

    def Allocate(
        self,
        state,
        next_state
    ):
        if self.states is not None:
            return
        s = cp.asarray(state)
        s2 = cp.asarray(next_state)
        self.states = cp.empty((self.capacity,) + tuple(s.shape), dtype=s.dtype)
        self.next_states = cp.empty((self.capacity,) + tuple(s2.shape), dtype=s2.dtype)
        self.actions = cp.empty(self.capacity, dtype=cp.int32)
        self.rewards = cp.empty(self.capacity, dtype=cp.float32)
        self.next_actions = cp.empty(self.capacity, dtype=cp.int32)
        self.dones = cp.empty(self.capacity, dtype=cp.bool_)

    def Add(
        self,
        state,
        action,
        reward,
        next_state,
        next_action=0,
        done=False,
        priority=None
    ):
        self.Allocate(state, next_state)
        idx = self.pos
        self.states[idx] = cp.asarray(state, dtype=self.states.dtype)
        self.actions[idx] = cp.asarray(action, dtype=cp.int32)
        self.rewards[idx] = cp.asarray(reward, dtype=cp.float32)
        self.next_states[idx] = cp.asarray(next_state, dtype=self.next_states.dtype)
        self.next_actions[idx] = cp.asarray(next_action, dtype=cp.int32)
        self.dones[idx] = cp.asarray(done, dtype=cp.bool_)
        if priority is None:
            priority = (float(cp.max(self.priorities[:self.size]).item()) if self.size > 0 else 1.0)
        # self.priorities[idx] = cp.asarray(max(float(priority), EPS2), dtype=cp.float32)
        self.priorities[idx] = cp.asarray(max(float(priority), EPS2), dtype=cp.float32,)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def Sample_Uniform(
        self,
        batch_size: int,
        replace: bool = True
    ):
        if self.size <= 0:
            raise ValueError("Replay buffer is still empty!")
        batch_size = int(batch_size)
        if replace:
            idx = self.rng.integers(0, self.size, size=batch_size, dtype=cp.int32)
        else:
            if batch_size > self.size:
                raise ValueError("Batch Size is bigger for replace=False")
            idx = self.rng.permutation(self.size)[:batch_size].astype(cp.int32)
        return self.Pack(idx)

    def Sample_Prioritized(
        self,
        batch_size: int
    ):
        if self.size <= 0:
            raise ValueError("Replay buffer is still empty!")
        batch_size = int(batch_size)
        p = cp.asarray(self.priorities[:self.size], dtype=cp.float32)
        s = cp.sum(p)
        if s <= 0:
            p = cp.ones_like(p) / p.size
        else:
            p = p / s
        cdf = cp.cumsum(p)
        u = self.rng.random(batch_size, dtype=cp.float32)
        idx = cp.searchsorted(cdf, u, side="right")
        idx = cp.clip(idx, 0, self.size - 1).astype(cp.int32)
        return self.Pack(idx)

    def Update_Priorities(
        self,
        indices,
        new_priorities
    ):
        indices = cp.asarray(indices, dtype=cp.int32).ravel()
        new_priorities = cp.asarray(new_priorities, dtype=cp.float32).ravel()
        if indices.size != new_priorities.size:
            raise ValueError("Indices and New Priorities must be the same length!")
        self.priorities[indices] = cp.maximum(new_priorities, EPS2)

    def Pack(
        self,
        idx
    ):
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "next_actions": self.next_actions[idx],
            "dones": self.dones[idx],
            "indices": idx,
        }

    def __len__(self):
        return self.size


class Forward_Propagation:
    def __init__(
        self,
        dtype=cp.float16,
        compute_dtype=cp.float32
    ):
        self.dtype = dtype
        self.compute_dtype = compute_dtype

    @staticmethod
    def Tanh_Activation(x):
        return cp.tanh(x)

    def As_2D(
        self,
        x
    ):
        x = cp.asarray(x)
        if x.ndim == 1:
            return x[None, :]
        if x.ndim == 2:
            return x
        raise ValueError("Input must be 1D or 2D")

    def Forward(
        self,
        weight_matrix,
        intput_vector,
        bias=None,
        activation: str = "tanh",
        transpose_input: bool = False,
    ):

        W = cp.asarray(weight_matrix, dtype=self.compute_dtype)
        X = self.As_2D(intput_vector).astype(self.compute_dtype, copy=False)
        if transpose_input:
            X = X.T
        if W.ndim != 2:
            raise ValueError("weight_matrix must be 2D!")
        if X.ndim != 2:
            raise ValueError("Input vector must be able to be formed into 2D!")
        if X.shape[1] == W.shape[1]:
            Y = X @ W.T
        elif X.shape[1] == W.shape[0]:
            Y = X @ W
        else:
            raise ValueError(f"Shape is not match: Weight Matrix={W.shape}, Input Vector={X.shape}")
        if bias is not None:
            b = cp.asarray(bias, dtype=self.compute_dtype)
            if b.ndim == 1:
                b = b[None, :]
            Y = Y + b
        activation = activation.lower().strip()
        if activation == "tanh":
            Y = self.Tanh_Activation(Y)
        elif activation == "relu":
            Y = cp.maximum(Y, 0)
        elif activation == "sigmoid":
            Y = 1.0 / (1.0 + cp.exp(-Y))
        elif activation == "linear":
            pass
        else:
            raise ValueError("Activation must be one of the: tanh, relu, sigmoid, linear!")
        Y = Y.astype(self.dtype, copy=False)
        if cp.asarray(intput_vector).ndim == 1 and Y.shape[0] == 1:
            return Y[0]
        return Y

    def Forward_Single_Layer(
        self,
        d_weight_matrix,
        d_input_vector,
        d_output_vector=None,
        activation: str = "tanh",
    ):
        y = self.Forward(
            weight_matrix=d_weight_matrix,
            input_vector=d_input_vector,
            activation=activation,
        )
        if d_output_vector is not None:
            d_output_vector = cp.asarray(d_output_vector)
            d_output_vector[...] = y.astype(d_output_vector.dtype, copy=False)
            return d_output_vector
        return y

    def Running_Forward_Propagation(
        self,
        weight_matrix,
        input_vector,
        **kwargs
    ):
        return self.Forward(
            weight_matrix,
            input_vector,
            input_vector,
            **kwargs
        )


class PDE_Residual_Loss:
    def __init__(
        self,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        eps: float = EPS2,
    ):
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.eps = float(eps)

    def As_2D(
        self,
        x,
        name: str
    ):
        x = cp.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"{name} must be 2D!")
        return x

    def Compute_Residual(
        self,
        d_model_output,
        d_initial_model,
        d_source_term,
        return_computed: bool = False,
    ):
        A = self.As_2D(d_model_output, "d_model_output").astype(self.compute_dtype, copy=False)
        B = self.As_2D(d_initial_model, "d_initial_model").astype(self.compute_dtype, copy=False)
        F = self.As_2D(d_source_term, "d_source_term").astype(self.compute_dtype, copy=False)
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"Dimension is not match for matrix multiplication: "
                f"d_Model_Output.shape={A.shape}, d_Initial_Model.shape={B.shape}!"
            )
        computed = A @ B
        if computed.shape != F.shape:
            raise ValueError(f"Shape computed={computed.shape} must be same as d_Source_Term={F.shape}!")
        residual = computed - F
        residual = residual.astype(self.dtype, copy=False)
        if return_computed:
            return residual, computed.astype(self.dtype, copy=False)
        return residual

    def Compute_Loss(
        self,
        residual,
        reduction: str = "sum",
    ):
        residual = cp.asarray(residual, dtype=self.compute_dtype)
        reduction = reduction.lower().strip()
        sq = residual * residual
        if reduction == "sum":
            return cp.sum(sq).astype(self.compute_dtype)
        if reduction == "mean":
            return cp.mean(sq).astype(self.compute_dtype)
        if reduction == "none":
            return sq.astype(self.dtype, copy=False)
        raise ValueError("reduction must be one of the: 'sum', 'mean', 'none'!")

    def PDE_Residual_Loss(
        self,
        d_model_output,
        d_initial_model,
        d_source_term,
        d_gradient=None,
        d_loss=None,
        reduction: str = "sum",
        return_computed: bool = False,
    ):
        residual, computed = self.Compute_Residual(
            d_model_output=d_model_output,
            d_initial_model=d_initial_model,
            d_source_term=d_source_term,
            return_computed=True,
        )
        if d_gradient is not None:
            d_gradient = cp.asarray(d_gradient)
            if d_gradient.shape != residual.shape:
                raise ValueError("Shape d_gradient must be the same as residual!")
            d_gradient[...] = residual.astype(d_gradient.dtype, copy=False)
        loss = self.Compute_Loss(residual, reduction=reduction)
        if d_loss is not None:
            d_loss = cp.asarray(d_loss)
            if d_loss.ndim == 0:
                d_loss[...] = loss.astype(d_loss.dtype, copy=False)
            else:
                d_loss[...] = cp.asarray(loss, dtype=d_loss.dtype)
        if return_computed:
            return residual, loss, computed.astype(self.dtype, copy=False)
        return residual, loss

    def PDE_Residual_Loss_Step(
        self,
        d_model_output,
        d_initial_model,
        d_source_term,
        d_gradient=None,
        d_loss=None,
        reduction: str = "sum",
    ):
        return self.PDE_Residual_Loss(
            d_model_output=d_model_output,
            d_initial_model=d_initial_model,
            d_source_term=d_source_term,
            d_gradient=d_gradient,
            d_loss=d_loss,
            reduction=reduction,
            return_computed=False,
        )


class Backpropagation:
    def __init__(
        self,
        alpha: float = ALPHA,
        beta1: float = BETA1,
        beta2: float = BETA2,
        epsilon: float = EPSILON,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        activation: str = "sigmoid",
        seed: int | None = None,
    ):
        self.alpha = float(alpha)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.activation = activation.lower().strip()
        self.rng = cp.random.default_rng(seed)
        self.m = None
        self.v = None
        self.prev_updates = None
        self.step_count = 0

    def As_Float(
        self,
        x
    ):
        cp.asarray(x, dtype=self.compute_dtype)

    def As_Half(
        self,
        x
    ):
        return cp.asarray(x, dtype=self.dtype)

    def Ensure_2D(
        self,
        x
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        if x.ndim == 1:
            return x[None, :]
        if x.ndim != 2:
            raise ValueError("Input must be 1D or 2D!")
        return x

    def Stack_Weights(
        self,
        weights
    ):
        if isinstance(weights, (list, tuple)):
            return [cp.asarray(w, dtype=self.compute_dtype) for w in weights]
        weights = cp.asarray(weights, dtype=self.compute_dtype)
        if weights.ndim == 3:
            return [weights[i].copy() for i in range(weights.shape[0])]
        if weights.ndim == 2:
            return [weights.copy()]
        raise ValueError("Weights must be list/tuple, 2D, or 3D array!")

    def Restore_Weights(
        self,
        weights_list,
        template
    ):
        if isinstance(templace, (list, tuple)):
            return  weights_list
        if cp.asarray(template).ndim == 3:
            return cp.stack(weights_list, axis=0)
        return weights_list[0]

    def Activation(
        self,
        x
    ):
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + cp.exp(-x))
        if self.activation == "tanh":
            return cp.tanh(x)
        if self.activation == "relu":
            return cp.maximum(x, 0.0)
        if self.activation == "linear":
            return x
        raise ValueError("Activation must be one of: sigmoid, tanh, relu, linear!")

    def Activation_Derivative(
        self,
        a
    ):
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        if self.activation == "tanh":
            return 1.0 - a * a
        if self.activation == "relu":
            return (a > 0).astype(self.compute_dtype)
        if self.activation == "linear":
            return cp.ones_like(a, dtype=self.compute_dtype)
        raise ValueError("Activation must be one of the: sigmoid, tanh, relu, linear!")

    def Compute_Total_Gradient(
        self,
        data_loss,
        residual_loss,
        model_input,
        observed_data,
        reduce: str = "sum",
    ):
        _ = data_loss
        x = self.As_Float(model_input)
        y = self.As_Float(observed_data)
        if x.shape != y.shape:
            raise ValueError("Model input and Observed Data must have the same shape!")
        residual_loss = self.As_Float(residual_loss)
        grad_data = x - y
        grad_residual = residual_loss * x
        grad_total = grad_data + grad_residual
        if reduce == "none":
            return grad_total.astype(self.dtype, copy=False)
        if reduce == "mean":
            return cp.mean(grad_total).astype(self.compute_dtype)
        if reduce == "sum":
            return cp.sum(grad_total).astype(self.compute_dtype)
        raise ValueError("reduce must be one of the: 'none', 'mean', 'sum'!")

    def Momentum_Gradient_Descent(
        self,
        gradients,
        momentum=None
    ):
        g = self.As_Float(gradients)
        if momentum is None:
            momentum = cp.zeros_like(g)
        else:
            momentum = self.As_Float(momentum)
        m_t = self.beta1 * momentum + (1.0 - self.beta1) * g
        return m_t.astype(self.dtype, copy=False)

    def Moving_Average(
        self,
        gradients,
        moving_avg=None
    ):
        g = self.As_Float(gradients)
        g2 = g * g
        if moving_avg is None:
            moving_avg = cp.zeros_like(g)
        else:
            moving_avg = self.As_Float(moving_avg)
        v_t = self.beta1 * moving_avg + (1.0 - self.beta1) * g2
        rms_denom = cp.sqrt(v_t) + self.epsilon
        return v_t.astype(self.dtype, copy=False), rms_denom.astype(self.compute_dtype, copy=False)

    def RMSProp_Update(
        self,
        params,
        gradients,
        moving_avg=None,
        lr=None
    ):
        p = self.As_Float(params)
        g = self.As_Float(gradients)
        v_t, rms_denom = self.Moving_Average(g, moving_avg)
        lr = self.alpha if lr is None else float(lr)
        p_new = p - lr * g / rms_denom
        return p_new.astype(self.dtype, copy=False), v_t

    def Adaptive_Learning_Rate_Adam(
        self,
        dt,
        gradients,
        moving_avg,
        rmsprop,
        momentum,
        alpha=None,
    ):
        g = self.As_Float(gradients)
        m = self.As_Float(momentum)
        v = self.As_Float(rmsprop)
        alpha = self.alpha if alpha is None else float(alpha)
        dt = float(dt)
        m_t = self.beta1 * m + (1.0 - self.beta1) * g
        v_t = self.beta2 * v + (1.0 - self.beta2) * (g * g)
        m_hat = m_t / (1.0 - self.beta1 ** dt + self.epsilon)
        v_vat = v_t / (1.0 - self.beta2 ** dt + self.epsilon)
        update = m_hat / (cp.sqrt(v_hat) + self.epsilon)
        alpha_new = alpha - self.alpha * update
        return (
            m_t.astype(self.dtype, copy=False),
            v_t.astype(self.dtype, copy=False),
            alpha_new.astype(self.dtype, copy=False),
            update.astype(self.dtype, copy=False),
        )

    def PDE_Residual_Loss(
        self,
        model_input,
        initial_model,
        observed_data
    ):
        x = self.As_Float(model_input)
        a0 = self.As_Float(initial_model)
        y = self.As_Float(observed_data)
        try:
            computed = x @ a0
        except Exception:
            if x.shape != y.shape:
                raise ValueError("If matrix multiplication cannot be performed, the input model and data observations must have the same shape!")
            computed = x
        if computed.shape != y.shape:
            raise ValueError("Shape residual results must be the same as observed data!")
        residual = computed - y
        loss = cp.sum(residual * residual)
        return residual.astype(self.dtype, copy=False), loss.astype(self.compute_dtype)

    def Initialize_Optimizer_State(
        self,
        weights
    ):
        w_list = self.Stack_Weights(weights)
        self.m = [cp.zeros_like(w, dtype=self.compute_dtype) for w in w_list]
        self.v = [cp.zeros_like(w, dtype=self.compute_dtype) for w in w_list]
        self.prev_updates = [cp.zeros_like(w, dtype=self.compute_dtype) for w in w_list]
        self.step_count = 0

    def Backpropagation_With_Gradient_Descent(
        self,
        weights,
        activations,
        observed_data,
        initial_model=None,
        pde_target=None,
        use_adam: bool = True,
        use_momentum: bool = True,
        residual_weight: float = RESIDUAL_WEIGHT,
        learning_rate: float | None = None,
        return_gradients: bool = False,
    ):
        W = self.Stack_Weights(weights)
        A = [self.Ensure_2D(a).astype(self.compute_dtype, copy=False) for a in activations]
        Y = self.Ensure_2D(observed_data).astype(self.compute_dtype, copy=False)
        if len(A) != len(W) + 1:
            raise ValueError("activations length must be the same as number of layer weight + 1!")
        batch = A[-1].shape[0]
        if Y.shape != A[-1].shape:
            raise ValueError("Observed Data must have the same shape as output activations!")
        self.step_count += 1
        delta = (A[-1] - Y) * self.activation_derivative(A[-1])
        pde_loss = None
        pde_residual = None
        if residual_weight != 0.0 and initial_model is not None and pde_target is not None:
            pde_residual, pde_loss = self.PDE_Residual_Loss(A[-1], initial_model, pde_target)
            pde_residual = self.Ensure_2D(pde_residual).astype(self.compute_dtype, copy=False)
            delta = delta + float(residual_weight) * pde_residual * self.Activation_Derivative(A[-1])
        grads = [None] * len(W)
        for l in range(len(W) - 1, -1, -1):
            grads[l] = (delta.T @ A[l]) / max(batch, 1)
            if l > 0:
                delta = (delta @ W[l]) * self.Activation_Derivative(A[l])
        if self.m is None or self.v is None or self.prev_updates is None:
            self.Initialize_Optimizer_State(W)
        lr = self.alpha if learning_rate is None else float(learning_rate)
        update_weights = []
        for i, w in enumerate(W):
            g = grads[i].astype(self.compute_dtype, copy=False)
            if use_adam:
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
                m_hat = self.m[i] / (1.0 - self.beta1 ** self.step_count + self.epsilon)
                v_hat = self.v[i] / (1.0 - self.beta2 ** self.step_count + self.epsilon)
                update = lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)
            else:
                if use_momentum:
                    self.prev_updates[i] = self.beta1 * self.prev_updates[i] + (1.0 - self.beta1) * g
                    update = lr * self.prev_updates[i]
                else:
                    update = lr * g
            w_new = w - update
            updated_weights.append(w_new.astype(self.dtype, copy=False))
        weights_out = self.Restore_Weights(update_weights, weights)
        if return_gradients:
            out = {
                "weights": weights_out,
                "gradients": [g.astype(self.dtype, copy=False) for g in grads],
                "output_delta": delta.astype(self.dtype, copy=False)
            }
            if pde_loss is not None:
                out["pde_loss"] = pde_loss.astype(self.compute_dtype, copy=False)
                out["pde_residual"] = pde_residual
            return out
        return weights_out

    def Backpropagation_Step(
        self,
        **kwargs
    ):
        return self.Backpropagation_With_Gradient_Descent(**kwargs)


class Global_Convergence_Update:
    def __init__(
        self,
        beta1: float = BETA1,
        epsilon: float = EPSILON,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
        dtype=cp.float16,
        compute_dtype=cp.float32,
    ):
        self.beta1 = float(beta1)
        self.epsilon = float(epsilon)
        self.convergence_threshold = float(convergence_threshold)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.v = None
        self.convergence_flag = cp.array(0, dtype=cp.int32)
        self.last_global_grad_norm = None

    def As_Array(
        self,
        x
    ):
        return cp.asarray(x)

    def Ensure_State(
        self,
        weights
    ):
        w = cp.asarray(weights)
        if self.v is None or self.v.shape != w.shape:
            self.v = cp.zeros_like(w, dtype=self.compute_dtype)

    def Compute_Global_Gradient_Norm(
        self,
        gradients
    ):
        g = cp.asarray(gradients, dtype=self.compute_dtype)
        return cp.sqrt(cp.sum(g*g) + self.epsilon)

    def Update(
        self,
        weights,
        gradients,
        alpha,
        return_state: bool = False,
    ):
        weights = cp.asarray(weights)
        gradients = cp.asarray(gradients, dtype=self.compute_dtype)
        if weights.shape != gradients.shape:
            raise ValueError("weights and gradients must have the same shape!")
        self.Ensure_State(weights)
        alpha = cp.asarray(alpha, dtype=self.compute_dtype)
        g2 = gradients * gradients
        self.v = self.beta1 * self.v + (1.0 - self.beta1) * g2
        denom = cp.sqrt(self.v) + self.epsilon
        update = alpha * gradients / denom
        updated_weights = weights.astype(self.compute_dtype, copy=False) - update
        weights[...] = updated_weights.astype(weights.dtype, copy=False)
        global_grad_norm = self.Compute_Global_Gradient_Norm(gradients)
        self.last_global_grad_norm = global_grad_norm
        converged = global_grad_norm <  self.convergence_threshold
        self.convergence_flag[...] = cp.asarray(converged, dtype=cp.int32)
        if return_state:
            return (
                weights,
                self.v.astype(self.dtype, copy=False),
                self.convergence_flag,
                global_grad_norm.astype(self.compute_dtype, copy=False),
            )
        return weights

    def Step(
        self,
        weights,
        gradients,
        alpha
    ):
        return self.update(weights, gradients, alpha, return_state=False)

    def Is_Converged(self):
        return bool(self.convergence_flag.item())

    def Reset(self):
        self.v = None
        self.convergence_flag[...] = 0
        self.last_global_grad_norm = None


class Transformer_Backbone:
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int = NUMBER_OF_LAYERS,
        ff_dim: int | None = None,
        output_dim: int | None = None,
        max_seq_len: int = MAXIMUM_SEQUENCE_LEN,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        seed: int | None = None,
        layer_norm_eps: float = LAYER_NORM_EPS,
    ):
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads!")
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ff_dim = int(ff_dim) if ff_dim is not None else int(4 * model_dim)
        self.output_dim = None if output_dim is None else int(output_dim)
        self.max_seq_len = int(max_seq_len)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.layer_norm_eps = float(layer_norm_eps)
        self.head_dim = self.model_dim // self.num_heads
        self.rng = cp.random.default_rng(seed)
        self._init_parameters()

    def Xavier(
        self,
        fan_in: int,
        fan_out: int,
        shape
    ):
        limit = cp.sqrt(cp.asarray(6.0 / (fan_in + fan_out), dtype=self.compute_dtype))
        return self.rng.uniform(
            low=-float(limit.item()),
            high=float(limit.item()),
            size=shape,
        ).astype(self.dtype)

    def _init_layer_norm(self):
        gamma = cp.ones((self.model_dim,), dtype=self.dtype)
        beta = cp.zeros((self.model_dim,), dtype=self.dtype)
        return gamma, beta

    def _init_parameters(self):
        self.W_in = self.Xavier(self.input_dim, self.model_dim, (self.model_dim, self.input_dim))
        self.b_in = cp.zeros((self.model_dim,), dtype=self.dtype)
        self.pos_emb = self.Xavier(self.max_seq_len, self.model_dim, (self.max_seq_len, self.model_dim))
        # Transformer blocks
        self.W_q = []
        self.b_q = []
        self.W_k = []
        self.b_k = []
        self.W_v = []
        self.b_v = []
        self.W_o = []
        self.b_o = []
        self.W_ff1 = []
        self.b_ff1 = []
        self.W_ff2 = []
        self.b_ff2 = []
        self.ln1_gamma = []
        self.ln1_beta = []
        self.ln2_gamma = []
        self.ln2_beta = []
        for _ in range(self.num_layers):
            self.W_q.append(self.Xavier(self.model_dim, self.model_dim, (self.model_dim, self.model_dim)))
            self.b_q.append(cp.zeros((self.model_dim,), dtype=self.dtype))
            self.W_k.append(self.Xavier(self.model_dim, self.model_dim, (self.model_dim, self.model_dim)))
            self.b_k.append(cp.zeros((self.model_dim,), dtype=self.dtype))
            self.W_v.append(self.Xavier(self.model_dim, self.model_dim, (self.model_dim, self.model_dim)))
            self.b_v.append(cp.zeros((self.model_dim,), dtype=self.dtype))
            self.W_o.append(self.Xavier(self.model_dim, self.model_dim, (self.model_dim, self.model_dim)))
            self.b_o.append(cp.zeros((self.model_dim,), dtype=self.dtype))
            self.W_ff1.append(self.Xavier(self.model_dim, self.ff_dim, (self.ff_dim, self.model_dim)))
            self.b_ff1.append(cp.zeros((self.ff_dim,), dtype=self.dtype))
            self.W_ff2.append(self.Xavier(self.ff_dim, self.model_dim, (self.model_dim, self.ff_dim)))
            self.b_ff2.append(cp.zeros((self.model_dim,), dtype=self.dtype))
            g1, b1 = self._init_layer_norm()
            g2, b2 = self._init_layer_norm()
            self.ln1_gamma.append(g1)
            self.ln1_beta.append(b1)
            self.ln2_gamma.append(g2)
            self.ln2_beta.append(b2)
        if self.output_dim is not None:
            self.W_out = self.Xavier(self.model_dim, self.output_dim, (self.output_dim, self.model_dim))
            self.b_out = cp.zeros((self.output_dim,), dtype=self.dtype)
        else:
            self.W_out = None
            self.b_out = None

    def As_3D(
        self,
        x
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        if x.ndim == 1:
            if x.size != self.input_dim:
                raise ValueError(f"1D input size {x.size} does not match input_dim={self.input_dim}")
            return x.reshape(1, 1, -1)
        if x.ndim == 2:
            if x.shape[1] == self.input_dim:
                return x[None, :, :]
            if x.size == self.input_dim:
                return x.reshape(1, 1, self.input_dim)
            raise ValueError(f"2D input shape {x.shape} is incompatible with input_dim={self.input_dim}")
        if x.ndim == 3:
            return x
        raise ValueError("Input must be 1D [D], 2D [T, D], or 3D [B, T, D].")

    def Gelu(
        self,
        x
    ):
        return 0.5 * x * (1.0 + cp.tanh(cp.sqrt(cp.asarray(2.0 / cp.pi, dtype=self.compute_dtype)) * (x + 0.044715 * x * x * x)))

    def Layer_Norm(
        self,
        x,
        gamma,
        beta
    ):
        mean = cp.mean(x, axis=-1, keepdims=True)
        var = cp.var(x, axis=-1, keepdims=True)
        y = (x - mean) / cp.sqrt(var + self.layer_norm_eps)
        return y * gamma[None, None, :] + beta[None, None, :]

    def Linear(
        self,
        x,
        W,
        b
    ):
        return cp.matmul(x, W.T.astype(self.compute_dtype, copy=False)) + b[None, :]

    def Split_Heads(
        self,
        x
    ):
        B, T, D = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def Merge_Heads(
        self,
        x
    ):
        B, H, T, Hd = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Hd)

    def Scaled_Dot_Product_Attention(
        self,
        q,
        k,
        v,
        mask=None
    ):
        scale = cp.asarray(1.0 / cp.sqrt(cp.asarray(self.head_dim, dtype=self.compute_dtype)), dtype=self.compute_dtype,)
        scores = cp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            scores = cp.where(mask, scores, cp.asarray(-1e9, dtype=self.compute_dtype))
        scores = scores - cp.max(scores, axis=-1, keepdims=True)
        attn = cp.exp(scores)
        attn = attn / (cp.sum(attn, axis=-1, keepdims=True) + EPS2)
        out = cp.matmul(attn, v)
        return out, attn

    def Causal_Mask(
        self,
        T,
        batch_size=BATCH_SIZE
    ):
        mask = cp.tril(cp.ones((T, T), dtype=cp.bool_))
        return mask[None, None, :, :]

    def Forward_Core(
        self,
        x,
        mask=None,
        causal: bool = False,
        return_attention: bool = False,
        return_hidden: bool = False,
        apply_output_head: bool = True,
    ):
        x = self.As_3D(x)
        B, T, _ = x.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}.")
        h = self.Linear(x, self.W_in, self.b_in)
        h = h + self.pos_emb[None, :T, :].astype(self.compute_dtype, copy=False)
        attn_cache = []
        hidden_cache = [h]
        attn_mask = None
        if causal:
            attn_mask = self.Causal_Mask(T)
        if mask is not None:
            m = cp.asarray(mask)
            if m.ndim == 2:
                m = m[None, None, :, :]
            elif m.ndim == 3:
                m = m[:, None, :, :]
            attn_mask = m if attn_mask is None else (attn_mask & m)
        for l in range(self.num_layers):
            x_norm = self.Layer_Norm(
                h,
                self.ln1_gamma[l].astype(self.compute_dtype, copy=False),
                self.ln1_beta[l].astype(self.compute_dtype, copy=False)
            )
            q = self.Linear(x_norm, self.W_q[l], self.b_q[l])
            k = self.Linear(x_norm, self.W_k[l], self.b_k[l])
            v = self.Linear(x_norm, self.W_v[l], self.b_v[l])
            q = self.Split_Heads(q)
            k = self.Split_Heads(k)
            v = self.Split_Heads(v)
            attn_out, attn = self.Scaled_Dot_Product_Attention(q, k, v, mask=attn_mask)
            attn_cache.append(attn)
            attn_out = self.Merge_Heads(attn_out)
            attn_out = self.Linear(attn_out, self.W_o[l], self.b_o[l])
            h = h + attn_out
            x_norm2 = self.Layer_Norm(
                h,
                self.ln2_gamma[l].astype(self.compute_dtype, copy=False),
                self.ln2_beta[l].astype(self.compute_dtype, copy=False)
            )
            ff = self.Linear(x_norm2, self.W_ff1[l], self.b_ff1[l])
            ff = self.Gelu(ff)
            ff = self.Linear(ff, self.W_ff2[l], self.b_ff2[l])
            h = h + ff
            hidden_cache.append(h)
        out = h
        if apply_output_head and self.W_out is not None:
            out = self.Linear(out, self.W_out, self.b_out)
        out = out.astype(self.dtype, copy=False)
        aux = {}
        if return_attention:
            aux["attention"] = attn_cache
        if return_hidden:
            aux["hidden"] = hidden_cache
        if return_attention or return_hidden:
            return out, aux
        return out

    def Forward(
        self,
        x,
        mask=None,
        causal: bool = False,
        return_attention: bool = False,
        return_hidden: bool = False,
    ):
        return self.Forward_Core(
            x,
            mask=mask,
            causal=causal,
            return_attention=return_attention,
            return_hidden=return_hidden,
            apply_output_head=True,
        )

    def __call__(
        self,
        x,
        **kwargs
    ):
        return self.Forward(x, **kwargs)

    def forward(
        self,
        x,
        **kwargs
    ):
        return self.Forward(x, **kwargs)

    def Encode(
        self,
        x,
        causal: bool = False
    ):
        return self.Forward_Core(
            x,
            causal=causal,
            return_attention=False,
            return_hidden=False,
            apply_output_head=False,
        )

    def encode(
        self,
        x,
        causal: bool = False
    ):
        return self.Encode(x, causal=causal)

    def Policy_Logits(
        self,
        x,
        causal: bool = False
    ):
        if self.W_out is None:
            raise ValueError("output_dim is None. Set output_dim = number of actions for policy logits!")
        out = self.Forward_Core(
            x,
            causal=causal,
            return_attention=False,
            return_hidden=False,
            apply_output_head=True,
        )
        out = cp.asarray(out, dtype=self.compute_dtype)
        if out.ndim == 3:
            out = out[:, -1, :]
        elif out.ndim == 1:
            out = out[None, :]
        return out.astype(self.dtype, copy=False)

    def policy_logits(
        self,
        x,
        causal: bool = False
    ):
        return self.Policy_Logits(x, causal=causal)

    def Value_Features(
        self,
        x,
        causal: bool = False
    ):
        return self.Encode(x, causal=causal)

    def value_features(
        self,
        x,
        causal: bool = False
    ):
        return self.Value_Features(x, causal=causal)

    def Parameters(self):
        params = [self.W_in, self.b_in, self.pos_emb]
        for l in range(self.num_layers):
            params.extend([
                self.W_q[l], self.b_q[l],
                self.W_k[l], self.b_k[l],
                self.W_v[l], self.b_v[l],
                self.W_o[l], self.b_o[l],
                self.W_ff1[l], self.b_ff1[l],
                self.W_ff2[l], self.b_ff2[l],
                self.ln1_gamma[l], self.ln1_beta[l],
                self.ln2_gamma[l], self.ln2_beta[l],
            ])
        if self.W_out is not None:
            params.extend([self.W_out, self.b_out])
        return params

    def State_Dict(self):
        return {
            "input_dim": self.input_dim,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "output_dim": self.output_dim,
            "max_seq_len": self.max_seq_len,
            "dtype": self.dtype,
            "compute_dtype": self.compute_dtype,
            "W_in": self.W_in.copy(),
            "b_in": self.b_in.copy(),
            "pos_emb": self.pos_emb.copy(),
            "layers": [
                {
                    "W_q": self.W_q[l].copy(), "b_q": self.b_q[l].copy(),
                    "W_k": self.W_k[l].copy(), "b_k": self.b_k[l].copy(),
                    "W_v": self.W_v[l].copy(), "b_v": self.b_v[l].copy(),
                    "W_o": self.W_o[l].copy(), "b_o": self.b_o[l].copy(),
                    "W_ff1": self.W_ff1[l].copy(), "b_ff1": self.b_ff1[l].copy(),
                    "W_ff2": self.W_ff2[l].copy(), "b_ff2": self.b_ff2[l].copy(),
                    "ln1_gamma": self.ln1_gamma[l].copy(), "ln1_beta": self.ln1_beta[l].copy(),
                    "ln2_gamma": self.ln2_gamma[l].copy(), "ln2_beta": self.ln2_beta[l].copy(),
                }
                for l in range(self.num_layers)
            ],
            "W_out": None if self.W_out is None else self.W_out.copy(),
            "b_out": None if self.b_out is None else self.b_out.copy(),
        }

    def Load_State_Dict(
        self,
        state
    ):
        self.input_dim = int(state["input_dim"])
        self.model_dim = int(state["model_dim"])
        self.num_heads = int(state["num_heads"])
        self.num_layers = int(state["num_layers"])
        self.ff_dim = int(state[ff_dim])
        self.output_dim = None if state["output_dim"] is None else int(state["output_dim"])
        self.max_seq_len = int(state["max_seq_len"])
        self.dtype = state["dtype"]
        self.compute_dtype = state["compute_dtype"]
        self.head_dim = self.model_dim // self.num_heads
        self.W_in = cp.asarray(state["W_in"], dtype=self.dtype)
        self.b_in = cp.asarray(state["b_in"], dtype=self.dtype)
        self.pos_emb = cp.asarray(state["pos_emb"], dtype=self.dtype)
        self.W_q, self.b_q = [], []
        self.W_k, self.b_k = [], []
        self.W_v, self.b_v = [], []
        self.W_o, self.b_o = [], []
        self.W_ff1, self.b_ff1 = [], []
        self.W_ff2, self.b_ff2 = [], []
        self.ln1_gamma, self.ln1_beta = [], []
        self.ln2_gamma, self.ln2_beta = [], []
        for layer in state["layers"]:
            self.W_q.append(cp.asarray(layer["W_q"], dtype=self.dtype))
            self.b_q.append(cp.asarray(layer["b_q"], dtype=self.dtype))
            self.W_k.append(cp.asarray(layer["W_k"], dtype=self.dtype))
            self.b_k.append(cp.asarray(layer["b_k"], dtype=self.dtype))
            self.W_v.append(cp.asarray(layer["W_v"], dtype=self.dtype))
            self.b_v.append(cp.asarray(layer["b_v"], dtype=self.dtype))
            self.W_o.append(cp.asarray(layer["W_o"], dtype=self.dtype))
            self.b_o.append(cp.asarray(layer["b_o"], dtype-self.dtype))
            self.W_ff1.append(cp.asarray(layer["W_ff1"], dtype=self.dtype))
            self.b_ff1.append(cp.asarray(layer["b_ff1"], dtype=self.dtype))
            self.W_ff2.append(cp.asarray(layer["W_ff2"], dtype=self.dtype))
            self.b_ff2.append(cp.asarray(layer["b_ff2"], dtype=self.dtype))
            self.ln1_gamma.append(cp.asarray(layer["ln1_gamma"], dtype=self.dtype))
            self.ln1_beta.append(cp.asarray(layer["ln1_beta"], dtype=sel.dtype))
            self.ln2_gamma.append(cp.asarray(layer["ln2_gamma"], dtype=self.dtype))
            self.ln2_beta.append(cp.asarray(layer["ln2_beta"], dtype=self.dtype))
        self.W_out = None if state["W_out"] is None else cp.asarray(state["W_out"], dtype=self.dtype)
        self.b_out = None if state["b_out"] is None else cp.asarray(state["b_out"], dtype=self.dtype)
        return self



class Vectorized_Environment:
    def __init__(
        self,
        env_factory,
        num_envs: int,
        state_shape,
        action_space_dim: int | None = None,
        dtype=cp.float32,
        seed: int | None = None,
        normalize_states: bool = False,
        normalize_rewards: bool = False,
        clip_reward: float | None = None,
    ):
        self.env_factory = env_factory
        self.num_envs = int(num_envs)
        self.state_shape = tuple(state_shape) if state_shape is not None else None
        self.action_space_dim = None if action_space_dim is None else int(action_space_dim)
        self.dtype = dtype
        self.seed = seed
        self.normalize_states = bool(normalize_states)
        self.normalize_rewards = bool(normalize_rewards)
        self.clip_reward = clip_reward
        self.rng = cp.random.default_rng(seed)
        self.envs = [self.Make_Environment(i) for i in range(self.num_envs)]
        self.state_mean = None
        self.state_std = None
        self.reward_mean = None
        self.reward_std = None
        self.last_states = None
        self.last_infos = None

    def Make_Environment(
        self,
        idx: int
    ):
        env = self.env_factory(idx) if callable(self.env_factory) else self.env_factory()
        if self.seed is not None and hasattr(env, "reset"):
            try:
                env.reset(seed=self.seed + idx)
            except TypeError:
                pass
        return env

    def As_Batch_States(
        self,
        states
    ):
        states = cp.asarray(states, dtype=self.dtype)
        if states.ndim == len(self.state_shape):
            states = states[None, ...]
        return states

    def Normalize_Batch_States(
        self,
        states
    ):
        if not self.normalize_states:
            return states
        states = cp.asarray(states, dtype=self.dtype)
        if self.state_mean is None or self.state_std is None:
            self.state_mean = cp.mean(states, axis=0, keepdims=True)
            self.state_std = cp.std(states, axis=0, keepdims=True) + EPS2
        return (states - self.state_mean) / self.state_std

    def Normalize_Batch_Rewards(
        self,
        rewards
    ):
        rewards = cp.asarray(rewards, dtype=self.dtype).ravel()
        if not self.normalize_rewards:
            return rewards
        if self.reward_mean is None or self.reward_std is None:
            self.reward_mean = cp.mean(rewards)
            self.reward_std = cp.std(rewards) + EPS2
        rewards = (rewards - self.reward_mean) / self.reward_std
        if self.clip_reward is not None:
            rewards = cp.clip(rewards, -float(self.clip_reward), float(self.clip_reward))
        return rewards

    def Reset(self):
        states = []
        infos = []
        for env in self.envs:
            out = env.reset()
            if isinstance(out, tuple) and len(out) == 2:
                s, info = out
            else:
                s, info = out, {}
            states.append(s)
            infos.append(info)
        states = cp.asarray(states, dtype=self.dtype)
        states = self.Normalize_Batch_States(states)
        self.last_states = states
        self.last_infos = infos
        return states, infos

    def Reset_At(
        self,
        idx: int
    ):
        out = self.envs[int(idx)].reset()
        if isinstance(out, tuple) and len(out) == 2:
            state, info = out
        else:
            state, info = out, {}
        state = cp.asarray(state, dtype=self.dtype)
        if self.normalize_states:
            state = self.Normalize_Batch_States(state[None, ...])[0]
        return state, info

    def Step(
        self,
        actions
    ):
        actions = cp.asarray(actions)
        if actions.ndim == 0:
            actions = actions[None]
        if actions.size != self.num_envs:
            raise ValueError(
                f"actions must have the same length as {self.num_envs}, "
                f"but accepted by {actions.size}."
            )
        next_states = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        for i, env in enumerate(self.envs):
            action_i = int(actions[i].item())
            out = env.step(action_i)
            if len(out) == 5:
                s2, r, terminated, truncated, info = out
            elif len(out) == 4:
                s2, r, done, info = out
                terminated = bool(done)
                truncated = False
            else:
                raise ValueError("environmental step output format is not recognized!")
            next_states.append(s2)
            rewards.append(r)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        next_states = cp.asarray(next_states, dtype=self.dtype)
        next_states = self.Normalize_Batch_States(next_states)
        rewards = cp.asarray(rewards, dtype=self.dtype)
        rewards = self.Normalize_Batch_Rewards(rewards)
        terminateds = cp.asarray(terminateds, dtype=cp.bool_)
        truncateds = cp.asarray(truncateds, dtype=cp.bool_)
        dones = cp.logical_or(terminateds, truncateds)
        self.last_states = next_states
        self.last_infos = infos
        return next_states, rewards, terminateds, truncateds, infos

    def Sample_Actions(
        self,
        policy_fn
    ):
        if self.last_states is None:
            raise ValueError("Environment has not been reset")
        actions = policy_fn(self.last_states)
        actions = cp.asarray(actions)
        if actions.size != self.num_envs:
            raise ValueError("policy_fn must return actions with length num_envs")
        return actions

    def Rollout(
        self,
        policy_fn,
        steps: int
    ):
        if self.last_states is None:
            self.reset()
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        for _ in range(int(steps)):
            actions = cp.asarray(policy_fn(self.last_states))
            next_states, rewards, terminateds, truncateds, infos = self.step(actions)
            trajectory["states"].append(self.last_states)
            trajectory["actions"].append(actions)
            trajectory["rewards"].append(rewards)
            trajectory["next_states"].append(next_states)
            trajectory["dones"].append(cp.logical_or(terminateds, truncateds))
            self.last_states = next_states
        return {
            k: cp.stack(v, axis=0) if len(v) > 0 else None
            for k, v in trajectory.items()
        }

    def Render(
        self,
        idx: int = 0
    ):
        if hasattr(self.envs[int(idx)], "render"):
            return self.envs[int(idx)].render()
        return None

    def Close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    def State_Dict(self):
        return {
            "num_envs": self.num_envs,
            "state_shape": self.state_shape,
            "action_space_dim": self.action_space_dim,
            "dtype": self.dtype,
            "seed": self.seed,
            "normalize_states": self.normalize_states,
            "normalize_rewards": self.normalize_rewards,
            "clip_reward": self.clip_reward,
            "state_mean": None if self.state_mean is None else self.state_mean.copy(),
            "state_std": None if self.state_std is None else self.state_std.copy(),
            "reward_mean": None if self.reward_mean is None else self.reward_mean.copy(),
            "reward_std": None if self.reward_std is None else self.reward_std.copy(),
        }

    def Load_State_Dict(
        self,
        state
    ):
        self.num_envs = int(state["num_envs"])
        self.state_shape = state["state_shape"]
        self.action_space_dim = state["action_space_dim"]
        self.dtype = state["dtype"]
        self.seed = state["seed"]
        self.normalize_states = bool(state["normalize_states"])
        self.normalize_rewards = bool(state["normalize_rewards"])
        self.clip_reward = state["clip_reward"]
        self.state_mean = None if state["state_mean"] is None else cp.asaray(state["state_mean"], dtype=self.dtype)
        self.state_std = None if state["state_std"] is None else cp.asarray(state["state_std"], dtype=self.dtype)
        self.reward_mean = None if state["reward_mean"] is None else cp.asarray(state["reward_mean"], dtype=self.dtype)
        self.reward_std = None if state["reward_std"] is None else cp.asarray(state["reward_std"], dtype=self.dtype)
        return self


class AutoDiff_PINNs:
    def __init__(
        self,
        diff_step: float = EPS3,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        eps: float = EPS2,
        cache_enabled: bool = False,
    ):
        self.diff_step = float(diff_step)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.eps = float(eps)
        self.cache_enabled = bool(cache_enabled)
        self._cache = {}

    def As_2D(
        self,
        x
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        if x.ndim == 1:
            return x[None, :]
        if x.ndim == 2:
            return x
        raise ValueError("Input must be 1D or 2D")

    def Ensure_Scalar_Output(
        self,
        y
    ):
        y = cp.asarray(y, dtype=self.compute_dtype)
        if y.ndim == 1:
            return y[:, None]
        if y.ndim == 2:
            return y
        raise ValueError("Output model must be 1D or 2D!")

    def Call_Model(
        self,
        model_fn,
        coords
    ):
        y = model_fn(coords)
        return self.Ensure_Scalar_Output(y)

    def Select_Component(
        self,
        y,
        component=0
    ):
        y = self.Ensure_Scalar_Output(y)
        if y.shape[1] == 1:
            return y[:, 0:1]
        if component is None:
            raise ValueError("For multi-dimensional output, components must be given!")
        return y[:, componen:component + 1]

    def Source_Value(
        self,
        source_term,
        coords
    ):
        if source_term is None:
            return cp.zeros((coords.shape[0], 1), dtype=self.compute_dtype)
        if callable(source_term):
            s = source_term(coords)
        else:
            s = source_term
        s = cp.asarray(s, dtype=self.compute_dtype)
        if s.ndim == 0:
            return cp.full((coords.shape[0], 1), s, dtype=self.compute_dtype)
        if s.ndim == 1:
            return s[:, None]
        if s.ndim == 2:
            return s
        raise ValueError("source_term must be scalar, 1D, 2D, or callable!")

    def Perturb(
        self,
        coords,
        dim,
        delta
    ):
        x = cp.asarray(coords, dtype=self.compute_dtype).copy()
        x[:, dim] += delta
        return x

    def Eval_Component(
        self,
        model_fn,
        coords,
        output_component=0
    ):
        y = self.Call_Model(model_fn, coords)
        return self.Select_Component(y, output_component)

    def Gradient(
        self,
        model_fn,
        coords,
        output_component=0,
        dims=None
    ):
        x = self.As_2D(coords)
        B, D = x.shape
        dims = list(range(D)) if dims is None else [int(d) for d in dims]
        h = self.diff_step
        grads = []
        for d in dims:
            xp = self.Perturb(x, d, +h)
            xm = self.Perturb(x, d, -h)
            yp = self.Eval_Component(model_fn, xp, output_component)
            ym = self.Eval_Component(model_fn, xm, output_component)
            g = (yp - ym) / (2.0 * h)
            grads.append(g)
        return cp.concatenate(grads, axis=1).astype(self.compute_dtype, copy=False)

    def Jacobian(
        self,
        model_fn,
        coords,
        dims=None
    ):
        x = self.As_2D(coords)
        y0 = self.Call_Model(model_fn, x)
        B, O = y0.shape
        D = x.shape[1]
        dims = list(range(D)) if dims is None else [int(d) for d in dims]
        jac = []
        for o in range(O):
            g = self.Gradient(model_fn, x, output_component=o, dims=dims)
            jac.append(g[:, None, :])
        return cp.concatenate(jac, axis=1).astype(self.compute_dtype, copy=False)

    def Hessian(
        self,
        model_fn,
        coords,
        output_component=0,
        dims=None,
        mixed=False
    ):
        x = self.As_2D(coords)
        B, D = x.shape
        dims = list(range(D)) if dims is None else [int(d) for d in dims]
        h = self.diff_step
        H = cp.zeros((B, D, D), dtype=self.compute_dtype)
        # Diagonal terms
        for i in dims:
            xp = self.Perturb(x, i, +h)
            xm = self.Perturb(x, i, -h)
            x0 = x
            fp = self.Eval_Component(model_fn, xp, output_component)
            f0 = self.Eval_Component(model_fn, x0, output_component)
            fm = self.Eval_Component(model_fn, xm, output_component)
            d2 = (fp - 2.0 * f0 + fm) / (h * h)
            H[:, i, i] = d2[:, 0]
        if mixed:
            for i in dims:
                for j in dims:
                    if j <= i:
                        continue
                    xpp = self.Perturb(self.Perturb(x, i, +h), j, +h)
                    xpm = self.Perturb(self.Perturb(x, i, +h), j, -h)
                    xmp = self.Perturb(self.Perturb(x, i, -h), j, +h)
                    xmm = self.Perturb(self.Perturb(x, i, -h), j, -h)
                    fpp = self.Eval_Component(model_fn, xpp, output_component)
                    fpm = self.Eval_Component(model_fn, xpm, output_component)
                    fmp = self.Eval_Component(model_fn, xmp, output_component)
                    fmm = self.Eval_Component(model_fn, xmm, output_component)
                    mixed_ij = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
                    H[:, i, j] = mixed_ij[:, 0]
                    H[:, j, i] = mixed_ij[:, 0]
        return H.astype(self.compute_dtype, copy=False)

    def Laplacian(
        self,
        model_fn,
        coords,
        output_component=0,
        space_dims=None
    ):
        x = self.As_2D(coords)
        D = x.shape[1]
        dims = list(range(D)) if space_dims is None else [int(d) for d in space_dims]
        H = self.Hessian(model_fn, x, output_component=output_component, dims=dims, mixed=False)
        lap = cp.zeros((x.shape[0], 1), dtype=self.compute_dtype)
        for d in dims:
            lap += H[:, d, d:d + 1]
        return lap

    def Time_Derivative(
        self,
        model_fn,
        coords,
        time_dim=TIME_DIM,
        output_component=0
    ):
        return self.Gradient(model_fn, coords, output_component=output_component, dims=[time_dim])

    def Second_Time_Derivative(
        self,
        model_fn,
        coords,
        time_dim=TIME_DIM,
        output_component=0
    ):
        x = self.As_2D(coords)
        h = self.diff_step
        xp = self.Perturb(x, time_dim, +h)
        xm = self.Perturb(x, time_dim, -h)
        fp = self.Eval_Component(model_fn, xp, output_component)
        f0 = self.Eval_Component(model_fn, x, output_component)
        fm = self.Eval_Component(model_fn, x, output_component)
        d2t = (fp - 2.0 * f0 + fm) / (h * h)
        return d2t.astype(self.compute_dtype, copy=False)

    def Poisson_Residual(
        self,
        model_fn,
        coords,
        source_term,
        output_component=0,
        space_dims=None
    ):
        u = self.Eval_Component(model_fn, coords, output_component)
        lap = self.Laplacian(model_fn, coords, output_component=output_component, space_dims=space_dims)
        f = self.Source_Value(source_term, self.As_2D(coords))
        return lap - f, u

    def Heat_Residual(
        self,
        model_fn,
        coords,
        alpha=ALPHA,
        source_term=None,
        output_component=0,
        time_dim=TIME_DIM,
        space_dims=None,
    ):
        x = self.As_2D(coords)
        ut = self.Time_Derivative(model_fn, x, time_dim=time_dim, output_component=output_component)
        lap = self.Laplacian(model_fn, x, output_component=output_component, space_dims=space_dims)
        f = self.Source_Value(source_term, x)
        alpha = cp.asarray(alpha, dtype=self.compute_dtype)
        residual = ut - alpha * lap - f
        return residual, self.Eval_Component(model_fn, x, output_component)

    def Wave_Residual(
        self,
        model_fn,
        coords,
        c=COURANT_NUMBER,
        source_term=None,
        output_component=0,
        time_dim=TIME_DIM,
        space_dims=None,
    ):
        x = self.As_2D(coords)
        utt = self.Second_Time_Derivative(model_fn, x, time_dim=time_dim, output_component=output_component)
        lap = self.Laplacian(model_fn, x, output_component=output_component, space_dims=space_dims)
        f = self.Source_Value(source_term, x)
        c = cp.asarray(c, dtype=self.compute_dtype)
        residual = utt - (c * c) * lap - f
        return residual, self.Eval_Component(model_fn, x, output_component)

    def Burgers_Residual(
        self,
        model_fn,
        coords,
        nu=NU,
        source_term=None,
        x_dim=X_DIM,
        time_dim=TIME_DIM,
        output_component=0,
    ):
        x = self.As_2D(coords)
        u = self.Eval_Component(model_fn, x, output_component)
        ut = self.Time_Derivative(model_fn, x, time_dim=time_dim, output_component=output_component)
        ux = self.Gradient(model_fn, x, output_component=output_component, dims=[x_dim])[:, 0:1]
        uxx = self.Hessian(model_fn, x, output_component=output_component, dims=[x_dim], mixed=False)[:, x_dim, x_dim:x_dim + 1]
        f = self.Source_Value(source_term, x)
        nu = cp.asarray(nu, dtype=self.compute_dtype)
        residual = ut + u * ux - nu * uxx - f
        return residual, u

    def Physics_Loss(
        self,
        residual,
        reduction="mean"
    ):
        residual = cp.asarray(residual, dtype=self.compute_dtype)
        sq = residual * residual
        reduction = reduction.lower().strip()
        if reduction == "mean":
            return cp.mean(sq).astype(self.compute_dtype)
        if reduction == "sum":
            return cp.sum(sq).astype(self.compute_dtype)
        if reduction == "none":
            return sq.astype(self.dtype, copy=False)
        raise ValueError("reduction must be one of the: 'mean', 'sum', 'none'")

    def Boundary_Loss(
        self,
        model_fn,
        boundary_coords,
        boundary_values,
        output_component=0,
        reduction="mean"
    ):
        pred = self.Eval_Component(model_fn, boundary_coords, output_component)
        target = cp.asarray(boundary_values, dtype=self.compute_dtype)
        if target.ndim == 1:
            target = target[:, None]
        diff = pred - target
        return self.physics_loss(diff, reduction=reduction)

    def Initial_Condition_Loss(
        self,
        model_fn,
        initial_coords,
        initial_values,
        output_component=0,
        reduction="mean"
    ):
        return self.Boundary_Loss(
            model_fn=model_fn,
            boundary_coords=initial_coords,
            boundary_values=initial_values,
            output_component=output_component,
            reduction=reduction,
        )

    def Residual_From_Operator(
        self,
        model_fn,
        coords,
        operator_fn,
        output_component=0
    ):
        x = self.As_2D(coords)
        u = self.Eval_Component(model_fn, x, output_component)

        def grad_fn(dims=None):
            return self.Gradient(model_fn, x, output_component=output_component, dims=dims)

        def hess_fn(dims=None, mixed=False):
            return self.Hessian(model_fn, x, output_component=output_component, dims=dims, mixed=mixed)

        def lap_fn(space_dims=None):
            return self.Laplacian(model_fn, x, output_component=output_component, space_dims=space_dims)

        residual = operator_fn(u, x, grad_fn, hess_fn, lap_fn)
        residual = cp.asarray(residual, dtype=self.compute_dtype)
        if residual.ndim == 1:
            residual = residual[:, None]
        return residual, u

    def Collocation_Points(
        self,
        bounds,
        n_points,
        seed=None
    ):
        bounds = cp.asarray(bounds, dtype=self.compute_dtype)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must be in the shape of [D, 2]!")
        rng = self._cache.get("rng", None)
        if seed is not None or rng is None:
            rng = cp.random.default_rng(seed)
            if self.cache_enabled:
                self._cache["rng"] = rng
        low = bounds[:, 0]
        high = bounds[:, 1]
        u = rng.random((int(n_points), bounds.shape[0]), dtype=self.compute_dtype)
        return (low + (high - low) * u).astype(self.compute_dtype, copy=False)

    def Step(
        self,
        model_fn,
        coords,
        source_term=None,
        operator="poisson",
        output_component=0,
        reduction="mean",
        **kwargs,
    ):
        operator = operator.lower().strip()
        if operator == "poisson":
            residual, u = self.Poisson_Residual(
                model_fn=model_fn,
                coords=coords,
                source_term=source_term,
                output_component=output_component,
                space_dims=kwargs.get("space_dims", None),
            )
        elif operator == "heat":
            residual, u = self.Heat_Residual(
                model_fn=model_fn,
                coords=coords,
                alpha=kwargs.get("alpha", 1.0),
                source_term=source_term,
                output_component=output_component,
                time_dim=kwargs.get("time_dim", -1),
                space_dims=kwargs.get("space_dims", None),
            )
        elif operator == "wave":
            residual, u = self.Wave_Residual(
                model_fn=model_fn,
                coords=coords,
                c=kwargs.get("c", 1.0),
                source_term=source_term,
                output_component=output_component,
                time_dim=kwargs.get("time_dim", -1),
                space_dims=kwargs.get("space_dims", None),
            )
        elif operator == "burgers":
            residual, u = self.Burgers_Residual(
                model_fn=model_fn,
                coords=coords,
                nu=kwargs.get("nu", 0.01),
                source_term=source_term,
                x_dim=kwargs.get("x_dim", 0),
                time_dim=kwargs.get("time_dim", -1),
                output_component=output_component,
            )
        else:
            residual, u = self.Residual_From_Operator(
                model_fn=model_fn,
                coords=coords,
                operator_fn=kwargs["operator_fn"],
                output_component=output_component,
            )
        loss = self.Physics_Loss(residual, reduction=reduction)
        return {
            "residual": residual.astype(self.dtype, copy=False),
            "loss": loss,
            "prediction": u.astype(self.dtype, copy=False),
        }



class PINNs:
    def _init_parameters(self):
        weights = []
        biases = []
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = cp.sqrt(cp.asarray(6.0 / (in_dim + out_dim), dtype=self.compute_dtype))
            w = self.rng.uniform(
                low=float(limit.item()),
                high=float(limit.item()),
                size=(out_dim, in_dim),
            ).astype(self.dtype)
            b = cp.zeros((out_dim,), dtype=self.dtype)
            weights.append(w)
            biases.append(b)
        return weights, biases

    def __init__(
        self,
        layer_sizes,
        hidden_activation="tanh",
        output_activation="linear",
        dtype=cp.float16,
        compute_dtype=cp.float32,
        alpha=ALPHA,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        physics_weight=PHYSICS_WEIGHT,
        data_weight=DATA_WEIGHT,
        boundary_weight=BOUNDARY_WEIGHT,
        seed: int | None = None,
        normalize_inputs: bool = False,
        normalize_outputs: bool = False,
    ):
        self.layer_sizes = list(layer_sizes)
        if len(self.layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output layers!")
        self.hidden_activation = hidden_activation.lower().strip()
        self.output_activation = output_activation.lower().strip()
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.alpha = float(alpha)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.convergence_threshold = float(convergence_threshold)
        self.physics_weight = float(physics_weight)
        self.data_weight = float(data_weight)
        self.boundary_weight = float(boundary_weight)
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_outputs = bool(normalize_outputs)
        self.rng = cp.random.default_rng(seed)
        self.forward_engine = Forward_Propagation(dtype=self.dtype, compute_dtype=self.compute_dtype)
        self.pde_engine = PDE_Residual_Loss(dtype=self.dtype, compute_dtype=self.compute_dtype, eps=self.epsilon)
        self.legacy_backprop = Backpropagation(
            alpha=self.alpha,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            activation=self.hidden_activation,
            seed=seed,
        )
        self.convergence_engine = Global_Convergence_Update(
            beta1=self.beta1,
            epsilon=self.epsilon,
            convergence_threshold=self.convergence_threshold,
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
        )
        self.weights, self.biases = self._init_parameters()
        self.Reset_Optimizer_State()
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.step_count = 0
        self.last_global_grad_norm = None
        self.convergence_flag = cp.array(0, dtype=cp.int32)

    def Reset_Optimizer_State(self):
        self.m_w = [cp.zeros_like(w, dtype=self.compute_dtype) for w in self.weights]
        self.v_m = [cp.zeros_like(w, dtype=self.compute_dtype) for w in self.weights]
        self.m_b = [cp.zeros_like(b, dtype=self.compute_dtype) for b in self.biases]
        self.v_b = [cp.zeros_like(b, dtype=self.compute_dtype) for b in self.biases]
        self.prev_updates_w = [cp.zeros_like(w, dtype=self.compute_dtype) for w in self.biases]
        self.prev_updates_b = [cp.zeros_like(b, dtype=self.compute_dtype) for b in self.biases]
        self.step_count = 0

    def Fit_Normalizer(
        self,
        x,
        y=None
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        self.x_mean = cp.mean(x, axis=0, keepdims=True)
        self.x_std = cp.std(x, axis=0, keepdims=True) + self.epsilon
        if y is not None:
            y = cp.asarray(y, dtype=self.compute_dtype)
            self.y_mean = cp.mean(y, axis=0, keepdims=True)
            self.y_std = cp.std(y, axis=0, keepdims=True) + self.epsilon
        return self

    def Normalize_X(
        self,
        x
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        if self.normalize_inputs and self.x_mean is not None and self.x_std is not None:
            return (x - self.x_mean) / self.x_std
        return x

    def Denormalize_Y(
        self,
        y
    ):
        y = cp.asarray(y, dtype=self.compute_dtype)
        if self.normalize_outputs and self.y_mean is not None and self.y_std is not None:
            return y * self.y_std + self.y_mean
        return y

    def Normalize_Y(
        self,
        y
    ):
        y = cp.asarray(y, dtype=self.compute_dtype)
        if self.normalize_outputs and self.y_mean is not None and self.y_std is not None:
            return (y - self.y_mean) / self.y_std
        return y

    def Ensure_2D(
        self,
        x
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        if x.ndim == 1:
            return x[None, :]
        if x.ndim != 2:
            raise ValueError("Input must be 1D or 2D!")
        return x

    def As_Float(
        self,
        x
    ):
        return cp.asarray(x, dtype=self.compute_dtype)

    def Activation_fn(
        self,
        x,
        name
    ):
        name = name.lower().strip()
        if name == "tanh":
            return cp.tanh(x)
        if name == "relu":
            return cp.maximum(x, 0.0)
        if name == "sigmoid":
            return 1.0 / (1.0 + cp.exp(-x))
        if name == "linear":
            return x
        raise ValueError("Activation must be one of the: tanh, relu, sigmoid, linear!")

    def Activation_Derivative(
        self,
        a,
        name
    ):
        name = name.lower().strip()
        if name == "tanh":
            return 1.0 - a * a
        if name == "relu":
            return (a > 0).astype(self.compute_dtype)
        if name == "sigmoid":
            return a * (1.0 - a)
        if name == "linear":
            return cp.ones_like(a, dtype=self.compute_dtype)
        raise ValueError("activation must be one of the: tanh, relu, sigmoid, linear!")

    def Layer_Activation_Name(
        self,
        layer_idx
    ):
        if layer_idx < len(self.weights) - 1:
            return self.hidden_activation
        return self.output_activation

    def Stack_Parameters(self):
        return [self.weights, self.biases]

    def Flatten_Grads(
        self,
        grads_w,
        grads_b=None
    ):
        flat = []
        for g in grads_w:
            flat.append(cp.ravel(cp.asarray(g, dtype=self.compute_dtype)))
        if grads_b is not None:
            for g in grads_b:
                flat.append(cp.ravel(cp.asarray(g, dtype=self.compute_dtype)))
        if not flat:
            return cp.asarray([], dtype=self.compute_dtype)
        return cp.concatenate(flat)

    def Zeros_Like_Grads(self):
        grads_w = [cp.zeros_like(w, dtype=self.compute_dtype) for w in self.weights]
        grads_b = [cp.zeros_like(b, dtype=self.compute_dtype) for b in self.biases]
        return grads_w, grads_b

    def Add_Grads(
        self,
        g1_w,
        g1_b,
        g2_w,
        g2_b
    ):
        if g1_w is None:
            return g2_w, g2_b
        out_w = [a + b for a, b in zip(g1_w, g2_w)]
        out_b = [a + b for a, b in zip(g1_b, g2_b)]
        return out_w, out_b

    def Forward_With_Cache(
        self,
        x
    ):
        x = self.Ensure_2D(x)
        x = self.Normalize_X(x)
        a = x.astype(self.compute_dtype, copy=False)
        activations = [a]
        pre_activations = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            act_name = self.Layer_Activation_Name(i)
            z = self.forward_engine.Forward(
                weight_matrix=w,
                input_vector=a,
                bias=b,
                activation=act_name,
            )
            z = self.Ensure_2D(z)
            pre_activations.append(z)
            a = z.astype(self.compute_dtype, copy=False)
            activations.append(a)
        return a.astype(self.dtype, copy=False), {
            "activations": activations,
            "pre_activations": pre_activations,
        }

    def Forward(
        self,
        x
    ):
        y, _ = self.Forward_With_Cache(x)
        return self.denormalize_y(y) if self.normalize_outputs else y

    def Predict(
        self,
        x
    ):
        y = self.forward(x)
        if cp.asarray(x).ndim == 1 and y.ndim == 2 and y.shape[0] == 1:
            return y[0]
        return y

    def PDE_Residual_Loss(
        self,
        d_model_output,
        d_initial_model,
        d_source_term,
        d_gradient=None,
        d_loss=None,
        reduction="sum",
        return_computed=False,
    ):
        return self.pde_engine.PDE_Residual_Loss(
            d_model_output=d_model_output,
            d_initial_model=d_initial_model,
            d_source_term=d_source_term,
            d_gradient=d_gradient,
            d_loss=d_loss,
            reduction=reduction,
            return_computed=return_computed,
        )

    def MSE_Loss(
        self,
        pred,
        target,
        reduction="mean"
    ):
        pred = cp.asarray(pred, dtype=self.compute_dtype)
        target = cp.asarray(target, dtype=self.compute_dtype)
        diff = pred - target
        sq = diff * diff
        reduction = reduction.lower().strip()
        if reduction == "mean":
            return cp.mean(sq)
        if reduction == "sum":
            return cp.sum(sq)
        if reduction == "none":
            return sq
        raise ValueError("Reduction must be one of the: 'mean', 'sum', 'none'!")

    def MSE_Output_Grad(
        self,
        pred,
        target,
        reduction="mean"
    ):
        pred = cp.asarray(pred, dtype=self.compute_dtype)
        target = cp.asarray(target, dtype=self.compute_dtype)
        diff = pred - target
        reduction = reduction.lower().strip()
        if reduction == "mean":
            return 2.0 * diff / diff.size
        if reduction == "sum":
            return 2.0 * diff
        if reduction == "none":
            return 2.0 * diff
        raise ValueError("reduction must be one of the: 'mean', 'summ', 'none'!")

    def Physics_Output_Grad(
        self,
        pred,
        initial_model,
        source_term,
        reduction="mean"
    ):
        residual, physics_loss = self.PDE_Residual_Loss(
            d_model_output=pred,
            d_initial_model=initial_model,
            d_source_term=source_term,
            reduction=reduction,
            return_computed=False,
        )
        B = cp.asarray(initial_model, dtype=self.compute_dtype)
        residual = cp.asarray(residual, dtype=self.compute_dtype)
        if reduction == "mean":
            scale = 2.0 / residual.size
        else:
            scale = 2.0
        grad_pred = scale * (residual @ B.T)
        return grad_pred.astype(self.compute_dtype, copy=False), residual, physics_loss

    def Backward_From_Output_Grad(
        self,
        output_grad,
        cache
    ):
        activations = cache["activations"]
        n_layers = len(self.weights)
        output_grad = cp.asarray(output_grad, dtype=self.compute_dtype)
        delta = output_grad * self.Activation_Derivative(
            activations,
            self.output_activation
        )
        grads_w = [None] * n_layers
        grads_b = [None] * n_layers
        batch = max(activations[0].shape[0], 1)
        for layer in range(n_layers - 1, -1, -1):
            a_prev = activations[layer].astype(self.compute_dtype, copy=False)
            grads_w[layer] = (delta.T @ a_prev) / batch
            grads_b[layer] = cp.mean(delta, axis=0)
            if layer > 0:
                delta = (delta @ self.weights[layer].astype(self.compute_dtype, copy=False))
                delta = delta * self.Activation_Derivative(
                    activations[layer],
                    self.hidden_activation
                )
        return grads_w, grads_b

    def Adam_Update(
        self,
        param,
        grad,
        m,
        v
    ):
        m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
        v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
        m_hat = n / (1.0 - self.beta1 ** self.step_count + self.epsilon)
        v_hat = v / (1.0 - self.beta2 ** self.step_count + self.epsilon)
        update = self.alpha * m_hat / (cp.sqrt(v_hat) + self.epsilon)
        param[...] = param - update
        return param

    def Momentum_Update(
        self,
        param,
        grad,
        m
    ):
        m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
        param[...] = param - self.alpha * m
        return param

    def GD_Update(
        self,
        param,
        grad
    ):
        param[...] = param - self.alpha * grad
        return param

    def Update_Parameters(
        self,
        grads_w,
        grads_b,
        optimizer="adam"
    ):
        optimizer = optimizer.lower().strip()
        self.step_count += 1
        for i in range(len(self.weights)):
            gw = cp.asarray(grads_w[i], dtype=self.compute_dtype)
            gb = cp.asarray(grads_b[i], dtype=self.compute_dtype)
            if optimizer == "adam":
                self.Adam_Update(self.weights[i], gw, self.m_w[i], self.v_w[i])
                self.Adam_Update(self.biases[i], gb, self.m_b[i], self.v_b[i])
            elif optimizer in ("momentum", "sgd_momentum"):
                self.Momentum_Update(self.weights[i], gw, self.m_w[i])
                self.Momentum_Update(self.biases[i], gb, self.m_b[i])
            elif optimizer in ("gb", "sgd"):
                self.GD_Update(self.weights[i], gw)
                self.GD_Update(self.biases[i], gb)
            else:
                raise ValueError("Optimizer must be one of the: Adam, momentum, sgd_momentum, GD, SGD")
        return self.weights, self.biases

    def Global_Convergence_Update(
        self,
        grads_w,
        grads_b=None
    ):
        flat = self.Flatten_Grads(grads_w, grads_b)
        global_grad_norm = self.convergence_engine.Compute_Global_Gradient_Norm(flat)
        self.last_global_grad_norm = global_grad_norm
        self.convergence_flag[...] = cp.asarray(
            global_grad_norm < self.convergence_threshold,
            dtype=cp.int32,
        )
        return self.convergence_flag, global_grad_norm

    def Is_Convergence(self):
        return bool(self.convergence_flag.item())

    def PINNs_Training_Step(
        self,
        x_data=None,
        y_data=None,
        x_phys=None,
        initial_model=None,
        source_term=None,
        x_bc=None,
        y_bc=None,
        data_weight=None,
        physics_weight=None,
        boundary_weight=None,
        data_reduction="mean",
        physics_reduction="mean",
        boundary_reduction="mean",
        optimizer="adam",
        use_legacy_backprop: bool = False,
        return_details: bool = True,
    ):
        if use_legacy_backprop:
            raise NotImplementedError("use_legacy_backprop is not recommended for multi-layer PINNs with bias!")
        dw = self.data_weight if data_weight is None else float(data_weight)
        pw = self.physics_weight if physics_weight is None else float(physics_weight)
        bw = self.boundary_weight if boundary_weight is None else float(boundary_weight)
        total_grads_w = None
        total_grads_b = None
        total_data_loss = cp.asarray(0.0, dtype=self.compute_dtype)
        total_physics_loss = cp.asarray(0.0, dtype=self.compute_dtype)
        total_boundary_loss = cp.asarray(0.0, dtype=self.compute_dtype)
        # Data Branch
        if x_data is not None and y_data is not None:
            x_d = self.Normalize_X(self.Ensure_2D(x_data))
            y_d = self.Normalize_Y(self.Ensure_2D(y_data))
            pred_d, cache_d = self.Forward_With_Cache(x_d)
            pred_d = self.Normalize_Y(pred_d) if self.normalize_outputs else pred_d
            data_out_grad = self.MSE_Output_Grad(
                pred_pred_d,
                target=y_d,
                reduction=data_reduction,
            )
            grads_w_d, grads_b_d = self.Backward_From_Output_Grad(data_out_grad, cache_d)
            data_loss = self.MSE_Loss(pred_d, y_d, reduction=data_reduction)
            total_data_loss = cp.asarray(data_loss, dtype=self.compute_dtype)
            total_grads_w, total_grads_b = self.Add_Grads(
                total_grads_w,
                total_grads_b,
                grads_w_d,
                grads_b_d
            )
        # Physics branch
        if x_phys is not None and initial_model is not None and source_term is not None:
            x_p = self.Normalize_X(self.Ensure_2D(x_phys))
            s_term = self.Normalize_Y(self.Ensure_2D(source_term))
            pred_p, cache_p = self.Forward_With_Cache(x_p)
            pred_p = self.Normalize_Y(pred_p) if self.normalize_outputs else pred_p
            phys_out_grad, phys_residual, phys_loss = self.Physics_Output_Grad(
                pred=pred_p,
                initial_model=initial_model,
                source_term=s_term,
                reduction=physics_reduction,
            )
            grads_w_p, grads_b_p = self.Backward_From_Output_Grad(phys_out_grad, cache_p)
            total_physics_loss = cp.asarray(phys_loss, dtype=self.compute_dtype)
            total_grads_w, total_grads_b = self.Add_Grads(
                total_grads_w,
                total_grads_b,
                grads_w_p,
                grads_b_p
            )
        # Boundary branch
        if x_bc is not None and y_bc is not None:
            x_b = self.Normalize_X(self.Ensure_2D(x_bc))
            y_b = self.Normalize_Y(self.Ensure_2D(y_bc))
            pred_b, cache_b = self.Forward_With_Cache(x_b)
            pred_b = self.Normalize_Y(pred_b) if self.normalize_outputs else pred_b
            b_out_grad = self.MSE_Output_Grad(
                pred=pred_b,
                target=y_b,
                reduction=boundary_reduction,
            )
            greads_w_b, grads_b_b = self.Backward_From_Output_Grad(b_out_grad, cache_b)
            boundary_loss = self.MSE_Loss(pred_b, y_b, reduction=boundary_reduction)
            total_boundary_loss = cp.asarray(boundary_loss, dtype=self.compute_dtype)
            total_grads_w, total_grads_b = self.Add_Grads(
                total_grads_w,
                total_grads_b,
                grads_w_b,
                grads_b_b
            )
        if total_grads_w is None or total_grads_b is None:
            raise ValueError("at least one branch of data physics, or boundary must be given!")
        # weight loss
        total_grads_w = [g * (dw + 0.0) for g in total_grads_w]
        total_grads_b = [g * (dw + 0.0) for g in total_grads_b]
        if x_phys is not None and initial_model is not None and source_term is not None:
            total_grads_w = [g + pw * 0.0 for g in total_grads_w]
            total_grads_b = [g + pw * 0.0 for g in total_grads_b]
        if x_bc is not None and y_bc is not None:
            total_grads_w = [g + bw * 0.0 for g in total_grads_w]
            total_grads_b = [g + bw * 0.0 for g in total_grads_b]
        if x_data is not None and y_data is not None:
            total_data_loss = dw * total_data_loss
        if x_phys is not None and initial_model is not None and source_term is not None:
            total_physics_loss = pw * total_physics_loss
        if x_bc is not None and y_bc is not None:
            total_boundary_loss = bw * total_boundary_loss
        # Optimizer update
        self.Update_Parameters(
            total_grads_w,
            total_grads_b,
            optimizer=optimizer
        )
        # Convergence
        flag, grad_norm = self.Global_Convergence_Update(
            total_grads_w,
            total_grads_b
        )
        # Total loss
        total_loss = total_data_loss + total_physics_loss + total_boundary_loss
        if not return_details:
            return total_loss
        return {
            "total_loss": total_loss.astype(self.compute_dtype, copy=False),
            "data_loss": total_data_loss.astype(self.compute_dtype, copy=False),
            "physics_loss": total_physics_loss.astype(self.compute_dtype, copy=False),
            "boundary_loss": total_boundary_loss.astype(self.compute_dtype, copy=False),
            "global_grad_norm": grad_norm.astype(self.compute_dtype, copy=False),
            "converged": bool(flag.item()),
        }

    def Fit_Loop(
        self,
        x_data=None,
        y_data=None,
        x_phys=None,
        initial_model=None,
        source_term=None,
        x_bc=None,
        y_bc=None,
        epochs=EPOCHS,
        optimizer="adam",
        verbose=True,
        patience=None,
        tol=None,
    ):
        history = {
            "loss": [],
            "data_loss": [],
            "physics_loss": [],
            "boundary_loss": [],
            "grad_norm": [],
        }
        best_loss = None
        wait = 0
        for epoch in range(int(epochs)):
            out = self. PINNs_Training_Step(
                x_data=x_data,
                y_data=y_data,
                x_phys=x_phys,
                initial_model=initial_model,
                source_term=source_term,
                x_bc=x_bc,
                y_bc=y_bc,
                optimizer=optimizer,
                return_details=True,
            )
            loss_val = float(out["total_loss"].item())
            data_val = float(out["data_loss"].item())
            phys_val = float(out["physics_loss"].item())
            bc_val = float(out["boundary_loss"].item())
            norm_val = float(out["global_grad_norm"].item())
            history["loss"].append(loss_val)
            history["data_loss"].append(data_val)
            history["physics_loss"].append(phys_val)
            history["boundary_loss"].append(bc_val)
            history["grad_norm"].append(norm_val)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch:05d} | "
                    f"loss={loss_val:.6e} | "
                    f"data={data_val:.6e} | "
                    f"physics={phys_val:.6e} | "
                    f"boundary={bc_val:.6e} | "
                    f"grad_norm={norm_val:.6e} | "
                    f"converged={out['converged']}"
                )
            if tol is not None:
                if best_loss is None or loss_val < best_loss - float(tol):
                    best_loss = loss_val
                    wait = 0
                else:
                    wait += 1
                    if patience is not None and wait >= int(patience):
                        break
            if out["converged"]:
                break
        return history

    def Sample_Collocation_Points(
        self,
        bounds,
        n_points
    ):
        b = cp.asarray(bounds, dtype=self.compute_dtype)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must be shaped [dim, 2]!")
        dim = b.shape[0]
        low = b[:, 0]
        high = b[:, 1]
        u = self.rng.random((int(n_points), dim), dtype=self.compute_dtype)
        return low + (high - low) * u

    def State_Dict(self):
        return {
            "layer_sizes": self.layer_sizes,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "dtype": self.dtype,
            "compute_dtype": self.compute_dtype,
            "alpha": self.alpha,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "convergence_threshold": self.convergence_threshold,
            "physics_weight": self.physics_weight,
            "data_weight": self.data_weight,
            "boundary_weight": self.boundary_weight,
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "m_w": [m.copy() for m in self.m_w],
            "v_w": [v.copy() for v in self.v_w],
            "m_b": [m_copy() for m in self.m_b],
            "v_b": [v.copy() for v in self.v_b],
            "prev_updates_w": [u.copy() for u in self.prev_updates_w],
            "prev_updates_b": [u.copy() for u in self.prev_updates_b],
            "step_count": self.step_count,
            "x_mean": None if self.x_mean is None else self.x_mean.copy(),
            "x_std": None if self.x_std is None else self.x_std.copy(),
            "y_mean": None if self.y_mean is None else self.y_mean.copy(),
            "y_std": None if self.y_std is None else self.y_std.copy(),
            "normalize_inputs": self.normalize_inputs,
            "normalize_outputs": self.normalize_outputs,
        }

    def Load_State_Dict(
        self,
        state
    ):
        self.layer_sizes = list(state["layer_sizes"])
        self.hidden_activation = state["hidden_activation"]
        self.output_activation = state["output_activation"]
        self.alpha = float(state["alpha"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.epsilon = float(state["epsilon"])
        self.convergence_threshold = float(state["convergence_threshold"])
        self.physics_weight = float(state["physics_weight"])
        self.data_weight = float(state["data_weight"])
        self.boundary_weight = float(state["boundary_weight"])
        self.weights = [cp.asarray(w, dtype=self.dtype) for w in state["weights"]]
        self.biases = [cp.asarray(b, dtype=self.dtype) for b in state["biases"]]
        self.m_w = [cp.asarray(m, dtype=self.compute_dtype) for m in state["m_w"]]
        self.v_w = [cp.asarray(v, dtype=self.compute_dtype) for v in state["v_w"]]
        self.m_b = [cp.asarray(m, dtype=self.compute_dtype) for m in state["m_b"]]
        self.v_b = [cp.asarray(v, dtype=self.compute_dtype) for v in state["v_b"]]
        self.prev_updates_w = [cp.asarray(u, dtype=self.compute_dtype) for u in state["prev_updates_w"]]
        self.prev_updates_b = [cp.asarray(u, dtype=self.compute_dtype) for u in state["prev_updates_b"]]
        self.step_count = int(state["step_count"])
        self.x_mean = None if state["x_mean"] is None else cp.asarray(state["x_mean"], dtype=self.compute_dtype)
        self.x_std = None if state["x_std"] is None else cp.asarray(state["x_std"], dtype=self.compute_dtype)
        self.y_mean = None if state["y_mean"] is None else cp.asarray(state["y_mean"], dtype=self.compute_dtype)
        self.y_std = None if state["y_std"] is None else cp.asarray(state["y_std"], dtype=self.compute_dtype)
        self.normalize_inputs = bool(state.get("normalize_inputs", False))
        self.normalize_outputs = bool(state.get("normalize_outputs", False))
        return self

    def Get_Weights(self):
        return self.weights

    def Get_Biases(self):
        return self.biases

    def Set_Weights(
        self,
        weights,
        biases=None
    ):
        if len(weights) != len(self.weights):
            raise ValueError("Number of weights does not match!")
        for i, w in enumerate(weights):
            self.weights[i][...] = cp.asarray(w, dtype=self.dtype)
        if biases is not None:
            if len(biases) != len(self.biases):
                raise ValueError("Number of biases does not match!")
            for i, b in enumerate(biases):
                self.biases[i][...] = cp.asarray(b, dtype=self.dtype)
        return self.weights, self.biases

    def Reset(self):
        self.weights, self.biases = self._init_parameters()
        self.Reset_Optimizer_State()
        self.convergence_flag[...] = 0
        self.last_global_grad_norm = None
        self.x_mean = self.x_std = self.y_mean = self.y_std = None
        return self
        

            
class Reinforcement_Learning:
    def __init__(
        self,
        num_actions: int,
        num_states: int | None = None,
        state_dim: int | None = None,
        buffer_capacity: int = BUFFER_CAPACITY,
        num_ensemble: int = NUM_ENSEMBLE,
        num_options: int = NUM_OPTIONS,
        seed: int | None = None,
        dtype=cp.float16,
        compute_dtype=cp.float32,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        lambda_: float = LAMBDA,
        tau: float = TAU,
        epsilon: float = EPSILON,
        temperature: float = TEMPERATURE,
        policy_lr: float = POLICY_LR,
        value_lr: float = VALUE_LR,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
    ):
        self.num_actions = int(num_actions)
        self.num_states = None if num_states is None else int(num_states)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.to_k = int(K_VALUE)
        self.greedy = False
        self.pinn_input_dim = None
        self.pinn_output_dim = None
        self.policy_lr = float(policy_lr)
        self.value_lr = float(value_lr)
        self.convergence_threshold = float(convergence_threshold)
        self.num_options = int(num_options)
        self.num_ensemble = int(num_ensemble)
        self.rng = cp.random.default_rng(seed)
        self.action_engine = Action_MultiStrategy(
            self.num_actions,
            seed
        )
        self.meta_engine = Adaptive_MetaLearning(
            num_agents=NUMBER_OF_AGENTS,
            temperature_init=self.temperature,
            global_temperature_init=self.temperature,
            epsilon_init=self.epsilon,
            dtype=cp.float32,
        )
        self.credit_engine = Credit_Assignment(
            num_actions=self.num_actions,
            alpha=self.alpha,
            gamma=self.gamma,
            lambda_=self.lambda_,
            learning_rate=self.policy_lr,
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
        )
        self.curiosity_engine = Curiosity_And_Regulation(
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            entropy_eps=ENTROPY_EPS,
            sparse_eps=SPARSE_EPS,
        )
        self.hierarchy_engine = Hierarchical_Temporal_Abstraction(
            num_actions=self.num_actions,
            num_options=self.num_options,
            state_embed_dim=self.state_dim,
            goal_embed_dim=None,
            max_steps_per_option=MAX_STEPS_PER_OPTION,
            option_exploration=OPTION_EXPLORATION,
            seed=seed,
            dtype=self.compute_dtype,
            eps=EPS,
        )
        self.noise_engine = Network_And_Noise(
            seed=seed,
            dtype=self.compute_dtype
        )
        self.normalization_engine = Normalization(
            eps=EPS,
            dtype=self.dtype,
            compute_dtype=self.compute_dtype
        )
        self.policy_loss_engine = Policy_Loss_Option(
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            eps=EPS
        )
        self.reward_engine = Reward_Aggregation(
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            eps=EPS
        )
        self.sarsa_engine = None
        self.td_engine = None
        self.q = None
        self.q1 = None
        self.q2 = None
        self.q_ensemble = None
        if self.num_states is not None:
            self.configure_tabular(self.num_states)
        self.policy_W = None
        self.value_W = None
        self.target_policy_W = None
        self.target_value_W = None
        self.linear_bp = Backpropagation(
            alpha-self.policy_lr,
            beta1=0.0,
            beta2=0.0,
            epsilon=EPSILON,
            dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            activation="linear",
            seed=seed,
        )
        try:
            self.replay = Reply_Buffer(
                capacity=buffer_capacity,
                state_shape=None,
                action_dtype=cp.int32,
                reward_dtype=cp.float16,
                seed=seed,
            )
        except Exception:
            self.replay = Safe_Reply_Buffer(buffer_capacity, seed=seed)

        self.q_log = None
        self.last_global_grad_norm = None
        self.convergence_flag = cp.array(0, dtype=cp.int32)
        self.last_metrics = {}

    # Configuration
    def Configure_Tabular(
        self,
        num_states: int
    ):
        self.num_states = int(num_states)
        try:
            self.sarsa_engine = State_Action_Reward_State_Action(
                self.num_states,
                self.num_actions,
                alpha=self.alpha,
                gamma = self.gamma,
                epsilon=self.epsilon,
                dtype=self.dtype,
                compute_dtype=self.compute_dtype,
                seed=None,
            )
            self.q = self.sarsa_engine.Q
        except Exception:
            self.sarsa_engine = None
            self.q = cp.zeros((self.num_states, self.num_actions), dtype=self.dtype)

        try:
            self.td_engine = Temporal_Difference(
                self.num_states,
                self.num_actions,
                num_ensemble=self.num_ensemble,
                dtype=self.dtype,
                compute_dtype=self.compute_dtype,
                seed=None,
            )
            self.q1 = self.td_engine.Q1
            self.q2 = self.td_engine.Q2
            self.q_ensemble = self.td_engine.Q_ensemble
        except Exception:
            self.td_engine = None
            self.q1 = cp.zeros((self.num_states, self.num_actions), dtype=self.dtype)
            self.q2 = cp.zeros((self.num_states, self.num_actions), dtype=self.dtype)
            self.q_ensemble = cp.zeros((self.num_ensemble, self.num_states, self.num_actions), dtype=self.dtype)
        self.hierarchy_engine.state_embed_dim = self.num_states
        return self

    def As_1D_Int(
        self,
        x
    ):
        return cp.asarray(x, dtype=cp.int32).ravel()

    def As_1D_Float(
        self,
        x
    ):
        return cp.asarray(x, dtype=self.compute_dtype)

    def Ensure_2D(
        self,
        x
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        if x.ndim == 0:
            return x[None, None]
        if x.ndim == 1:
            return x[None, :]
        return x.reshape(x.shape[0], -1)

    def State_Features(
        self,
        states
    ):
        x = cp.asarray(states)
        if x.ndim == 0:
            x = x[None]
        if x.dtype.kind in "iu" and self.num_states is not None and x.ndim == 1:
            idx = x.astype(cp.int32).ravel()
            feats = cp.eye(self.num_states, dtype=self.compute_dtype)[idx]
        else:
            feats = cp.asarray(x, dtype=self.compute_dtype)
            if feats.ndim == 1:
                feats = feats[None, :]
            feats = feats.reshape(feats.shape[0], -1)
        ones = cp.ones((feats.shape[0], 1), dtype=self.compute_dtype)
        return cp.concatenate([feats, ones], axis=1)

    def Ensure_Linear_Head(
        self,
        head_name: str, feature_dim: int,
        out_dim: int
    ):
        attr = f"{head_name}_W"
        W = getattr(self, attr)
        if W is None or W.shape != (out_dim, feature_dim):
            limit = cp.sqrt(cp.asarray(6.0 / (feature_dim + out_dim), dtype=self.compute_dtype))
            W = self.rng.uniform(
                low = -float(limit.item()),
                high=float(limit.item()),
                size=(out_dim, feature_dim),
            ).astype(self.dtype)
            setattr(self, attr, W)
            if head_name == "policy":
                self.target_policy_W = W.copy()
            elif head_name == "value":
                self.target_value_W = W.copy()
        return getattr(self, attr)

    def Linear_Forward_Direct(
        self,
        W,
        X
    ):
        X = cp.asarray(X, dtype=self.compute_dtype)
        return self.forward_engine.forward_single_layer(W, X, activation="linear")

    def Linear_Forward(
        self,
        W,
        X
    ):
        return self.action_engine.Prepare_Q_Rows(cp.asarray(X, dtype=self.compute_dtype) @ W.T, cp.arange(X.shape[0], dtype=cp.int32))[0] \
            if False else self.Linear_Forward_Direct(W, X)

    def Softmax(
        self,
        x,
        temperature=TEMPERATURE
    ):
        x = cp.asarray(x, dtype=self.compute_dtype)
        temperature = max(float(temperature), EPS2)
        x = x - cp.max(x, axis=1, keepdims=True)
        z = cp.exp(x/temperature)
        return z / (cp.sum(z, axis=1, keepdims=True) + self.normalization_engine.eps)

    def Backpropagation_Linear_Head(
        self,
        W,
        X,
        target,
        lr
    ):
        try:
            updated = self.linear_bp.Backpropagation_With_Gradient_Descent(
                weights=[W],
                activations=[X, pred],
                observed_data=cp.asarray(target, dtype=self.compute_dtype),
                learning_rate=float(lr),
                use_adam=False,
                use_momentum=False,
                residual_weight=0.0,
                return_gradients=False,
            )
            return updated[0]
        except Exception:
            X = cp.asarray(X, dtype=self.compute_dtype)
            target = cp.asarray(target, dtype=self.compute_dtype)
            grad = (2.0 / max(X.shape[0], 1)) * ((pred.astype(self.compute_dtype) - target).T @ X)
            return (W.astype(self.compute_dtype) - float(lr) * grad).astype(self.dtype)

    def Sample_From_Probs(
        self,
        probs
    ):
        probs = cp.asarray(probs, dtype=self.compute_dtype)
        cdf = cp.cumsum(probs, axis=1)
        u = self.rng.random((probs.shape[0], 1), dtype=self.compute_dtype)
        return cp.argmax(cdf >= u, axis=1).astype(cp.int32)

    def Fallback_Select_Actions(
        self,
        scores,
        strategy="auto",
        epsilon=None,
        temperature=None,
        K=None,
        reward_var=REWARD_VAR,
        entropy=ENTROPY,
        entropy_threshold=ENTROPY_THRESHOLD,
        var_threshold=VAR_THRESHOLD
    ):
        scores = cp.asarray(scores, dtype=self.compute_dtype)
        batch = scores.shape[0]
        idx = cp.arange(batch, dtype=cp.int32)
        epsilon = self.epsilon if epsilon is None else float(epsilon)
        temperature = self.temperature if temperature is None else float(temperature)
        K = 5 if K is None else int(K)
        if strategy == "epsilon_greedy":
            greedy = cp.argmax(scores, axis=1).astype(cp.int32)
            explore = self.rng.random(batch) < epsilon
            rnd = self.rng.integers(0, self.num_actions, size=batch, dtype=cp.int32)
            return cp.where(explore, rnd, greedy).astype(cp.int32)
        if strategy == "softmax":
            return self.Sample_From_Probs(self.Softmax(scores, temperature=temperature))
        if strategy == "topk":
            K = max(1, min(K, self.num_actions))
            topk_idx = cp.argpartition(scores, -K, axis=1)[:, -K:]
            topk_vals = cp.take_along_axis(scores, topk_idx, axis=1)
            weights = topk_vals - cp.min(topk_vals, axis=1, keepdims=True) + 1e-6
            probs = weights / cp.sum(weights, axis=1, keepdims=True)
            pos = self.Sample_From_Probs(probs)
            return topk_idx[idx, pos].astype(cp.int32)
        if reward_var > var_threshold:
            return self.Fallback_Select_Actions(scores, "epsilon_greedy", epsilon=epsilon)
        if entropy > entropy_threshold:
            return self.Fallback_Select_Actions(scores, "topk", K=K)
        return self.Fallback_Select_Actions(scores, "softmax", temperature=temperature)

    def Hierarchical_Action(
        self,
        states,
        option_scores=None,
        epsilon=None
    ):
        states = cp.asarray(states)
        if states.ndim == 0:
            states = states[None]
        batch = states.shape[0]
        epsilon = self.epsilon if epsilon is None else float(epsilon)
        state_ids = states.astype(cp.int32).ravel()
        if option_scores is not None:
            option_ids = cp.argmax(cp.asarray(option_scores), axis=1).astype(cp.int32)
        else:
            option_ids = (state_ids % max(self.num_options, 1)).astype(cp.int32)
        base_action = (state_ids + option_ids) % self.num_actions
        explore = self.rng.random(batch) < epsilon
        rnd = self.rng.integers(0, self.num_actions, size=batch, dtype=cp.int32)
        return cp.where(explore, rnd, base_action).astype(cp.int32), option_ids

    def Policy_Scores(
        self,
        states
    ):
        X = self.State_Features(states)
        W = self.Ensure_Linear_Head("policy", X.shape[1], self.num_actions)
        try:
            return self.forward_engine.forward_single_layer(W, X, activation="linear")
        except Exception:
            return X @ W.T

    def Value_Scores(
        self,
        states
    ):
        X = self.State_Features(states)
        W = self.Ensure_Linear_Head("value", X.shape[1], 1)
        try:
            return self.forward_engine.forward_single_layer(W, X, activation="linear")
        except Exception:
            return X @ W.T

    def Train_Policy_From_Targets(
        self,
        states,
        target_scores,
        lr=None
    ):
        X = self.State_Features(states)
        W = self.Ensure_Linear_Head("policy", X.shape[1], self.num_actions)
        target_scores = cp.asarray(target_scores, dtype=self.compute_dtype)
        lr = self.policy_lr if lr is None else float(lr)
        self.policy_W = self.Backpropagation_Linear_Head(W, X, target_scores, lr=lr)
        return self.policy_W

    def Train_Value_From_Targets(
        self,
        states,
        target_values,
        lr=None
    ):
        X = self.State_Features(states)
        W = self.Ensure_Linear_Head("value", X.shape[1], 1)
        target_values = cp.asarray(target_values, dtype=self.compute_dtype).reshape(-1,1)
        lr = self.value_lr if lr is None else float(lr)
        self.value_W = self.Backpropagation_Linear_Head(W, X, target_values, lr=lr)
        return self.value_W

    def Select_Action(
        self,
        states,
        strategy="auto",
        use_policy_head: bool = False,
        hierarchical: bool = False,
        epsilon = None,
        temperature = None,
        K = None,
        reward_var: float = REWARD_VAR,
        entropy: float = ENTROPY,
        entropy_threshold: float = ENTROPY_THRESHOLD,
        var_threshold: float = VAR_THRESHOLD,
        option_scores=None,
    ):
        if hierarchical:
            actions, option_ids = self.Hierarchical_Action(states, option_scores=option_scores, epsilon=epsilon)
            return actions, option_ids
        if use_policy_head or self.policy_W is not None:
            scores = self.policy_scores(states)
        else:
            if self.num_states is None:
                raise ValueError("number of states is not configurate yet. Call Configure_Tabular(number_of_states) or use policy head.")
            s = cp.asarray(states, dtype=cp.int32).ravel()
            scores = self.q[s]

        try:
            batch_idx = cp.arange(scores.shape[0], dtype=cp.int32)
            if strategy == "epsilon_greedy":
                return self.action_engine.Select_Action_Epsilon_Greedy(
                    scores,
                    batch_idx,
                    epsilon if epsilon is not None else self.epsilon
                )
            if strategy == "softmax":
                return self.action_engine.Softmax_Policy_Action_Select(
                    scores,
                    batch_idx,
                    temperature if temperature is not None else self.temperature
                )
            if strategy == "topk":
                return self.action_engine.Select_Action_TopK_Sampling(
                    scores,
                    batch_idx,
                    K if K is not None else 5
                )
            return self.action_engine.Auto_Select_Action_MultiStrategy(
                scores,
                batch_idx,
                epsilon if epsilon is not None else self.epsilon,
                temperature if temperature is not None else self.temperature,
                K if K is not None else 5,
                reward_var,
                entropy,
                entropy_threshold,
                var_threshold,
            )
        except Exception:
            return self.Fallback_Select_Actions(
                scores,
                strategy=strategy,
                epsilon=epsilon,
                temperature=temperature,
                K=K,
                reward_var=reward_var,
                entropy=entropy,
                entropy_threshold=entropy_threshold,
                var_threshold=var_threshold,
            )

    def Store_Transition(
        self,
        state,
        action,
        reward,
        next_state,
        next_action=0,
        done=False,
        priority=None
    ):
        try:
            self.replay.Add(state, action, reward, next_state, next_action, priority=priority)
        except Exception:
            self.replay.add(state, action, reward, next_state, next_action=next_action, done=done, priority=priority)

    def Sample_Batch(
        self,
        batch_size: int,
        prioritized: bool = False
    ):
        try:
            if prioritized:
                return self.replay.Sample_Prioritized(batch_size)
            return self.replay.Sample_Uniform(batch_size, replace=True)
        except Exception:
            if prioritized:
                return self.replay.Sample_Prioritized(batch_size)
            return self.replay.Sample_Uniform(batch_size, replace=True)

    def Normalize_Q(
        self,
        Q=None,
        inplace: bool = True
    ):
        Q = self.q if Q is None else Q
        try:
            return self.normalization_engine.Normalize_Q(Q, inplace=inplace)
        except Exception:
            Q = cp.asarray(Q)
            if Q.ndim != 2:
                raise ValueError("Q harus 2D.")
            work = Q if inplace else Q.copy()
            qf = work.astype(self.compute_dtype, copy=False)
            q_min = cp.min(qf, axis=1, keepdims=True)
            q_max = cp.max(qf, axis=1, keepdims=True)
            denom = cp.where((q_max - q_min) > 0, q_max - q_min, 1.0)
            work[...] = ((qf - q_min) / denom).astype(work.dtype, copy=False)
            return work

    def Normalize_Rewards(
        self,
        rewards,
        running: bool = False,
        momentum: float = MOMENTUM,
        return_stats: bool = True
    ):
        try:
            if running:
                return self.normalization_engine.Normalize_Rewards_Block(
                    rewards,
                    running=True,
                    momentum=momentum,
                    return_stats=return_stats
                )
            return self.normalization_engine.Normalize_Rewards(
                rewards,
                return_stats=return_stats
            )
        except Exception:
            r = cp.asarray(rewards, dtype=self.compute_dtype).ravel()
            mean = cp.mean(r)
            std = cp.sqrt(cp.var(r) + self.normalization_engine.eps)
            norm = (r - mean) / std
            if return_stats:
                return norm.astype(self.dtype), mean, std
            return norm.astype(self.dtype)

    def Aggregate_Rewards(
        self,
        rewards,
        tile_size=TILE_SIZE,
        pad_value=PAD_VALUE
    ):
        try:
            return self.reward_engine.Aggregate(rewards, tile_size=tile_size, pad_value=pad_value)
        except Exception:
            r = cp.asarray(rewards, dtype=self.compute_dtype)
            if r.ndim == 1:
                r = r[None, :]
            steps = r.shape[1]
            if steps % tile_size != 0:
                pad = tile_size - (steps % tile_size)
                r = cp.pad(r, ((0, 0), (0, pad)), mode="constant", constant_values=pad_value)
            tile_sums = cp.sum(r.reshape(r.shape[0], -1, tile_size), axis=2)
            ep_sums = cp.sum(tile_sums, axis=1)
            return {
                "tile_sums": tile_sums.astype(self.dtype),
                "episode_sums": ep_sums.astype(self.dtype),
                "stats": {
                    "mean": cp.mean(ep_sums).astype(self.compute_dtype),
                    "var": cp.var(ep_sums).astype(self.compute_dtype),
                    "sum": cp.sum(ep_sums).astype(self.compute_dtype),
                    # "sumsq": cp.sum(ep_sums *MOMENTUM = MOMENTUM ep_sums).astype(self.compute_dtype),
                    "sumsq": cp.sum(cp.square(ep_sums)).astype(self.compute_dtype),
                },
                "episode_rewards": ep_sums.astype(self.dtype),
            }

    def Apply_Curiosity_And_Regulation(
        self,
        state_embeddings,
        predicted_embeddings,
        log_probs=None,
        Q=None,
        sparsity_lambda=SPARSE_EPS,
        etropy_reduce="mean"
    ):
        try:
            if log_probs is None or Q is None:
                return self.curiosity_engine.Curiosity_Reward(
                    state_embeddings,
                    predicted_embeddings
                )
            return self.curiosity_engine.Regulate_Policy_And_Q(
                state_embeddings,
                predicted_embeddings,
                log_probs,
                Q,
                sparsity_lambda,
                entropy_reduce=entropy_reduce,
                in_place_q=True,
            )
        except Exception:
            s = cp.asarray(state_embeddings, dtype=self.compute_dtype)
            p = cp.asarray(predicted_embeddings, dtype=self.compute_dtype)
            intrinsic = cp.sum((s-p) ** 2, axis=-1)
            if log_probs is None or Q is None:
                return intrinsic.astype(self.dtype)
            lp = cp.asarray(log_probs, dtype=self.compute_dtype)
            lp = lp - cp.max(lp, axis=-1, keepdims=True)
            probs = cp.exp(lp)
            entropy = -cp.sum(probs * lp, axis=-1)
            Q = cp.asarray(Q)
            Q_reg = cp.sign(Q) * cp.maximum(cp.abs(Q) - float(sparsity_lambda), 0.0)
            return {
                "Curiosity_Rewards": intrinsic.astype(self.dtype),
                "Policy_Entropy": entropy.astype(self.dtype),
                "Q_regularized": Q_reg.astype(Q.dtype, copy=False),
            }

    def Compute_Advantages(
        self,
        rewards,
        values,
        gamma=None,
        lam=None
    ):
        rewards = cp.asarray(rewards, dtype=self.compute_dtype).ravel()
        values = cp.asarray(values, dtype=self.compute_dtype).ravel()
        gamma = self.gamma if gamma is None else float(gamma)
        lam = self.lambda_ if lam is None else float(lam)
        if values.size != rewards.size + 1:
            raise ValueError("Values must be T+1 for GAE.")
        deltas = rewards + gamma * values[1:] - values[:-1]
        adv = cp.empty_like(rewards)
        gae = cp.asarray(0.0, dtype=self.compute_dtype)
        for t in range(rewards.size-1, -1, -1):
            gae = deltas[t] + gamma * lam * gae
            adv[t] = gae
        return adv.astype(self.dtype)

    def Compute_Policy_Loss_Metrics(
        self,
        old_log_probs,
        new_log_probs,
        advantages,
        epsilon=EPSILON,
        kl_coeff=KULLBACK_LEIBLER_COEFF
    ):
        old_log_probs = cp.asarray(old_log_probs, dtype=self.compute_dtype).ravel()
        new_log_probs = cp.asarray(new_log_probs, dtype=self.compute_dtype).ravel()
        advantages = cp.asarray(advantages, dtype=self.compute_dtype).ravel()
        eps = float(epsilon)
        try:
            return self.policy_loss_engine.Policy_Loss_Option_Step(
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                advantages=advantages,
                epsilon=eps,
                kl_coeff=kl_coeff,
                old_policy=None,
                new_poolicy=None,
                max_kl=None,
                inplace_trust_region=True,
                loss_reduce="mean",
            )
        except Exception:
            ratio = cp.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = cp.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages
            ppo_loss = -cp.mean(cp.minimum(unclipped, clipped))
            kl = cp.mean(old_log_probs - new_log_probs)
            return {"ppo_loss": ppo_loss, "kl_penalty": kl}

    def Update_Meta_Parameters(
        self,
        mean_entropy_local,
        mean_entropy_global,
        target_entropy,
        lr_global,
        lr_local,
        mean_reward,
        var_reward,
        decay_rate,
        min_epsilon,
    ):
        try:
            temp_array, global_temp = self.meta_engine.Adaptive_Temperature_Scheduler(
                mean_entropy_local=mean_entropy_local,
                mean_entropy_global=mean_entropy_global,
                target_entropy=target_entropy,
                lr_global=lr_global,
                lr_local=lr_local,
            )
        except Exception:
            me_local = cp.asarray(mean_entropy_local, dtype=self.compute_dtype).ravel()
            me_global = cp.asarray(mean_entropy_global, dtype=self.compute_dtype).ravel()
            target_entropy = cp.asarray(target_entropy, dtype=self.compute_dtype)
            lr_global = cp.asarray(lr_global, dtype=self.compute_dtype)
            lr_local = cp.asarray(lr_local, dtype=self.compute_dtype)
            self.meta_engine.global_temperature[...] = cp.clip(
                self.meta_engine.global_temperature + lr_global * (me_global - target_entropy),
                GLOBAL_TEMPERATURE_MIN,
                GLOBAL_TEMPERATURE_MAX,
            )
            self.meta_engine.temperature_array = cp.clip(
                self.meta_engine.temperature_array + lr_local * (me_local - target_entropy) * self.meta_engine.global_temperature,
                LOCAL_TEMPERATURE_MIN,
                LOCAL_TEMPERATURE_MAX,
            )
            temp_array = self.meta_engine.temperature_array
            global_temp = self.meta_engine.global_temperature

        try:
            eps_array = self.meta_engine.Epsilon_Decay(
                mean_reward=mean_reward,
                var_reward=var_reward,
                decay_rate=decay_rate,
                min_epsilon=min_epsilon,
            )
        except Exception:
            var_reward = cp.asarray(var_reward, dtype=self.compute_dtype)
            decay_rate = cp.asarray(decay_rate, dtype=self.compute_dtype)
            min_epsilon = cp.asarray(min_epsilon, dtype=self.compute_dtype)
            confidence = cp.clip(1.0 / (1.0 + var_reward), 0.01, 1.0)
            self.meta_engine.epsilon = cp.maximum(self.meta_engine.epsilon * decay_rate * confidence + min_epsilon, min_epsilon)
        self.epsilon = float(cp.asarray(eps_array).mean().item())
        self.temperature = float(cp.asarray(global_temp).item())
        return temp_array, global_temp, eps_array

    def Update_Tabular(
        self,
        batch,
        method="sarsa",
        alpha=None,
        gamma=None,
        lambda_=None,
        tau=None,
        tau_temperature=TAU_TEMPERATURE,
        munchausen_coef=MUNCHAUSEN_COEF,
        log_clip_lower=LOG_CLIP_LOWER
    ):
        if self.num_states is None:
            raise ValueError("Tabular mode needs number of states. Call configure_tabular(num_states) first!")
        method = method.lower().strip()
        alpha = self.alpha if alpha is None else float(alpha)
        gamma = self.gamma if gamma is None else float(gamma)
        lambda_ = self.lambda_ if lambda_ is None else float(lambda_)
        tau = self.tau if tau is None else float(tau)
        states = cp.asarray(batch["states"], dtype=cp.int32).ravel()
        actions = cp.asarray(batch["actions"], dtype=cp.int32).ravel()
        rewards = cp.asarray(batch["rewards"], dtype=self.compute_dtype).ravel()
        next_states = cp.asarray(batch["next_states"], dtype=cp.int32).ravel()
        next_actions = cp.asarray(batch.get("next_actions", actions), dtype=cp.int32).ravel()
        dones = cp.asarray(batch.get("dones", cp.zeros_like(states, dtype=cp.bool_)), dtype=cp.bool_).ravel()
        if method == "sarsa":
            if self.sarsa_engine is not None:
                try:
                    self.sarsa_engine.Update(states, actions, rewards, next_states, next_actions, alpha=alpha, gamma=gamma)
                    return self.q
                except Exception:
                    pass
            q_sa = self.q[states, actions].astype(self.compute_dtype, copy=False)
            q_next = self.q[next_states, next_actions].astyype(self.compute_dtype, copy=False)
            td_target = rewards + gamma * q_next * (~dones).astype(self.compute_dtype)
            td_error = td_target - q_sa
            self.q[states, actions] = (q_sa + alpha * td_error).astype(self.dtype, copy=False)
            return td_error.astype(self.dtype)
        if method == "double_q":
            if self.td_engine is not None:
                try:
                    return self.td_engine.Double_Q_Update(
                        states,
                        actions,
                        rewards,
                        next_states,
                        alpha=alpha,
                        gamma=gamma
                    )
                except Exception:
                    pass
            rng_mask = self.rng.random(states.size) < 0.5
            a_star = cp.argmax(self.q1[next_states].astype(self.compute_dtype, copy=False), axis=1).astype(cp.int32)
            q_eval = cp.where(rng_mask, self.q2[next_states, a_star], self.q1[next_states, a_star]).astype(self.compute_dtype)
            q_cur = cp.where(rng_mask, self.q1[states, actions], self.q2[states, actions]).astype(self.compute_dtype)
            td = rewards + gamma * q_eval * (~dones).astype(self.compute_dtype) - q_cur
            new_q = q_cur + alpha * td
            for i in range(states.size):
                s = int(states[i].item())
                a = int(actions[i].item())
                if bool(rng_mask[i].item()):
                    self.q1[s, a] = cp.asarray(new_q[i], dtype=self.dtype)
                else:
                    self.q2[s, a] = cp.asarray(new_q[i], dtype=self.dtype)
            return td.astype(self.dtype)
        if method == "ensemble_q":
            if self.td_engine is not None:
                try:
                    return self.td_engine.Ensemble_Q_Update(
                        states,
                        actions,
                        rewards,
                        next_states,
                        alpha=alpha,
                        gamma=gamma
                    )
                except Exception:
                    pass
            max_vals = []
            for e in range(self.num_ensemble):
                max_vals.append(cp.max(self.q_ensemble[e, next_states].astype(self.compute_dtype, copy=False), axis=1))
            avg_maxQ = cp.mean(cp.stack(max_vals, axis=0), axis=0)
            target = rewards + gamma * avg_maxQ * (~dones).astype(self.compute_dtype)
            for e in range(self.num_ensemble):
                q_cur = self.q_ensemble[e, states, actions].astype(self.compute_dtype, copy=False)
                self.q_ensemble[e, states, actions] = (q_cur + alpha * (target - q_cur)).astype(self.dtype, copy=False)
            return target.astype(self.dtype)
        if method == "munchausen":
            q0 = self.q1 if self.q1 is not None else self.q
            q_values = q0[states].astype(self.compute_dtype, copy=False)
            max_q = cp.max(q_values, axis=1, keepdims=True)
            log_pi = (q_values - max_q) / max(tau_temperature, EPS2)
            log_pi = log_pi - cp.log(cp.sum(cp.exp(log_pi), axis=1, keepdims=True) + EPS2)
            log_pi_a = log_pi[cp.arange(states.size), actions]
            log_pi_a = cp.maximum(log_pi_a, log_clip_lower)
            r_prime = rewards + munchausen_coef * log_pi_a
            max_next = []
            for e in range(self.num_ensemble):
                max_next.append(cp.max(self.q_ensemble[e, next_states].astype(self.compute_dtype, copy=False), axis=1))
            avg_max_next = cp.mean(cp.stack(max_next, axis=0), axis=0)
            target = r_prime + gamma * avg_max_next * (~dones).astype(self.compute_dtype, copy=False)
            for e in range(self.num_ensemble):
                q_cur = self.q_ensemble[e, states, actions].astype(self.compute_dtype, copy=False)
                self.q_ensemble[e, states, actions] = (q_cur + alpha * (target-q_cur)).astype(self.dtype, copy=False)
            return target.astype(self.dtype)
        if method == "td_lambda":
            transitions = cp.stack([states, actions, rewards.astype(self.dtype), next_states, next_actions], axis=1)
            try:
                return self.credit_engine.TD_Lambda(transitions, self.q, alpha=alpha, gamma=gamma, lambda_=lambda_)
            except Exception:
                e = cp.zeros(states.size, dtype=self.compute_dtype)
                for t in range(states.size):
                    s = int(states[t].item())
                    a = int(actions[t].item())
                    s2 = int(next_states[t].item())
                    a2 = int(next_actions[t].item())
                    delta = rewards[t] + gamma * self.q[s2, a2] * (1.0 - float(dones[t].item())) - self.q[s,a]
                    if t > 0:
                        e[:t] *= gamma * lambda_
                    e[t] += 1.0
                    for i in range(t+1):
                        si = int(states[i].item())
                        ai = int(actions[i].item())
                        self.q[si, ai] = (self.q[si, ai].astype(self.compute_dtype) + alpha * delta * e[i]).astype(self.dtype)
                return self.q
        raise ValueError("Method must be one of the: sarsa, double_q, ensemble_q, munchausen, td_lambda!")

    def Train_Policy_Head_From_Q(
        self,
        states,
        q_values=None,
        temperature=None,
        lr=None
    ):
        if q_values is None:
            if self.num_states is None:
                raise ValueError("Q values must be present if number of states has not been configured!")
            states_idx = cp.asarray(states, dtype=cp.int32).ravel()
            q_value = self.q[states_idx]
        temperature = self.temperature if temperature is None else float(temperature)
        target_probs = self._softmax(q_values, temperature=temperature)
        return self.Train_Policy_From_Targets(states, target_probs, lr=lr)

    def Train_Value_Head_From_Returns(
        self,
        states,
        returns,
        lr=None
    ):
        return self.Train_Value_From_Targets(states, returns, lr=lr)

    def Compute_Physics_Auxiliary_Loss(
        self,
        model_output,
        initial_model,
        source_term,
        reduction="mean"
    ):
        residual, loss = self.pde_engine.PDE_Residual_Loss(
            d_model_output=model_output,
            d_initial_model=initial_model,
            d_source_term=source_term,
            reduction=reduction,
            return_computed=False,
        )
        return residual, loss

    def Update_Convergence(
        self,
        gradients
    ):
        g = cp.asarray(gradient, dtype=self.compute_dtype)
        norm = cp.sqrt(cp.sum(g*g) + EPS)
        self.last_global_grad_norm = norm
        self.convergence_flag[...] = cp.asarray(norm < self.convergence_threshold, dtype=cp.int32)
        return self.convergence_flag, norm

    def Soft_Update_Targets(
        self,
        tau=None
    ):
        tau = self.tau if tau is None else float(tau)
        if self.policy_W is not None and self.target_policy_W is not None:
            self.target_policy_W = self.noise_engine.Update_Target_Network(self.policy_W, self.target_policy_W, tau)
        if self.value_W is not None and self.target_policy_W is not None:
            self.target_value_W = self.noise_engine.Update_Target_Network(self.value_W, self.target_value_W, tau)
        return self.target_policy_W, self.target_value_W

    def Apply_Parameter_Noise(
        self,
        sigma: float = SIGMA
    ):
        if self.policy_W is not None:
            self.policy_W = self.noise_engine.Apply_Parameter_Noise_Single(self.policy_W, sigma=sigma)
        if self.value_W is not None:
            self.value_W = self.noise_engine.Apply_Parameter_Noise_Single(self.value_W, sigma=sigma)
        return self.policy_W, self.value_W

    def Log_Q_Snapshot(
        self,
        snapshot_id: int = 0,
        Q = None,
        max_snapshots: int | None = None
    ):
        Q = self.q if Q is None else Q
        if Q is None:
            raise ValueError("Q is not available!")
        try:
            self.q_log = Logging_Q_Snapshot(
                self,
                Q,
                Q_log=self.q_log,
                snapshot_id=snapshot_id,
                max_snapshots=max_snapshots
            )
            return self.q_log
        except Exception:
            Q = cp.asarray(Q)
            if Q.ndim == 1:
                if self.num_states is None:
                    raise ValueError("number of states is needed for Logging Q 1D!")
                Q = Q.reshape(self.num_states, self.num_actions)
            if self.q_log is None:
                max_snapshots = max(snapshot_id + 1, 1) if max_snapshots is None else int(max_snapshots)
                self.q_log = cp.zeros((max_snapshots, Q.shape[0], Q.shape[1]), dtype=Q.dtype)
            elif snapshot_id >= self.q_log.shape[0]:
                new_size = max(snapshot_id + 1, 2 * self.q_log.shape[0])
                new_buf = cp.zeros((new_size, Q.shape[0], Q.shape[1]), dtype=self.q_log.dtype)
                new_buf[: self.q_log.shape[0]] = self.q_log
                self.q_log = new_buf
            self.q_log[snapshot_id] = Q.astype(self.q_log.dtype, copy=False)
            return self.q_log

    def Training_Step(
        self,
        batch=None,
        tabular_method="sarsa",
        update_policy_head: bool = True,
        update_value_head: bool = True,
        use_curiosity: bool = False,
        state_embeddings=None,
        predicted_embeddings=None,
        log_probs=None,
        sparsity_lambda=SPARSE_EPS,
        use_reward_normalization: bool = True,
        use_replay_prioritized: bool = False,
        policy_temperature=None,
        prioritize_by_td_error: bool = False,
        reward_bonus_scale: float = REWARD_BONUS_SCALE,
    ):
        if batch is None:
            raise ValueError("batch is needed for training step!")
        rewards = cp.asarray(batch["rewards"], dtype=self.compute_dtype).ravel()
        if use_reward_normalization:
            rewards, reward_mean, reward_std = self.Normalize_Rewards(
                rewards,
                running=False,
                return_stats=True
            )
        else:
            reward_mean = cp.mean(rewards)
            reward_std = cp.std(rewards) + EPS
        if use_curiosity and state_embeddings is not None and predicted_embeddings is not None:
            curiosity_out = self.Apply_Curiosity_And_Regulation(
                state_embeddings=state_embeddings,
                predicted_embeddings=predicted_embeddings,
                log_probs=log_probs,
                Q=self.q if self.q is not None else self.q1,
                sparsity_lambda=sparsity_lambda,
            )
            if isinstance(curiosity_out, dict) and "Curiosity_Rewards" in curiosity_out:
                rewards = rewards + reward_bonus_scale * cp.asarray(curiosity_out["Curiosity_Rewards"], dtype=self.compute_dtype).ravel()
        batch = dict(batch)
        batch["rewards"] = rewards.astype(self.dtype)
        tabular_info = None
        if self.num_states is not None and ("states" in batch) and ("actions" in batch) and ("next_states" in batch):
            try:
                tabular_info = self.Update_Tabular(
                    batch,
                    method=tabular_method,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    lambda_=self.lambda_
                )
            except Exception as exc:
                tabular_info = exc
        policy_loss_metrics = None
        if update_policy_head and self.policy_W is not None and "states" in batch:
            if self.num_states is not None and self.q is not None:
                q_values = self.q[cp.asarray(batch["states"], dtype=cp.int32).ravel()]
            elif self.q1 is not None:
                q_values = self.q1[cp.asarray(batch["states"], dtype=cp.int32).ravel()]
            else:
                q_values = None
            if q_values is not None:
                self.Train_Policy_Head_From_Q(
                    batch["states"],
                    q_values=q_values,
                    temperature=policy_temperature
                )
        if update_value_head and self.value_w is not None and "returns" in batch:
            self.Train_Value_Head_From_Returns(batch["states"], batch["returns"])
        grad_proxy = rewards - cp.mean(rewards)
        flag, norm = self.Update_Convergence(grad_proxy)
        self.last_metrics = {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "converged": bool(flag.item()),
            "grad_norm": norm,
            "tabular_info": tabular_info,
            "policy_loss_metrics": policy_loss_metrics,
        }
        return self.last_metrics

    def Fit(
        self,
        env,
        episodes: int = EPISODES,
        max_steps: int = MAX_STEPS,
        tabular_method: str = "sarsa",
        batch_size: int | None = None,
        train_every: int = TRAIN_EVERY,
        prioritized_replay: bool = False,
        online_update: bool = True,
        use_curiosity: bool = False,
        use_reward_normalization: bool = True,
        render: bool = False,
    ):
        history = {
            "episode_reward": [],
            "episode_length": [],
            "loss_like": [],
            "converged": [],
        }
        for ep in range(int(episodes)):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            episode_reward = 0.0
            for t in range(int(max_steps)):
                action = self.select_action(state, strategy="auto", use_policy_head=(self.policy_W is not None))
                if isinstance(action, tuple):
                    action = action[0]
                action_scalar = int(cp.asarray(action).ravel()[0].item())
                step_out = env.step(action_scalar)
                if len(step_out) == 5:
                    next_state, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    next_state, reward, done, info = step_out
                next_action = self.Select_Action(next_state, strategy="auto", use_policy_head=(self.policy_W is not None))
                if isinstance(next_action, tuple):
                    next_action = next_action[0]
                next_action_scalar = int(cp.asarray(next_action).ravel()[0].item())
                self.store_transition(state, action_scalar, reward, next_state, next_action_scalar, done=done)
                episode_reward += float(reward)
                if online_update and (t % int(train_every) == 0) and len(self.replay) > 0:
                    if batch_size is None:
                        batch = self.Sample_Batch(min(len(self.replay), 32), prioritized=prioritized_replay)
                    else:
                        batch = self.Sample_Batch(min(int(batch_size), len(self.replay)), prioritized=prioritized_replay)
                    if self.value_W is not None and "states" in batch:
                        with (cp.errstate(all="ignore")):
                            v_next = self.Value_Scores(batch["next_states"]).astype(self.compute_dtype).ravel()
                            returns = cp.asarray(batch["rewards"], dtype=self.compute_dtype).ravel() + self.gamma * v_next * (~cp.asarray(batch["dones"], dtype=cp.bool_)).astype(self.compute_dtype)
                        batch["returns"] = returns.astype(self.dtype)
                    self.train_step(
                        batch=batch,
                        tabular_method=tabular_method,
                        update_policy_head=True,
                        update_value_head=(self.value_W is not None),
                        use_curiosity=use_curiosity,
                        use_reward_normalization=use_reward_normalization,
                        reward_bonus_scale=REWARD_BONUS_SCALE,
                    )
                state = next_state
                if render and hasattr(env, "render"):
                    env.render()
                if done:
                    break
            agg = self.aggregate_rewards(cp.asarray([episode_reward], dtype=self.compute_dtype))
            history["episode_reward"].append(episode_reward)
            history["episode_length"].append(t+1)
            history["loss_like"].append(float(agg["stats"]["mean"].item()))
            history["converged"].append(self.is_converged())
        return history

    def Is_Converged(self):
        return bool(self.convergence_flag.item())

    def Reset(self):
        if self.sarsa_engine is not None:
            self.sarsa_engine.Reset()
            self.q = self.sarsa_engine.Q
        if self.td_engine is not None:
            try:
                self.td_engine.Q1.fill(0)
                self.td_engine.Q2.fill(0)
                self.td_engine.Q_ensemble.fill(0)
            except Exception:
                pass
        self.q_log = None
        self.last_metrics = {}
        self.last_global_grad_norm = None
        self.convergence_flag[...] = 0
        return self

    def State_Dict(self):
        return {
            "num_actions": self.num_actions,
            "num_states": self.num_states,
            "state_dim": self.state_dim,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda_": self.lambda_,
            "tau": self.tau,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "policy_W": None if self.policy_W is None else self.policy_W.copy(),
            "value_W": None if self.value_W is None else self.value_W.copy(),
            "q": None if self.q is None else self.q.copy(),
            "q1": None if self.q1 is None else self.q1.copy(),
            "q2": None if self.q2 is None else self.q2.copy(),
            "q_ensemble": None if self.q_ensemble is None else self.q_ensemble.copy(),
            "q_log": None if self.q_log is None else self.q_log.copy(),
        }

    def Load_State_Dict(
        self,
        state
    ):
        self.num_actions = int(state["num_actions"])
        self.num_states = None if state["num_states"] is None else int(state["num_states"])
        self.state_dim = None if state["state_dim"] is None else int(state["state_dim"])
        self.alpha = float(state["alpha"])
        self.gamma = float(state["gamma"])
        self.lambda_ = float(state["lambda_"])
        self.tau = float(state["tau"])
        self.epsilon = float(state["epsilon"])
        self.temperature = float(state["temperature"])
        self.policy_W = None if state["policy_W"] is None else cp.asarray(state["policy_W"], dtype=self.dtype)
        self.value_W = None if state["value_W"] is None else cp.asarray(state["value_W"], dtype=self.dtype)
        self.q = None if state["q"] is None else cp.asarray(state["q"], dtype=self.dtype)
        self.q1 = None if state["q1"] is None else cp.asarray(state["q1"], dtype=self.dtype)
        self.q2 = None if state["q2"] is None else cp.asarray(state["q2"], dtype=self.dtype)
        self.q_ensemble = None if state["q_ensemble"] is None else cp.asarray(state["q_ensemble"], dtype=self.dtype)
        self.q_log = None if state["q_log"] is None else cp.asarray(state["q_log"], dtype=self.dtype)
        return self

    def Get_Q(self):
        return self.q

    def Set_Q(
        self,
        Q
    ):
        Q = cp.asarray(Q)
        if self.q is None:
            self.q = Q.astype(self.dtype, copy=False)
        else:
            if self.q.shape != Q.shape:
                raise ValueError("Shape Q does not match!")
            self.q[...] = Q.astype(self.dtype, copy=False)
        return self.q



class Physics_Informed_Reinforcement_Learning_Agent:
    def __init__(
        self,
        pinn=None,
        rl=None,
        env=None,
        vec_env=None,
        backbone=None,
        autodiff=None,
        pinn_kwargs=None,
        rl_kwargs=None,
        num_actions: int | None = None,
        num_states: int | None = None,
        state_dim: int | None = None,
        hidden_dim: int | None = None,
        buffer_capacity: int = BUFFER_CAPACITY,
        seed: int | None = None,
        physics_weight: float = PHYSICS_WEIGHT,
        rl_weight: float = RL_WEIGHT,
        curiosity_weight: float = CURIOSITY_WEIGHT,
        entropy_weight: float = ENTROPY_WEIGHT,
        use_prioritized_replay: bool = True,
        normalize_states: bool = False,
        normalize_rewards: bool = False,
        device_id: int | None = None,
    ):
        self.seed = seed
        self.rng = cp.random.default_rng(seed)
        if device_id is not None:
            cp.cuda.Device(int(device_id)).use()
        self.env = env
        self.vec_env = vec_env
        self.backbone = backbone
        self.autodiff = autodiff
        self._pinn_kwargs = dict(pinn_kwargs or {})
        self._rl_kwargs = dict(rl_kwargs or {})
        self.num_actions = None if num_actions is None else int(num_actions)
        self.num_states = None if num_states is None else int(num_states)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.buffer_capacity = int(buffer_capacity)
        self.physics_weight = float(physics_weight)
        self.rl_weight = float(rl_weight)
        self.curiosity_weight = float(curiosity_weight)
        self.entropy_weight = float(entropy_weight)
        self.use_prioritized_replay = bool(use_prioritized_replay)
        self.normalize_states = bool(normalize_states)
        self.normalize_rewards = bool(normalize_rewards)
        self.last_reward_mean = cp.asarray(0.0, dtype=cp.float32)
        self.last_reward_var = cp.asarray(0.0, dtype=cp.float32)
        self.last_entropy = cp.asarray(0.0, dtype=cp.float32)
        self.pinn = pinn
        self.rl = rl
        self.last_metrics = {}
        self.temperature = float(TEMPERATURE)
        self.epsilon = float(EPSILON)
        self.top_k = int(K_VALUE)
        self.pinn_input_dim = None
        self.pinn_output_dim = None
        self.Configure_From_Environment(self.env or self.vec_env)
        self.Sync_Dimensions()
        self.Build_Missing_Components()
        self.Resolve_Submodules()
        self.Sync_Dimensions()

    def Backbone_Is_Compatible(self):
        if self.backbone is None:
            return False
        return (
            getattr(self.backbone, "input_dim", None) == self.state_dim and
            getattr(self.backbone, "output_dim", None) == self.num_actions and
            getattr(self.backbone, "model_dim", None) == self.hidden_dim and
            getattr(self.backbone, "num_heads", None) == NUMBER_OF_HEADS and
            getattr(self.backbone, "num_layers", None) == NUMBER_OF_LAYERS
        )

    def Build_Missing_Components(self):
        self.Sync_Dimensions()
        # Transformer Backbone
        if self.state_dim is not None and self.num_actions is not None:
            if not self.Backbone_Is_Compatible():
                self.backbone = Transformer_Backbone(
                    input_dim=self.state_dim,
                    model_dim=self.hidden_dim,
                    num_heads=NUMBER_OF_HEADS,
                    num_layers=NUMBER_OF_LAYERS,
                    ff_dim=FF_DIM,
                    output_dim=self.num_actions,
                    max_seq_len=MAXIMUM_SEQUENCE_LEN,
                    dtype=cp.float16,
                    compute_dtype=cp.float32,
                )
        # Automatic Differentiation
        if self.autodiff is None:
            self.autodiff = AutoDiff_PINNs()
        # Vectorized Environment
        if (self.vec_env is None
            and self.env is not None
            and hasattr(self.env, "observation_space")
        ):
            self.vec_env = Vectorized_Environment(
                env_factory=lambda _: self.env,
                num_envs=NUMBER_OF_ENVIRONMENT,
                state_shape=self.env.observation_space.shape,
                action_space_dim=self.num_actions,
                seed=self.seed,
                normalize_states=self.normalize_states,
                normalize_rewards=self.normalize_rewards,
            )
        # PINNs
        if self.state_dim is not None and self.num_actions is not None:
            need_pinn = self.pinn is None
            if self.pinn is not None and hasattr(self.pinn, "layer_sizes"):
                ls = list(self.pinn.layer_sizes)
                if len(ls) >= 2:
                    need_pinn = not(int(ls[0]) == self.state_dim and int(ls[-1]) == self.num_states)
                if need_pinn:
                    cfg = dict(self._pinn_kwargs)
                    cfg.setdefault("layer_sizes", [self.state_dim, self.hidden_dim, self.hidden_dim, self.num_actions])
                    cfg.setdefault("hidden_activation", "tanh")
                    cfg.setdefault("output_activation", "linear")
                    cfg.setdefault("seed", self.seed)
                    cfg.setdefault("normalize_inputs", self.normalize_states)
                    self.pinn = PINNs(**cfg)
        # RL
        if self.rl is None and self.num_actions is not None:
            cfg = dict(self._rl_kwargs)
            cfg.setdefault("num_actions", self.num_actions)
            if self.num_states is not None:
                cfg.setdefault("num_states", self.num_states)
            if self.state_dim is not None:
                cfg.setdefault("state_dim", self.state_dim)
            cfg.setdefault("buffer_capacity", self.buffer_capacity)
            cfg.setdefault("seed", self.seed)
            self.rl = Reinforcement_Learning(**cfg)

    def Sync_Dimensions(self):
        if self.pinn is not None and hasattr(self.pinn, "layer_sizes"):
            if len(self.pinn.layer_sizes) >= 2:
                self.pinn_input_dim = int(self.pinn.layer_sizes[0])
                self.pinn_output_dim = int(self.pinn.layer_sizes[-1])
        """
        if self.backbone is not None:
            if getattr(self.backbone, "input_dim", None) is not None:
                self.state_dim = self.state_dim or int(self.backbone.input_dim)
                self.pinn_input_dim = self.pinn_input_dim or int(self.backbone.input_dim)
            if getattr(self.backbone, "output_dim", None) is not None:
                self.num_actions = self.num_actions or int(self.backbone.output_dim)
                self.pinn_output_dim = self.pinn_output_dim or int(self.backbone.output_dim)
            # self.pinn_input_dim = self.pinn_input_dim or getattr(self.backbone, "input_dim", None)
            # self.pinn_output_dim = self.pinn_output_dim or getattr(self.backbone, "output_dim", None)
        """
        if self.backbone is not None:
            backbone_input = getattr(self.backbone, "input_dim", None)
            backbone_output = getattr(self.backbone, "output_dim", None)
            if backbone_input is not None:
                if getattr(self, "state_dim", None) is None:
                    self.state_dim = int(backbone_input)
                if getattr(self, "pinn_input_dim", None) is None:
                    self.pinn_input_dim = int(backbone_input)
            if backbone_output is not None:
                if getattr(self, "num_actions", None) is None:
                    self.num_actions = int(backbone_output)
                if getattr(self, "pinn_output_dim", None) is None:
                    self.pinn_output_dim = int(backbone_output)
        if self.rl is not None:
            if self.num_actions is None and hasattr(self.rl, "num_actions"):
                self.num_actions = int(self.rl.num_actions)
            if self.num_states is None and getattr(self.rl, "num_states", None) is not None:
                self.num_states = int(self.rl.num_states)
            if self.state_dim is None and getattr(self.rl, "state_dim", None) is not None:
                self.state_dim = int(self.rl.state_dim)
        if self.num_actions is None and self.pinn_output_dim is not None:
            self.num_actions = int(self.pinn_output_dim)
        if self.state_dim is None and self.pinn_input_dim is not None:
            self.state_dim = int(self.pinn_input_dim)

    def Resolve_Submodules(self):
        self.reward_engine = (getattr(self.rl, "reward_engine", None) if self.rl is not None else None) or Reward_Aggregation()
        self.curiosity_engine = (getattr(self.rl, "curiosity_engine", None) if self.rl is not None else None) or Curiosity_And_Regulation()
        self.normalization_engine = (getattr(self.rl, "normalization_engine", None) if self.rl is not None else None) or Normalization()
        self.policy_loss_engine = (getattr(self.rl, "policy_loss_engine", None) if self.rl is not None else None) or Policy_Loss_Option()
        self.meta_engine = (
            getattr(self.rl, "meta_engine", None)
            if self.rl is not None else None
        ) or Adaptive_MetaLearning(
            num_agents=NUMBER_OF_AGENTS,
            temperature_init=TEMPERATURE,
            global_temperature_init=GLOBAL_TEMPERATURE,
            epsilon_init=EPSILON,
        )
        self.network_noise = (
            getattr(self.rl, "noise_engine", None)
            if self.rl is not None else None
        ) or Network_And_Noise(seed=self.seed)
        self.hierarchy_engine = (
            getattr(self.rl, "hierarchy_engine", None)
            if self.rl is not None else None
        ) or Hierarchical_Temporal_Abstraction(
            num_actions=self.num_actions if self.num_actions is not None else NUM_ACTIONS,
            num_options=NUM_OPTIONS,
            state_embed_dim=self.state_dim,
            seed=self.seed,
        )
        self.replay = getattr(self.rl, "replay", None) if self.rl is not None else None
        if self.replay is None:
            self.replay = Reply_Buffer(
                capacity=self.buffer_capacity,
                state_shape=None,
                action_dtype=cp.int32,
                reward_dtype=cp.float16,
                seed=self.seed,
            )
            if self.rl is not None:
                self.rl.replay = self.replay
        # self.pinn_input_dim = None
        # self.pinn_output_dim = None
        self.Sync_Dimensions()
        if self.backbone is not None:
            self.feature_extractor = self.backbone
            if self.pinn is not None:
                self.pinn.backbone = self.backbone
            if self.rl is not None:
                self.rl.backbone = self.backbone
            self.backbone.physics_engine = self.autodiff
            self.backbone.environment = self.vec_env
        if self.autodiff is not None:
            self.physics_engine = self.autodiff
            if self.pinn is not None:
                self.pinn.autodiff = self.autodiff
            if self.rl is not None:
                self.rl.autodiff = self.autodiff
            self.autodiff.backbone = self.backbone
            self.autodiff.environment = self.vec_env
        if self.pinn is not None:
            self.policy_network = self.backbone if self.backbone is not None else self.pinn
            self.pinn.backbone = self.backbone
            self.pinn.autodiff = self.autodiff
            if self.rl is not None:
                self.rl.pinn = self.pinn
        if self.vec_env is not None:
            self.environment = self.vec_env
            # if self.rl is not None:
            #     self.rl.vec_env = self.vec_env
            self.vec_env.rl = self.rl
            self.vec_env.backbone = self.backbone
            self.vec_env.pinn = self.pinn
            if self.rl is not None:
                self.rl.vec_env = self.vec_env
        if self.rl is not None:
            self.rl.backbone = self.backbone
            self.rl.pinn = self.pinn
            self.rl.autodiff = self.autodiff
            self.rl.vec_env = self.vec_env
            # self.action_engine = getattr(self.rl, "action_engine", None)
            self.rl.reward_engine = self.reward_engine
            self.rl.curiosity_engine = self.curiosity_engine
            self.rl.normalization_engine = self.normalization_engine
            self.rl.policy_loss_engine = self.policy_loss_engine
            self.rl.meta_engine = self.meta_engine
            self.rl.noise_engine = self.network_noise
            self.rl.hierarchy_engine = self.hierarchy_engine
            self.rl.replay = self.replay
        self.engine_graph = {
            "feature_extractor": self.backbone,
            "policy_network": self.pinn if self.pinn is not None else self.backbone,
            "physics_engine": self.autodiff,
            "environment": self.vec_env if self.vec_env is not None else self.env,
            "rl_engine": self.rl,
            "replay_buffer": self.replay,
            "reward_engine": self.reward_engine,
            "curiosity_engine": self.curiosity_engine,
            "normalization_engine": self.normalization_engine,
            "policy_loss_engine": self.policy_loss_engine,
            "meta_engine": self.meta_engine,
            "noise_engine": self.network_noise,
            "hierarchy_engine": self.hierarchy_engine,
        }
        self.training_modules = [
            self.backbone,
            self.pinn,
            self.rl,
        ]
        self.physics_modules = [
            self.autodiff,
            self.pinn,
        ]
        self.environment_modules = [
            self.vec_env,
            self.replay,
        ]
        self.Sync_Dimensions()


    def Configure_From_Environment(
        self,
        env=None
    ):
        env = env or self.env or self.vec_env
        if env is None:
            self.Build_Missing_Components()
            self.Sync_Dimensions()
            return self
        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if self.num_actions is None and hasattr(act_space, "n"):
            self.num_actions = int(act_space.n)
        if self.num_states is None and hasattr(obs_space, "n"):
            self.num_states = int(obs_space.n)
        if self.state_dim is None and hasattr(obs_space, "shape") and obs_space.shape is not None:
            self.state_dim = int(np.prod(obs_space.shape))
        self.Build_Missing_Components()
        self.Resolve_Submodules()
        self.Sync_Dimensions()
        return self

    def Call_First(
        self,
        obj,
        names,
        *args,
        default=None,
        **kwargs
    ):
        if obj is None:
            return default
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        return default

    def State_To_Features(
        self,
        state,
        input_dim=None
    ):
        x = cp.asarray(state)
        if x.ndim == 0:
            x = x[None]
        if x.dtype.kind in "iu" and self.num_states is not None and x.ndim == 1:
            idx = x.astype(cp.int32).ravel()
            feats = cp.eye(self.num_states, dtype=cp.float32)[idx]
        else:
            feats = x.astype(cp.float32, copy=False)
            if feats.ndim == 1:
                feats = feats[None, :]
            feats = feats.reshape(feats.shape[0], -1)
        dim = input_dim or self.pinn_input_dim
        if dim is None:
            return feats
        dim = int(dim)
        if feats.shape[1] == dim:
            return feats
        if feats.shape[1] < dim:
            pad = cp.zeros((feats.shape[0], dim - feats.shape[1]), dtype=feats.dtype)
            return cp.concatenate([feats, pad], axis=1)
        return feats[:, :dim]

    def Match_Dim(
        self,
        x,
        dim
    ):
        x = cp.asarray(x, dtype=cp.float32)
        if x.ndim == 1:
            x = x[None, :]
        dim = int(dim)
        if x.shape[1] == dim:
            return x
        if x.shape[1] < dim:
            pad = cp.zeros((x.shape[0], dim - x.shape[1]), dtype=x.dtype)
            return cp.concatenate([x, pad], axis=1)
        return x[:, :dim]

    """
    def Policy_Logits(
        self,
        state
    ):
        # 1) Backbone first
        if self.backbone is not None:
            try:
                logits = self.backbone.policy_logits(state)
            except Exception:
                try:
                    logits = self.backbone.forward(state)
                except Exception:
                    try:
                        logits = self.backbone.forward(state)
                    except Exception:
                        logits = None
                if logits is not None:
                    logits = cp.asarray(logits, dtype=cp.float32)
                    if logits.ndim == 3:
                        logits = logits[:, -1, :]
                    elif logits.ndim > 3:
                        logits = logits.reshape(logits.shape[0], -1)
                    return self.Match_Dim(logits, self.num_actions)
        # 2) PINNs as policy net
        if self.pinn is not None:
            x = self.State_To_Features(state, input_dim=self.pinn_input_dim)
            try:
                logits = self.Call_First(self.pinn, ["Forward", "forward"], x)
            except Exception:
                logits = None
            if logits is not None:
                logits = cp.asarray(logits, dtype=cp.float32)
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                elif logits.ndim > 3:
                    logits = logits.reshape(logits.shape[0], -1)
                return self.Match_Dim(logits, self.num_actions)
        # 3) Tabular Q fallback
        if self.rl is not None and getattr(self.rl, "q", None) is not None and self.num_states is not None:
            s = cp.asarray(state, dtype=cp.int32).ravel()
            if s.ndim == 0:
                s = s[None]
            return self.Match_Dim(self.rl.q[s], self.num_actions)
        raise AttributeError("There is not source of logits/policy available!")
    """

    def Normalize_Logits(
        self,
        logits
    ):
        if logits is None:
            return None
        if isinstance(logits, (tuple, list)):
            if len(logits) == 0:
                return None
            logits = logits[0]
        logits = cp.asarray(logits, dtype=cp.float32)
        if logits.ndim == 0:
            logits = logits.reshape(1, 1)
        elif logits.ndim == 1:
            logits = logits.reshape(1, -1)
        elif logits.ndim == 3:
            logits = logits[:, -1, :]
        elif logits.ndim > 3:
            logits = logits.reshape(logits.shape[0], -1)
        return logits

    """
    def Policy_Logits(
        self,
        state
    ):
        num_actions = getattr(self, "num_actions", None)
        if num_actions is None:
            raise AttributeError("number of actions is not initialized!")
        # 1) Backbone
        backbone = getattr(self, "backbone", None)
        if backbone is not None:
            for method_name in ("policy_logits", "logits", "forward", "__call__"):
                try:
                    if method_name == "__call__":
                        logits = backbone(state)
                    else:
                        fn = getattr(backbone, method_name, None)
                        if fn is None:
                            continue
                        logits = fn(state)
                    logits = self.Normalize_Logits(logits)
                    if logits is not None:
                        return self.Match_Dim(logits, num_actions)
                except Exception:
                    continue
        # 2) PINNs
        pinn = getattr(self, "pinn", None)
        if pinn is not None:
            try:
                pinn_input_dim = getattr(self, "pinn_input_dim", None)
                if pinn_input_dim is not None:
                    x = self.State_To_Features(state, input_dim=pinn_input_dim)
                else:
                    x = self.State_To_Features(state)
                for method_name in ("policy_logits", "Forward", "forward", "__call__"):
                    try:
                        if method_name == "__call__":
                            logits = pinn(x)
                        else:
                            fn = getattr(pinn, method_name, None)
                            if fn is None:
                                continue
                            logits = fn(x)
                        logits = self.Normalize_Logits(logits)
                        if logits is not None:
                            return self.Match_Dim(logits, num_actions)
                    except Exception:
                        continue
            except Exception:
                pass
        # 3) Tabular Q fallback
        rl = getattr(self, "rl", None)
        q_table = getattr(rl, "q", None) if rl is not None else None
        num_states = getattr(self, "num_states", None)
        if q_table is not None and num_states is not None:
            try:
                s = cp.asarray(state).ravel()
                if s.ndim == 0:
                    s = s[None]
                if not cp.issubdtype(s.dtype, cp.integer):
                    s = cp.rint(s).astype(cp.int32)
                else:
                    s = s.astype(cp.int32)
                s = cp.clip(s, 0, int(num_states) - 1)
                logits = self.Normalize_Logits(q_table[s])
                if logits is not None:
                    return self.Match_Dim(logits, num_actions)
            except Exception:
                pass
        # 4) No valid source found
        raise AttributeError(
            "There is no source of logits/policy available!"
            f"backbone={type(backbone).__name__ if pinn is not None else None}, "
            f"pinn={type(pinn).__name__ if pinn is not None else None}, "
            f"rl={'present' if rl is not None else None}, "
            f"q={'present' if q_table is not None else None}, "
            f"num_actions={num_actions}, "
            f"num_states={num_states}"
        )
    """

    def Policy_Logits(self, state):
        num_actions = getattr(self, "num_actions", None)
        if num_actions is None:
            raise AttributeError("number of actions is not initialized!")

        # 1) Backbone
        backbone = getattr(self, "backbone", None)
        if backbone is not None:
            try:
                # x = self.State_To_Features(state, input_dim=getattr(backbone, "input_dim", None))
                backbone_input_dim = getattr(backbone, "input_dim", self.state_dim)
                x = self.State_To_Features(state, input_dim=backbone_input_dim)
                if hasattr(backbone, "Policy_Logits"):
                    logits = backbone.Policy_Logits(x)
                elif hasattr(backbone, "policy_logits"):
                    logits = backbone.policy_logits(x)
                elif hasattr(backbone, "Forward"):
                    logits = backbone.Forward(x)
                elif hasattr(backbone, "forward"):
                    logits = backbone.forward(x)
                else:
                    logits = backbone(x)
                logits = self.Normalize_Logits(logits)
                if logits is not None:
                    return self.Match_Dim(logits, num_actions)
            except Exception as e:
                print("\n============== BACKBONE ERROR ===============")
                print("state shape:", type(state))
                try:
                    print("state shape:", cp.asarray(state).shape)
                except Exception:
                    pass
                traceback.print_exc()
                print("===============================================")

        # 2) PINNs
        pinn = getattr(self, "pinn", None)
        if pinn is not None:
            try:
                pinn_input_dim = self.pinn_input_dim or self.state_dim
                x = self.State_To_Features(state, input_dim=pinn_input_dim)
                if hasattr(pinn, "Predict"):
                    logits = pinn.Predict(x)
                elif hasattr(pinn, "predict"):
                    logits = pinn.predict(x)
                elif hasattr(pinn, "Forward"):
                    logits = pinn.Forward(x)
                elif hasattr(pinn, "forward"):
                    logits = pinn.forward(x)
                else:
                    logits = None
                logits = self.Normalize_Logits(logits)
                if logits is not None:
                    return self.Match_Dim(logits, num_actions)
            except Exception as e:
                print("\n===================== PINN ERROR =====================")
                traceback.print_exc()
                print("========================================================\n")

        # 3) Tabular fallback
        rl = getattr(self, "rl", None)
        q_table = getattr(rl, "q", None) if rl is not None else None
        num_states = getattr(self, "num_states", None)
        if q_table is not None and num_states is not None:
            try:
                s = cp.asarray(state).ravel()
                if s.ndim == 0:
                    s = s[None]
                if not cp.issubdtype(s.dtype, cp.integer):
                    s = cp.rint(s).astype(cp.int32)
                else:
                    s = s.astype(cp.int32)
                s = cp.clip(s, 0, int(num_states) - 1)
                logits = self.Normalize_Logits(q_table[s])
                if logits is not None:
                    return self.Match_Dim(logits, num_actions)
            except Exception:
                print("\n================== RL FALLBACK ERROR ====================")
                traceback.print_exc()
                print("===========================================================\n")
        raise AttributeError(
            "There is no source of logits/policy available! "
            f"backbone={type(backbone).__name__ if backbone is not None else None}, "
            f"pinn={type(pinn).__name__ if pinn is not None else None}, "
            f"rl={'present' if rl is not None else None}, "
            f"q={'present' if q_table is not None else None}, "
            f"num_actions={num_actions}, "
            f"num_states={num_states}"
        )

    def Softmax(
        self,
        logits,
        temperature=TEMPERATURE
    ):
        logits = cp.asarray(logits, dtype=cp.float32)
        temperature = max(float(temperature), EPS2)
        logits = logits / temperature
        logits = logits - cp.max(logits, axis=1, keepdims=True)
        exp_logits = cp.exp(logits)
        return exp_logits / (cp.sum(exp_logits, axis=1, keepdims=True) + EPS2)

    def Entropy_From_Logits(
        self,
        logits,
        temperature=TEMPERATURE
    ):
        p = self.Softmax(logits, temperature=temperature)
        return -cp.sum(p * cp.log(p+EPS2), axis=1)

    def Sample_From_Probs(
        self,
        probs
    ):
        probs = cp.asarray(probs, dtype=cp.float32)
        cdf = cp.cumsum(probs, axis=1)
        u = self.rng.random((probs.shape[0], 1), dtype=cp.float32)
        return cp.argmax(cdf >= u, axis=1).astype(cp.int32)

    def Select_From_Logits(
        self,
        logits,
        strategy="auto",
        epsilon=None,
        temperature=None,
        K=None,
        reward_var=REWARD_VAR,
        entropy=ENTROPY,
        entropy_threshold=ENTROPY_THRESHOLD,
        var_threshold=VAR_THRESHOLD,
        deterministic=False,
    ):
        logits = cp.asarray(logits, dtype=cp.float32)
        if logits.ndim == 1:
            logits = logits[None, :]
        logits = self.Match_Dim(logits, self.num_actions)
        batch = logits.shape[0]
        startegy = strategy.lower().strip()
        if deterministic:
            return cp.argmax(logits, axis=1).astype(cp.int32), logits, None
        if strategy == "auto":
            if reward_var > var_threshold:
                strategy = "epsilon_greedy"
            elif entropy > entropy_threshold:
                strategy = "topk"
            else:
                strategy = "softmax"
        if strategy == "epsilon_greedy":
            eps = self.epsilon if epsilon is None else float(epsilon)
            greedy = cp.argmax(logits, axis=1).astype(cp.int32)
            explore = self.rng.random(batch) < eps
            random_actions = self.rng.integers(0, self.num_actions, size=batch, dtype=cp.int32)
            actions = cp.where(explore, random_actions, greedy).astype(cp.int32)
            return actions, logits, None
        """
        if strategy == "topk":
            K = int(K or K_VALUE)
            K = max(1, min(K, self.num_actions))
            topk_idx = cp.argpartition(logits, topk_idx, axis=1)
            weights = topk_vals - cp.min(topk_vals, axis=1, keepdims=True) + EPS
            weights = cp.maximum(weights, EPS)
            probs = weights / (cp.sum(weights, axis=1, keepdims=True) + EPS2)
            pos = self.Sample_From_Probs(probs)
            actions = topk_idx[cp.arange(batch), pos].astype(cp.int32)
            return actions, logits, probs
        """
        if strategy == "topk":
            K = int(K or getattr(self, "top_k", K_VALUE))
            K = max(1, min(K, self.num_actions))
            topk_idx = cp.argpartition(logits, -K, axis=1)[:, -K:]
            topk_vals = cp.take_along_axis(logits, topk_idx, axis=1)
            weights = topk_vals - cp.min(topk_vals, axis=1, keepdims=True) + EPS
            weights = cp.maximum(weights, EPS)
            probs = weights / (cp.sum(weights, axis=1, keepdims=True) + EPS2)
            pos = self.Sample_From_Probs(probs)
            actions = topk_idx[cp.arange(batch), pos].astype(cp.int32)
            return actions, logits, probs
        # temp = self.temperature if temperature is None else float(temperature)
        temp = float(getattr(self, "temperature", TEMPERATURE)) if temperature is None else float(temperature)
        probs = self.Softmax(logits, temperature=temp)
        actions = self.Sample_From_Probs(probs)
        return actions, logits, probs

    def Environment_Reset(self):
        env = self.vec_env or self.env
        if env is None:
            raise ValueError("env or vec_env is not provided!")
        out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def Environment_Step(
        self,
        actions
    ):
        env = self.vec_env or self.env
        if env is None:
            raise ValueError("env or vec_env is not provided!")
        if self.vec_env is not None:
            return env.step(actions)
        a = int(cp.asarray(actions).ravel()[0].item())
        out = env.step(a)
        if len(out) == 5:
            next_state, reward, terminated, truncated, info = out
        elif len(out) == 4:
            next_state, reward, done, info = out
            terminated, truncated = done, False
        else:
            raise ValueError("env.step output format is not recognized")
        return next_state, reward, terminated, truncated, info

    def Maybe_Reset_Done_Environment(
        self,
        next_state,
        done
    ):
        if self.vec_env is None:
            return next_state
        if hasattr(self.vec_env, "reset_at"):
            done_arr = cp.asarray(done).ravel()
            next_state = cp.asarray(next_state)
            for i, d in enumerate(done_arr):
                if bool(d.item()):
                    s_i, _ = self.vec_env.reset_at(i)
                    next_state[i] = cp.asarray(s_i, dtype=next_state.dtype)
            return next_state
        return next_state

    def Store_Transition(
        self,
        state,
        action,
        reward,
        next_state,
        next_action=None,
        done=False,
        priority=None
    ):
        if next_action is None:
            next_action = action
        buffer = self.replay
        state_arr = cp.asarray(state)
        action_arr = cp.asarray(action)
        reward_arr = cp.asarray(reward)
        next_state_arr = cp.asarray(next_state)
        next_action_arr = cp.asarray(next_action)
        done_arr = cp.asarray(done)
        is_batch = (
            state_arr.ndim >= 2
            and action_arr.ndim >= 1
            and action_arr.size == state_arr.shape[0]
        )
        if is_batch:
            fn = getattr(buffer, "Add_Batch", None) or getattr(buffer, "add_batch", None)
            if callable(fn):
                try:
                    return fn(
                        states=state_arr,
                        actions=action_arr,
                        rewards=reward_arr,
                        next_states=next_state_arr,
                        next_actions=next_action_arr,
                        priority=priority,
                    )
                except TypeError:
                    pass
            for i in range(state_arr.shape[0]):
                self.Store_Transition(
                    state_arr[i],
                    action_arr[i],
                    reward_arr[i],
                    next_state_arr[i],
                    next_action_arr[i] if next_action_arr.ndim >= 1 else next_action_arr,
                    done_arr[i] if done_arr.ndim >= 1 else done_arr,
                    priority=priority,
                )
            return
        fn = getattr(buffer, "Add", None) or getattr(buffer, "add", None)
        if callable(fn):
            return fn(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                done=done,
                priority=priority,
            )
        raise AttributeError("Replay buffer does not have Add/add or Add_Batch/add_batch method!")

    def Sample_Batch(
        self,
        batch_size: int,
        prioritized: bool = False
    ):
        batch_size = int(batch_size)
        buffer = self.replay
        if prioritized and self.use_prioritized_replay:
            fn = getattr(buffer, "Sample_Prioritized", None) or getattr(buffer, "sample_prioritized", None)
            if callable(fn):
                return fn(batch_size)
        fn = getattr(buffer, "Sample_Uniform", None) or getattr(buffer, "sample_uniform", None)
        if callable(fn):
            return fn(batch_size, replace=True)
        raise AttributeError("Replay buffer does not have match sample!")

    def Tabular_Update(
        self,
        batch,
        method="sarsa",
        alpha=None,
        gamma=None,
        lambda_=None,
        tau=None,
        tau_temperature=TAU_TEMPERATURE,
        munchausen_coef=MUNCHAUSEN_COEF,
        log_clip_lower=LOG_CLIP_LOWER
    ):
        if self.rl is None:
            return None
        fn = getattr(self.rl, "Update_Tabular", None) or getattr(self.rl, "update_tabular", None)
        if not callable(fn):
            return None
        try:
            return fn(
                batch,
                method=method,
                alpha=alpha,
                gamma=gamma,
                lambda_=lambda_,
                tau=tau,
                tau_temperature=tau_temperature,
                munchausen_coef=munchausen_coef,
                log_clip_lower=log_clip_lower,
            )
        except Exception:
            return None

    def Build_PINNs_Target(
        self,
        batch,
        gamma=None
    ):
        states = cp.asarray(batch["states"])
        actions = cp.asarray(batch.get("actions", cp.zeros(states.shape[0], dtype=cp.int32))).astype(cp.int32).ravel()
        rewards = cp.asarray(batch["rewards"], dtype=cp.float32).ravel()
        next_states = cp.asarray(batch.get("next_states", states))
        dones = cp.asarray(batch.get("dones", cp.zeros(rewards.shape[0], dtype=cp.bool_)), dtype=cp.bool_).ravel()
        gamma = self.rl.gamma if (gamma is None and self.rl is not None and hasattr(self.rl, "gamma")) else (0.99 if gamma is None else float(gamma))
        # 1) if tabular Q exists, use Q-row supervision
        q_obj = None
        if self.rl is not None:
            q_obj = getattr(self.rl, "q", None)
            if q_obj is None:
                q1 = getattr(self.rl, "q1", None)
                q2 = getattr(self.rl, "q2", None)
                if q1 is not None and q2 is not None:
                    q_obj = 0.5 * (cp.asarray(q1) + cp.asarray(q2))
        if q_obj is not None and self.num_states is not None and states.dtype.kind in "iu":
            s = states.astype(cp.int32).ravel()
            target = cp.asarray(q_obj[s], dtype=cp.float32)
            return self.Match_Dim(target, self.pinn_output_dim or self.num_actions), rewards
        # 2) Bootstrap target from current model/backbone
        try:
            next_logits = self.Policy_Logits(next_states)
            next_values = cp.max(next_logits, axis=1)
        except Exception:
            next_values = cp.zeros_like(rewards)
        td_target = rewards + gamma * next_values * (~dones).astype(cp.float32)
        if (self.pinn_output_dim or self.num_actions or 1) == 1:
            target = td_target[:, None]
        else:
            out_dim = int(self.pinn_output_dim or self.num_actions)
            target = cp.zeros((td_target.size, out_dim), dtype=cp.float32)
            # idx = actions
            # if idx.max(initial=0) < out_dim:
            #     target[cp.arange(td_target.size), idx] = td_target
            idx = cp.asarray(actions, dtype=cp.int32).ravel()
            if idx.size != td_target.size:
                raise ValueError(f"actions size ({idx.size}) != td_target size ({td_target.size})")
            valid = (idx >= 0) & (idx < out_dim)
            if cp.any(valid):
                target[
                    cp.arange(td_target.size)[valid],
                    idx[valid]
                ] = td_target[valid]
        return target.astype(cp.float32, copy=False), rewards

    def Update_Meta(
        self,
        rewards,
        logits=None,
        target_entropy=None,
        lr_global=LEARNING_RATE_GLOBAL,
        lr_local=LEARNING_RATE_LOCAL,
        decay_rate=DECAY_RATE,
        min_epsilon=MIN_EPSILON
    ):
        if self.meta_engine is None:
            return None
        rewards = cp.asarray(rewards, dtype=cp.float32).ravel()
        stats = self.reward_engine.Stats_Stochasticity(rewards)
        mean_reward = stats["mean"]
        var_reward = stats["var"]
        if logits is None:
            mean_entropy_local = cp.asarray([0.0], dtype=cp.float32)
            mean_entropy_global = cp.asarray(0.0, dtype=cp.float32)
        else:
            ent = self.Entropy_From_Logits(logits)
            mean_entropy_local = ent.astype(cp.float32, copy=False)
            mean_entropy_global = cp.mean(ent).astype(cp.float32, copy=False)
        if target_entropy is None:
            target_entropy = cp.log(cp.asarray(max(self.num_actions or 2, 2), dtype=cp.float32))
        try:
            temp_arr, global_temp = self.meta_engine.Adaptive_Temperature_Scheduler(
                mean_entropy_local=mean_entropy_local,
                mean_entropy_global=mean_entropy_global,
                target_entropy=target_entropy,
                lr_global=lr_global,
                lr_local=lr_local,
            )
            eps_arr = self.meta_engine.Epsilon_Decay(
                mean_reward=mean_reward,
                var_reward=var_reward,
                decay_rate=decay_rate,
                min_epsilon=min_epsilon,
            )
        except Exception:
            return None
        self.last_reward_mean = mean_reward
        self.last_reward_var = var_reward
        self.last_entropy = mean_entropy_global
        self.temperature = float(cp.asarray(global_temp).item())
        self.epsilon = float(cp.asarray(eps_arr).mean().item())
        if self.rl is not None:
            if hasattr(self.rl, "temperature"):
                self.rl.temperature = self.temperature
            if hasattr(self.rl, "epsilon"):
                self.rl.epsilon = self.epsilon
        return {
            "temperature_array": temp_arr,
            "global_temperature": global_temp,
            "epsilon_array": eps_arr,
            "mean_reward": mean_reward,
            "reward_var": var_reward,
            "entropy": mean_entropy_global,
        }

    def Select_Action(
        self,
        state,
        strategy="auto",
        epsilon=None,
        temperature=None,
        K=None,
        reward_var=REWARD_VAR,
        entropy=ENTROPY,
        entropy_threshold=ENTROPY_THRESHOLD,
        var_threshold=VAR_THRESHOLD,
        deterministic=False,
        use_tabular_policy=False,
        actions_out=None,
        return_info=False,
    ):
        if use_tabular_policy and self.rl is not None and hasattr(self.rl, "Select_Action"):
            try:
                out = self.rl.Select_Action(
                    states=state,
                    strategy=strategy,
                    use_policy_head=False,
                    hierarchical=False,
                    epsilon=epsilon,
                    temperature=temperature,
                    K=K,
                    reward_var=reward_var,
                    entropy=entropy,
                    entropy_threshold=entropy_threshold,
                    var_threshold=var_threshold,
                )
                if return_info:
                    return out, None
                return out
            except Exception:
                pass
        logits = self.Policy_Logits(state)
        actions, logits, probs = self.Select_From_Logits(
            logits=logits,
            strategy=strategy,
            epsilon=epsilon,
            temperature=temperature,
            K=K,
            reward_var=reward_var,
            entropy=entropy,
            entropy_threshold=entropy_threshold,
            var_threshold=var_threshold,
            deterministic=deterministic,
        )
        if actions_out is not None:
            actions_out = cp.asarray(actions_out, dtype=cp.int32)
            actions_out[: actions.size] = actions
            actions = actions_out
        if return_info:
            return actions, {"logits": logits, "probs": probs}
        return actions

    def Act(
        self,
        state,
        **kwargs
    ):
        return self.Select_Action(state, **kwargs)

    def Collect_Experience(
        self,
        steps: int = STEPS,
        strategy="auto",
        prioritized_replay: bool = False,
        train_after: bool = False,
        batch_size: int = 128,
        train_every: int = TRAIN_EVERY,
        optimizer="adam",
        tabular_method="sarsa",
        physics_batch_fn=None,
        **action_kwargs,
    ):
        state, info = self.reset_env()
        trajectory = []
        for t in range(int(steps)):
            action = self.Select_Action(state, strategy=strategy, **action_kwargs)
            next_state, reward, terminated, truncated, step_info = self.Environment_Step(action)
            done = cp.asarray(terminated) | cp. asarray(truncated)
            next_action = self.Select_Action(next_state, strategy=strategy, **action_kwargs)
            self.Store_Transition(state, action, reward, next_state, next_action, done=done)
            trajectory.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "next_action": next_action,
                    "done": done,
                    "infor": step_info,
                }
            )
            if physics_batch_fn is not None:
                try:
                    physics_extra = physics_batch_fn(t, state, action, reward, next_state, done)
                    if isinstance(physics_extra, dict):
                        trajectory[-1].update(physics_extra)
                except Exception:
                    pass
            state = self.Maybe_Reset_Done_Environment(next_state, done) if self.vec_env is not None else next_state
            if self.vec_env is None and bool(cp.asarray(done).item()):
                state, info = self.reset_env()
            if train_after and len(self.replay) >= batch_size and (t % int(train_every) == 0):
                batch = self.Sample_Batch(batch_size, prioritized=prioritized_replay)
                self.Training_Step(
                    batch=batch,
                    optimizer=optimizer,
                    tabular_method=tabular_method,
                )
        return trajectory

    def Training_Step(
        self,
        batch,
        optimizer="adam",
        tabular_method="sarsa",
        gamma=None,
        lambda_=None,
        tau=None,
        tau_temperature=TAU_TEMPERATURE,
        munchausen_coef=MUNCHAUSEN_COEF,
        log_clip_lower=LOG_CLIP_LOWER,
        data_weight=None,
        physics_weight=None,
        boundary_weight=None,
        use_curiosity=True,
        target_entropy=None,
        lr_global=LEARNING_RATE_GLOBAL,
        lr_local=LEARNING_RATE_LOCAL,
        decay_rate=DECAY_RATE,
        min_epsilon=MIN_EPSILON,
    ):
        if batch is None:
            return None
        batch = dict(batch)
        rewards = cp.asarray(batch["rewards"], dtype=cp.float32).ravel()
        if use_curiosity and self.curiosity_engine is not None:
            if "state_embeddings" in batch and "predicted_embeddings" in batch:
                try:
                    curiosity = self.curiosity_engine.Curiosity_Reward(
                        batch["state_embeddings"],
                        batch["predicted_embeddings"]
                    )
                    rewards = rewards + self.curiosity_weight * cp.asarray(curiosity, dtype=cp.float32).ravel()
                except Exception:
                    pass
            if "log_probs" in batch:
                try:
                    entropy_bonus = self.curiosity_engine.Policy_Entropy(batch["log_probs"], reduce="none")
                    rewards = rewards + self.entropy_weight * cp.asarray(entropy_bonus, dtype=cp.float32).ravel()
                except Exception:
                    pass
        batch["rewards"] = rewards
        # 1) Tabular Reinforcement Learning Update
        rl_update = self.Tabular_Update(
            batch=batch,
            method=tabular_method,
            alpha=None,
            gamma=gamma,
            lambda_=lambda_,
            tau=tau,
            tau_temperature=tau_temperature,
            munchausen_coef=munchausen_coef,
            log_clip_lower=log_clip_lower,
        )
        # 2) Build PINNs supervision target
        targets, reward_stats_input = self.Build_PINNs_Target(batch, gamma=gamma)
        if targets is None:
            return {"rl_update": rl_update, "pinn_update": None}
        x_data = self.State_To_Features(batch["states"], input_dim=self.pinn_input_dim)
        # 3) current logits + proxy loss for logging / action entropy
        try:
            current_logits = self.Policy_Logits(batch["states"])
            current_logits = self.Match_Dim(current_logits, targets.shape[1])
            rl_proxy_loss = cp.mean((current_logits - targets) ** 2)
            entropy = self.Entropy_From_Logits(current_logits)
        except Exception:
            current_logits = None
            rl_proxy_loss = cp.asarray(0.0, dtype=cp.float32)
            entropy = cp.asarray([0.0], dtype=cp.float32)
        # 4) PINNs training (data + optional physics terms from batch)
        pinn_out = None
        if self.pinn is not None:
            x_phys = batch.get("x_phys", None)
            initial_model = batch.get("initial_model", None)
            source_term = batch.get("source_term", None)
            x_bc = batch.get("x_bc", None)
            y_bc = batch.get("y_bc", None)
            fn = getattr(self.pinn, "PINNs_Training_Step", None) or getattr(self.pinn, "Train_Step", None) or getattr(self.pinn, "step", None)
            if callable(fn):
                try:
                    pinn_out = fn(
                        x_data=x_data,
                        y_data=targets,
                        x_phys=x_phys,
                        initial_model=initial_model,
                        source_term=source_term,
                        x_bc=x_bc,
                        y_bc=y_bc,
                        data_weight=data_weight,
                        physics_weight=physics_weight if physics_weight is not None else self.physics_weight,
                        boundary_weight=boundary_weight,
                        data_reduction="mean",
                        physics_reduction="mean",
                        boundary_reduction="mean",
                        optimizer=optimizer,
                        use_legacy_backprop=False,
                        return_details=True,
                    )
                except TypeError:
                    pinn_out = fn(x_data=x_data, y_data=targets, optimizer=optimizer)
        if pinn_out is None:
            pinn_out = {}
        if not isinstance(pinn_out, dict):
            pinn_out = {"total_loss": pinn_out}
        # 5) Meta-learning update
        meta_out = self.Update_Meta(
            rewards=rewards,
            logits=current_logits,
            target_entropy=target_entropy,
            lr_global=lr_global,
            lr_local=lr_local,
            decay_rate=decay_rate,
            min_epsilon=min_epsilon,
        )
        # 6) Priorities (if sampled from replay and td error returned)
        if "indices" in batch and hasattr(self.replay, "Update_Priorities"):
            try:
                if isinstance(rl_update, cp.ndarray) and rl_update.ndim == 1 and rl_update.size == cp.asarray(batch["indices"]).size:
                    self.replay.Update_Priorities(batch["indices"], cp.abs(rl_update) + EPS)
            except Exception:
                pass
        # 7) Normalization hook
        if self.rl is not None and hasattr(self.rl, "Normalize_Q"):
            try:
                self.rl.Normalize_Q(inplace=True)
            except Exception:
                pass
        self.last_metrics = {
            "rl_update": rl_update,
            "pinn_update": pinn_out,
            "meta_update": meta_out,
            "reward_mean": float(cp.asarray(self.last_reward_mean).item()),
            "reward_var": float(cp.asarray(self.last_reward_var).item()),
            "entropy": float(cp.asarray(self.last_entropy).item()),
            "rl_proxy_loss": float(cp.asarray(rl_proxy_loss).item()),
            "converged": bool(getattr(self.pinn, "Is_Convergence", lambda: False)()),
        }
        return self.last_metrics

    def Training(
        self,
        episodes: int = EPISODES,
        max_steps: int = MAX_STEPS,
        batch_size: int = BATCH_SIZE,
        train_every: int = TRAIN_EVERY,
        strategy="auto",
        tabular_method="sarsa",
        render: bool = False,
        physics_batch_fn=None,
        save_every: int | None = None,
        checkpoint_dir: str | None = None,
        **action_kwargs,
    ):
        history = {
            "episode_reward": [],
            "episode_length": [],
            "loss": [],
            "reward_mean": [],
            "reward_var": [],
            "entropy": [],
        }
        for ep in range(int(episodes)):
            state, info = self.reset_env()
            ep_reward = 0.0
            for t in range(int(max_steps)):
                action = self.Select_Action(state, strategy=strategy, **action_kwargs)
                next_state, reward, terminated, truncated, step_info = self.Environment_Step(action)
                done = cp.asarray(terminated) | cp.asarray(truncated)
                next_action = self.Select_Action(next_state, strategy=strategy, **action_kwargs)
                self.Store_Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_action=next_action,
                    done=done,
                )
                if physics_batch_fn is not None:
                    try:
                        phys_extra = physics_batch_fn(ep, t, state, action, reward, next_state, done)
                        if isinstance(phys_extra, dict):
                            pass
                    except Exception:
                        pass
                ep_reward += float(cp.asarray(reward).mean().item()) if cp.asarray(reward).ndim > 0 else float(cp.asarray(reward).item())
                if len(self.replay) >= batch_size and (t % int(train_every) == 0):
                    batch = self.Sample_Batch(batch_size, prioritized=prioritized_replay)
                    out = self.Training_Step(
                        batch=batch,
                        optimizer=optimizer,
                        tabular_method=tabular_method,
                        physics_batch_fn=physics_batch_fn,
                    )
                    loss_val = 0.0
                    if isinstance(out, dict) and "pinn_update" in out and isinstance(out["pinn_update"], dict) and "total_loss" in out["pinn_update"]:
                        loss_val = float(cp.asarray(out["pinn_update"]["total_loss"]).item())
                    history["loss"].append(loss_val)
                state = self.Maybe_Reset_Done_Environment(next_state, done) if self.vec_env is not None else next_state
                if render and hasattr(self.env or self.vec_env, "render"):
                    try:
                        (self.env or self.vec_env).render()
                    except Exception:
                        pass
                if self.vec_env is None and bool(cp.asarray(done).item()):
                    break
            history["episode_reward"].append(ep_reward)
            history["episode_length"].append(t+1)
            history["reward_mean"].append(float(cp.asarray(self.last_reward_mean).item()))
            history["reward_var"].append(float(cp.asarray(self.last_reward_var).item()))
            history["entropy"].append(float(cp.asarray(self.last_entropy).item()))
            if save_every is not None and checkpoint_dir is not None and ((ep+1) % int(save_every) == 0):
                self.save(f"{checkpoint_dir.rstrip('/')}/pirl_ep_{ep+1}.pkl")
        return history

    def Evaluate(
        self,
        episodes: int = EPISODES,
        max_steps: int = MAX_STEPS,
        deterministic: bool = True
    ):
        rewards = []
        for _ in range(int(episodes)):
            state, info = self.reset_env()
            ep_reward = 0.0
            for _t in range(int(max_steps)):
                action = self.Select_Action(state, deterministic=deterministic)
                next_state, reward, terminated, truncated, step_info = self.Environment_Step(action)
                done = cp.asarray(terminated) | cp.asarray(truncated)
                ep_reward += float(cp.asarray(reward).mean().item()) if cp.asarray(reward).ndim > 0 else float(cp.asarray(reward).item())
                state = next_state
                if self.vec_env is None and bool(cp.asarray(done).item()):
                    break
            rewards.append(ep_reward)
        rewards = cp.asarray(rewards, dtype=cp.float32)
        return {
            "mean_reward": float(cp.mean(rewards).item()),
            "std_reward": float(cp.std(rewards).item()),
            "episode_rewards": rewards,
        }

    def State_Dict(
        self,
        include_replay: bool = False
    ):
        state = {
            "num_actions": self.num_actions,
            "num_states": self.num_states,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "physics_weight": self.physics_weight,
            "rl_weight": self.rl_weight,
            "curiosity_weight": self.curiosity_weight,
            "entropy_weight": self.entropy_weight,
            "normalize_states": self.normalize_states,
            "normalize_rewards": self.normalize_rewards,
            "pinn_state": self.Call_First(self.pinn, ["State_Dict", "state_dict"], default=None),
            "rl_state": self.Call_First(self.rl, ["State_Dict", "state_dict"], default=None),
            "backbone_state": self.Call_First(self.backbone, ["State_Dict", "state_dict"], default=None),
            "meta_state": self.Call_First(self.meta_engine, ["get_state"], default=None),
            "last_reward_mean": self.last_reward_mean,
            "last_reward_var": self.last_reward_var,
            "last_entropy": self.last_entropy,
        }
        if include_replay and hasattr(self.replay, "Get_All"):
            try:
                state["replay"] = self.replay.Get_All()
            except Exception:
                state["replay"] = None
        return state

    def Load_State_Dict(
        self,
        state
    ):
        self.num_actions = state.get("num_actions", self.num_actions)
        self.num_states = state.get("num_states", self.num_states)
        self.state_dim = state.get("state_dim", self.state_dim)
        self.hidden_dim = state.get("hidden_dim", self.hidden_dim)
        self.physics_weight = state.get("physics_weight", self.physics_weight)
        self.rl_weight = state.get("rl_weight", self.rl_weight)
        self.curiosity_weight = state.get("curiosity_weight", self.curiosity_weight)
        self.entropy_weight = state.get("entropy_weight", self.entropy_weight)
        self.normalize_states = state.get("normalize_states", self.normalize_states)
        self.normalize_rewards = state.get("normalize_rewards", self.normalize_rewards)
        if self.pinn is not None and state.get("pinn_state") is not None:
            fn = getattr(self.pinn, "Load_State_Dict", None) or getattr(self.pinn, "load_state_dict", None)
            if callable(fn):
                fn(state["pinn_state"])
        if self.rl is not None and state.get("rl_state") is not None:
            fn = getattr(self.rl, "Load_State_Dict", None) or getattr(self.rl, "load_state_dict", None)
            if callable(fn):
                fn(state["rl_state"])
        if self.backbone is not None and state.get("backbone_state") is not None:
            fn = getattr(self.backbone, "Load_State_Dict", None) or getattr(self.backbone, "load_state_dict", None)
            if callable(fn):
                fn(state["backbone_state"])
        if self.meta_engine is not None and state.get("meta_state") is not None:
            if hasattr(self.meta_engine, "temperature_array") and "temperature_array" in state["meta_state"]:
                self.meta_engine.temperature_array = cp.asarray(state["meta_state"]["temperature_array"])
            if hasattr(self.meta_engine, "global_temperature") and "global_temperature" in state["meta_state"]:
                self.meta_engine.global_temperature = cp.asarray(state["meta_state"]["global_temperature"])
            if hasattr(self.meta_engine, "epsilon") and "epsilon" in state["meta_state"]:
                self.meta_engine.epsilon = cp.asarray(state["meta_state"]["epsilon"])
        self.last_reward_mean = cp.asarray(state.get("last_reward_mean", 0.0), dtype=cp.float32)
        self.last_reward_var = cp.asarray(state.get("last_reward_var", 0.0), dtype=cp.float32)
        self.last_entropy = cp.asarray(state.get("last_entropy", 0.0), dtype=cp.float32)
        self.Sync_Dimensions()
        return self

    def Save(
        self,
        path: str,
        include_replay: bool = False
    ):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(include_replay=include_replay), f)

    def Load(
        self,
        path: str
    ):
        with open(path, "rb") as f:
            state = pickle.load(f)
        return self.load_state_dict(state)

    def Reset(self):
        if self.pinn is not None:
            self.Call_First(self.pinn, ["Reset", "reset"])
        if self.rl is not None:
            self.Call_First(self.rl, ["Reset", "reset"])
        if hasattr(self.replay, "Clear"):
            try:
                self.replay.Clear()
            except Exception:
                pass
        self.last_reward_mean = cp.asarray(0.0, dtype=cp.float32)
        self.last_reward_var = cp.asarray(0.0, dtype=cp.float32)
        self.last_entropy = cp.asarray(0.0, dtype=cp.float32)
        self.last_metrics = {}
        return self

























