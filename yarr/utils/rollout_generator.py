from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition


class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    @staticmethod
    def _stack_history_list(v_list):
        """Stack per-timestep numpy entries to (T, ...) for batch dim B=1 in generator."""
        if len(v_list) == 0:
            raise ValueError("empty observation history")
        el0 = v_list[0]
        if isinstance(el0, np.ndarray) and el0.dtype != object:
            stacked = np.stack([np.asarray(x, dtype=el0.dtype) for x in v_list], axis=0)
            return stacked
        raise TypeError(
            "RolloutGenerator cannot stack observation history for type %r; "
            "expected numeric numpy arrays." % type(el0)
        )

    def _history_batch_to_tensors(self, obs_history):
        """Convert obs_history dict of length-T lists to tensors (1, T, ...) on env device."""
        dev = self._env_device
        out = {}
        for k, v in obs_history.items():
            if len(v) == 0:
                continue
            el0 = v[0]
            if isinstance(el0, np.ndarray) and el0.dtype != object:
                stacked = self._stack_history_list(v)
                t = torch.from_numpy(stacked).to(dev)
                if t.dtype == torch.float64:
                    t = t.float()
                out[k] = t.unsqueeze(0)
            else:
                # Non-numeric (e.g. lang strings): keep latest only, shape (1, 1, ...)
                last = v[-1]
                if isinstance(last, np.ndarray):
                    out[k] = torch.from_numpy(np.asarray(last)).unsqueeze(0).unsqueeze(0).to(dev)
                else:
                    out[k] = last
        return out

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
        else:
            obs = env.reset()

        agent.reset()
        obs_history = {}
        for k, v in obs.items():
            if not isinstance(v, np.ndarray):
                continue
            obs_history[k] = [np.array(v, dtype=self._get_type(v))] * timesteps
        for step in range(episode_length):

            prepped_data = self._history_batch_to_tensors(obs_history)

            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = self._history_batch_to_tensors(obs_history)
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and (
                transition.terminal or timeout or step == episode_length - 1
            ):
                arm_mode = env.env._action_mode.arm_action_mode
                if hasattr(arm_mode, "record_end"):
                    arm_mode.record_end(env.env._scene, steps=60, step_scene=True)

            obs = dict(transition.observation)
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
