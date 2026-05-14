import copy
import logging
import os
import shutil
import signal
import sys
import threading
import time
from typing import Optional, List
from typing import Union

from omegaconf import DictConfig
import gc
import numpy as np
import psutil
import torch
import pandas as pd
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.train_runner import TrainRunner
from yarr.utils.log_writer import LogWriter
from yarr.utils.stat_accumulator import StatAccumulator
from yarr.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer


class OfflineTrainRunner():

    def __init__(self,
                 agent: Agent,
                 wrapped_replay_buffer: PyTorchReplayBuffer,
                 train_device: torch.device,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(6e6),
                 logdir: str = '/tmp/yarr/logs',
                 logging_level: int = logging.INFO,
                 log_freq: int = 10,
                 weightsdir: str = '/tmp/yarr/weights',
                 num_weights_to_keep: int = 60,
                 save_freq: int = 100,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False,
                 wandb_logging: bool = False,
                 wandb_project: str = None,
                 wandb_entity: str = None,
                 wandb_group: str = None,
                 wandb_name: str = None,
                 wandb_tags=None,
                 wandb_mode: str = None,
                 wandb_config: dict = None,
                 load_existing_weights: bool = True,
                 rank: int = None,
                 world_size: int = None,
                 val_wrapped_replay_buffer: Optional[PyTorchReplayBuffer] = None,
                 val_loss_freq: int = 0,
                 val_loss_batches: int = 8):
        self._agent = agent
        self._wrapped_buffer = wrapped_replay_buffer
        self._stat_accumulator = stat_accumulator
        self._iterations = iterations
        self._logdir = logdir
        self._logging_level = logging_level
        self._log_freq = log_freq
        self._weightsdir = weightsdir
        self._num_weights_to_keep = num_weights_to_keep
        self._save_freq = save_freq

        self._wrapped_buffer = wrapped_replay_buffer
        self._train_device = train_device
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        self._wandb_logging = wandb_logging
        self._load_existing_weights = load_existing_weights
        self._rank = rank
        self._world_size = world_size

        self._val_wrapped = val_wrapped_replay_buffer
        self._val_loss_freq = int(val_loss_freq or 0)
        self._val_loss_batches = max(1, int(val_loss_batches))

        self._writer = None
        if logdir is None:
            logging.info("'logdir' was None. No logging will take place.")
        else:
            self._writer = LogWriter(
                self._logdir,
                tensorboard_logging,
                csv_logging,
                wandb_logging=wandb_logging,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_group=wandb_group,
                wandb_name=wandb_name,
                wandb_tags=wandb_tags,
                wandb_mode=wandb_mode,
                wandb_config=wandb_config)

        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)

    def _save_model(self, i):
        d = os.path.join(self._weightsdir, str(i))
        os.makedirs(d, exist_ok=True)
        self._agent.save_weights(d)

        # remove oldest save
        prev_dir = os.path.join(self._weightsdir, str(
            i - self._save_freq * self._num_weights_to_keep))
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)

    def _step(self, i, sampled_batch):
        update_dict = self._agent.update(i, sampled_batch)
        total_losses = update_dict['total_losses'].item()
        return total_losses

    def _get_resume_eval_epoch(self):
        starting_epoch = 0
        eval_csv_file = self._weightsdir.replace('weights', 'eval_data.csv') # TODO(mohit): check if it's supposed be 'env_data.csv'
        if os.path.exists(eval_csv_file):
             eval_dict = pd.read_csv(eval_csv_file).to_dict()
             epochs = list(eval_dict['step'].values())
             return epochs[-1] if len(epochs) > 0 else starting_epoch
        else:
            return starting_epoch

    def start(self):
        logging.getLogger().setLevel(self._logging_level)
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device)

        if self._weightsdir is not None:
            existing_weights = sorted([int(f) for f in os.listdir(self._weightsdir)])
            if (not self._load_existing_weights) or len(existing_weights) == 0:
                self._save_model(0)
                start_iter = 0
            else:
                resume_iteration = existing_weights[-1]
                self._agent.load_weights(os.path.join(self._weightsdir, str(resume_iteration)))
                start_iter = resume_iteration + 1
                if self._rank == 0:
                    logging.info(f"Resuming training from iteration {resume_iteration} ...")

        dataset = self._wrapped_buffer.dataset()
        data_iter = iter(dataset)

        val_data_iter = None
        if (
            self._val_wrapped is not None
            and self._val_loss_freq > 0
            and hasattr(self._agent, "compute_validation_loss")
        ):
            val_data_iter = iter(self._val_wrapped.dataset())

        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        for i in range(start_iter, self._iterations):
            log_iteration = i % self._log_freq == 0 and i > 0

            if log_iteration:
                process.cpu_percent(interval=None)

            t = time.time()
            sampled_batch = next(data_iter)
            sample_time = time.time() - t

            batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if isinstance(v, torch.Tensor)}
            if "lang_goal" in sampled_batch:
                batch["lang_goal"] = sampled_batch["lang_goal"]
            t = time.time()
            loss = self._step(i, batch)
            step_time = time.time() - t

            if self._rank == 0:
                run_val = (
                    val_data_iter is not None
                    and self._val_loss_freq > 0
                    and i > 0
                    and i % self._val_loss_freq == 0
                )
                if log_iteration and self._writer is not None:
                    agent_summaries = self._agent.update_summaries()
                    self._writer.add_summaries(i, agent_summaries)

                    self._writer.add_scalar(
                        i, 'monitoring/memory_gb',
                        process.memory_info().rss * 1e-9)
                    self._writer.add_scalar(
                        i, 'monitoring/cpu_percent',
                        process.cpu_percent(interval=None) / num_cpu)

                    logging.info(f"Train Step {i:06d} | Loss: {loss:0.5f} | Sample time: {sample_time:0.6f} | Step time: {step_time:0.4f}.")

                if run_val and self._writer is not None:
                    n_b = self._val_loss_batches
                    totals = []
                    sub_keys = None
                    sub_sums = None
                    for _ in range(n_b):
                        sampled_val = next(val_data_iter)
                        vbatch = {
                            k: v.to(self._train_device)
                            for k, v in sampled_val.items()
                            if isinstance(v, torch.Tensor)
                        }
                        if "lang_goal" in sampled_val:
                            vbatch["lang_goal"] = sampled_val["lang_goal"]
                        vout = self._agent.compute_validation_loss(vbatch)
                        totals.append(vout["total_loss"])
                        if sub_keys is None:
                            sub_keys = [k for k in vout if k.startswith("losses/")]
                            sub_sums = {k: 0.0 for k in sub_keys}
                        for k in sub_keys:
                            sub_sums[k] += vout[k]
                    mean_val = float(sum(totals) / len(totals))
                    self._writer.add_scalar(i, "val_diffusion/total_loss", mean_val)
                    for k in sub_keys:
                        short = k.replace("losses/", "")
                        self._writer.add_scalar(
                            i, f"val_diffusion/{short}", sub_sums[k] / n_b
                        )
                    logging.info(
                        "Train Step %06d | val_diffusion/total_loss: %0.5f (%d batches).",
                        i, mean_val, n_b,
                    )

                if self._writer is not None:
                    self._writer.end_iteration()

                if i % self._save_freq == 0 and self._weightsdir is not None:
                    self._save_model(i)

        if self._rank == 0 and self._writer is not None:
            self._writer.close()
            logging.info('Stopping envs ...')

            self._wrapped_buffer.replay_buffer.shutdown()
            if self._val_wrapped is not None:
                self._val_wrapped.replay_buffer.shutdown()

