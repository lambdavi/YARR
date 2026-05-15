import csv
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from yarr.agents.agent import ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary, TextSummary
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None


class LogWriter(object):

    def __init__(self,
                 logdir: str,
                 tensorboard_logging: bool,
                 csv_logging: bool,
                 wandb_logging: bool = False,
                 wandb_project: str = None,
                 wandb_entity: str = None,
                 wandb_group: str = None,
                 wandb_name: str = None,
                 wandb_tags=None,
                 wandb_mode: str = None,
                 wandb_config: dict = None,
                 train_csv: str = 'train_data.csv',
                 env_csv: str = 'env_data.csv'):
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        self._wandb_logging = wandb_logging
        self._wandb = None
        os.makedirs(logdir, exist_ok=True)
        if tensorboard_logging:
            self._tf_writer = SummaryWriter(logdir)
        if wandb_logging:
            if wandb is None:
                logging.warning("wandb_logging=True but wandb is not installed. Disabling wandb logging.")
                self._wandb_logging = False
            else:
                init_kwargs = {
                    'project': wandb_project,
                    'entity': wandb_entity,
                    'group': wandb_group,
                    'name': wandb_name,
                    'dir': logdir,
                    'tags': list(wandb_tags) if wandb_tags is not None else None,
                    'config': wandb_config,
                }
                if wandb_mode is not None:
                    init_kwargs['mode'] = wandb_mode
                init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
                wandb.init(**init_kwargs)
                self._wandb = wandb
        if csv_logging:
            self._train_prev_row_data = self._train_row_data = OrderedDict()
            self._train_csv_file = os.path.join(logdir, train_csv)
            self._env_prev_row_data = self._env_row_data = OrderedDict()
            self._env_csv_file = os.path.join(logdir, env_csv)
            self._train_field_names = None
            self._env_field_names = None

    def wandb_update_summary(self, metrics: dict, log_step: int = 0):
        """Record one-time metrics in wandb summary and optionally at a training step."""
        if not self._wandb_logging or self._wandb is None:
            return
        self._wandb.summary.update(metrics)
        if log_step is not None:
            self._wandb.log(dict(metrics), step=int(log_step))

    def add_scalar(self, i, name, value):
        if self._tensorboard_logging:
            self._tf_writer.add_scalar(name, value, i)
        if self._wandb_logging:
            v = value.item() if isinstance(value, torch.Tensor) else value
            if isinstance(i, (int, np.integer)):
                wandb.log({name: v}, step=int(i))
            else:
                wandb.log({name: v})
        if self._csv_logging:
            if 'env' in name or 'eval' in name or 'test' in name:
                if len(self._env_row_data) == 0:
                    self._env_row_data['step'] = i
                self._env_row_data[name] = value.item() if isinstance(
                    value, torch.Tensor) else value
            else:
                if len(self._train_row_data) == 0:
                    self._train_row_data['step'] = i
                self._train_row_data[name] = value.item() if isinstance(
                    value, torch.Tensor) else value

    @staticmethod
    def _infer_image_dataformat(img):
        """Infer TensorBoard image format for 2D/3D numpy or torch arrays."""
        shape = img.shape
        if len(shape) == 2:
            return 'HW'
        if len(shape) != 3:
            # Fallback for unexpected shapes.
            return 'CHW'

        c_first = shape[0] in (1, 3, 4)
        c_last = shape[2] in (1, 3, 4)
        if c_last and not c_first:
            return 'HWC'
        return 'CHW'

    def add_summaries(self, i, summaries):
        for summary in summaries:
            try:
                if isinstance(summary, ScalarSummary):
                    self.add_scalar(i, summary.name, summary.value)
                elif self._tensorboard_logging:
                    if isinstance(summary, HistogramSummary):
                        self._tf_writer.add_histogram(
                            summary.name, summary.value, i)
                    elif isinstance(summary, ImageSummary):
                        # Only grab first item in batch
                        v = (summary.value if summary.value.ndim == 3 else
                             summary.value[0])
                        dataformats = self._infer_image_dataformat(v)
                        self._tf_writer.add_image(
                            summary.name, v, i, dataformats=dataformats)
                    elif isinstance(summary, VideoSummary):
                        # Only grab first item in batch
                        v = (summary.value if summary.value.ndim == 5 else
                             np.array([summary.value]))
                        self._tf_writer.add_video(
                            summary.name, v, i, fps=summary.fps)
                    elif isinstance(summary, TextSummary):
                        self._tf_writer.add_text(summary.name, summary.value, i)
                if self._wandb_logging:
                    if isinstance(summary, HistogramSummary):
                        payload = {summary.name: wandb.Histogram(summary.value)}
                    elif isinstance(summary, ImageSummary):
                        v = (summary.value if summary.value.ndim == 3 else
                             summary.value[0])
                        payload = {summary.name: wandb.Image(v)}
                    elif isinstance(summary, VideoSummary):
                        v = (summary.value if summary.value.ndim == 5 else
                             np.array([summary.value]))
                        payload = {summary.name: wandb.Video(v, fps=summary.fps, format='mp4')}
                    elif isinstance(summary, TextSummary):
                        payload = {summary.name: str(summary.value)}
                    else:
                        payload = None

                    if payload is not None:
                        if isinstance(i, (int, np.integer)):
                            wandb.log(payload, step=int(i))
                        else:
                            wandb.log(payload)
            except Exception as e:
                logging.error('Error on summary: %s' % summary.name)
                raise e

    def end_iteration(self):
        # write train data
        if self._csv_logging and len(self._train_row_data) > 0:
            should_write_train_header = not os.path.exists(self._train_csv_file)
            with open(self._train_csv_file, mode='a+') as csv_f:
                names = self._train_row_data.keys()
                writer = csv.DictWriter(csv_f, fieldnames=names)
                if should_write_train_header:
                    if self._train_field_names is None:
                        writer.writeheader()
                    else:
                        if not np.array_equal(self._train_field_names, self._train_row_data.keys()):
                            # Special case when we are logging faster than new
                            # summaries are coming in.
                            missing_keys = list(set(self._train_field_names) - set(
                                self._train_row_data.keys()))
                            for mk in missing_keys:
                                self._train_row_data[mk] = self._train_prev_row_data[mk]
                self._train_field_names = names
                try:
                    writer.writerow(self._train_row_data)
                except Exception as e:
                    print(e)
            self._train_prev_row_data = self._train_row_data
            self._train_row_data = OrderedDict()

        # write env data (also eval or test during evaluation)
        if self._csv_logging and len(self._env_row_data) > 0:
            should_write_env_header = not os.path.exists(self._env_csv_file)
            with open(self._env_csv_file, mode='a+') as csv_f:
                names = self._env_row_data.keys()
                writer = csv.DictWriter(csv_f, fieldnames=names)
                if should_write_env_header:
                    if self._env_field_names is None:
                        writer.writeheader()
                    else:
                        if not np.array_equal(self._env_field_names, self._env_row_data.keys()):
                            # Special case when we are logging faster than new
                            # summaries are coming in.
                            missing_keys = list(set(self._env_field_names) - set(
                                self._env_row_data.keys()))
                            for mk in missing_keys:
                                self._env_row_data[mk] = self._env_prev_row_data[mk]
                self._env_field_names = names
                try:
                    writer.writerow(self._env_row_data)
                except Exception as e:
                    print(e)
            self._env_prev_row_data = self._env_row_data
            self._env_row_data = OrderedDict()

    def close(self):
        if self._tensorboard_logging:
            self._tf_writer.close()
        if self._wandb_logging:
            wandb.finish()

