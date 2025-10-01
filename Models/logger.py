import os
import logging
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from Utils.tools import is_image_pth, PlotBoard
from typing import Optional, Iterable

class Logger:
    def __init__(self, log_dir="logs", log_name="train.log"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_name)

        logging.basicConfig(
            filename=self.log_path,
            filemode="w",
            level=logging.INFO,
            format="%(message)s"
        )
        self.logger = logging.getLogger()


        self.history = {
            "train_step": [],
            "train_loss": [],
            "val_step": [],
            "val_loss": [],
            "val_metrics": {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "confusion_matrix": []
            },
            "lr": []
        }

    def log_base_info(self, model_name, hyperparams, dataset, device):
        self.logger.info(f"Experiment: {model_name}")
        self.logger.info(f"Hyperparams: {json.dumps(hyperparams)}")
        self.logger.info(f"Dataset: {dataset}")
        self.logger.info(f"Device: {device}")

    def log_user_info(self, *args):
        msg = ""
        for item in args:
            msg += item
        self.logger.info(msg)

    def log(
            self, train_step=None, train_loss=None, val_step=None, val_metrics=None, lr=None, time=None
    ):
        msg = ""
        if train_step is not None and val_step is not None:
            assert train_step == val_step, "when you try to integrate train_step and val_step, please make them equal"
        step = train_step or val_step
        if train_loss is not None:
            msg += f"Step [{step}] | Train Loss: {train_loss:.6f}\n"
        if val_metrics is not None:
            for tag, value in val_metrics.items():
                msg += f"Step [{step}] | {tag}: {_format_metric(value)}\n"
        if lr is not None:
            msg += f"Step [{step}] | LR: {lr:.6f}\n"
        if time is not None:
            msg += f"Step [{step}] | Time: {time}s\n"
        self.logger.info(msg)

        if train_step is not None:
            self.history["train_step"].append(step)
        if train_loss is not None:
            self.history["train_loss"].append(train_loss)
        if val_step is not None:
            self.history["val_step"].append(step)
        if val_metrics is not None and val_metrics.get('val_loss') is not None:
            self.history["val_loss"].append(val_metrics["val_loss"])
        if lr is not None:
            self.history["lr"].append(lr) 
        for key, value in self.history["val_metrics"].items():
            if val_metrics is not None and val_metrics[key] is not None: 
                value.append(val_metrics[key])


    def save_history(self, path=None):
        path = path or self.log_path.replace(".log", "_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)

    def close_logger(self):
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    

class LogVisualizer:
    def __init__(self, history, figsize):
        self.history = history
        self.plotboard = PlotBoard(figsize)

    def plot_loss(self, loss_type, save_pth=None):
        assert is_image_pth(save_pth)

        if loss_type == 'train':
            x_raw = self.history.get("train_step", [])
            y_raw = self.history.get("train_loss", [])
        elif loss_type == 'val':
            x_raw = self.history.get("val_step", [])
            y_raw = self.history.get("val_loss", [])
        else:
            raise ValueError(f"Unknown loss type {loss_type!r}")
    
        x_arr = _list_to_numpy(x_raw)
        y_arr = _list_to_numpy(y_raw)
    
        if x_arr.size == 0 and y_arr.size > 0:
            x_arr = np.arange(len(y_arr))
        if y_arr.size == 0:
            print(f"No {loss_type} loss data to plot.")
            return
    
        
        plt.figure()
        axes = plt.gca()
        try:
            self.plotboard.plot(X=x_arr.tolist(), Y=y_arr.tolist(),
                                xlabel='step', ylabel='loss', axes=axes)
        except Exception as e:
            print("plotboard.plot failed, fallback to matplotlib. Error:", e)
            axes.cla()
            axes.plot(x_arr, y_arr, marker='o')
            axes.set_xlabel('step')
            axes.set_ylabel('loss')
            axes.grid()
            
        plt.tight_layout()
        plt.savefig(save_pth, bbox_inches='tight')
        plt.close()

    def plot_metrics(self, save_pth: Optional[str] = None):
        assert is_image_pth(save_pth)
        x_raw = self.history.get("val_step", [])
        x_arr = _seq_to_numpy(x_raw)
    
        y_list = []
        legend = []
        val_metrics = self.history.get("val_metrics", {})
        if not isinstance(val_metrics, dict):
            return
    
        for key, value in val_metrics.items():
            if key == "confusion_matrix":
                continue
            arr = _seq_to_numpy(value)
            if arr.size == 0:
                continue
            legend.append(key)
            y_list.append(arr.tolist())
    
        if not y_list:
            return
    
        if x_arr.size == 0:
            length = len(y_list[0])
            x_arr = np.arange(length)
    
        x_len = x_arr.shape[0]
        aligned_y = []
        for y in y_list:
            if len(y) == x_len:
                aligned_y.append(y)
            elif len(y) > x_len:
                aligned_y.append(y[:x_len])
            else:
                y_ext = list(y) + [float('nan')] * (x_len - len(y))
                aligned_y.append(y_ext)
    
        plt.figure()
        axes = plt.gca()
        try:
            self.plotboard.plot(X=x_arr.tolist(), Y=aligned_y, xlabel='step', ylabel='metrics', legend=legend, axes=axes)
        except Exception as e:
            print("plotboard.plot failed, fallback to matplotlib. Error:", e)
            axes.cla()
            for y, lg in zip(aligned_y, legend):
                axes.plot(x_arr, y, marker='o', label=lg)
            axes.set_xlabel('step')
            axes.set_ylabel('metrics')
            axes.legend()
            axes.grid()
    
        plt.tight_layout()
        plt.savefig(save_pth, bbox_inches='tight')
        plt.close()

def _to_scalar(x):
    if x is None:
        return None
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        else:
            return float(x.detach().cpu().mean().item())
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.flatten()[0])
        else:
            return float(x.mean())
    try:
        return float(x)
    except Exception:
        return None

def _list_to_numpy(seq: Iterable):
    vals = []
    for e in seq:
        v = _to_scalar(e)
        if v is None:
            continue
        vals.append(v)
    if not vals:
        return np.array([], dtype=float)
    return np.array(vals, dtype=float)


def _seq_to_numpy(seq: Iterable):
    if isinstance(seq, torch.Tensor):
        try:
            arr = seq.detach().cpu().numpy()
        except Exception:
            arr = np.array([])

        if arr.ndim == 1:
            return arr.astype(float)
        else:
            return np.array([np.mean(v) for v in arr]).astype(float)

    if isinstance(seq, np.ndarray):
        if seq.ndim == 1:
            return seq.astype(float)
        else:
            return np.array([np.mean(v) for v in seq]).astype(float)

    vals = []
    for e in seq:
        v = _to_scalar(e)
        if v is None:
            continue
        vals.append(v)
    if not vals:
        return np.array([], dtype=float)
    return np.array(vals, dtype=float)

def _format_metric(value):
    if value is None:
        return "None"
    try:
        import torch
        if torch.is_tensor(value):
            if value.numel() == 1:
                return f"{float(value.item()):.6f}"
            return str(value.tolist())
    except Exception:
        pass
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)
