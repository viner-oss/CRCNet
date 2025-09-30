import torch
import os.path
from torch.utils.data import DataLoader
from Models.logger import Logger, LogVisualizer
from Utils.tools_setting import *
from Utils.tools import *
from Utils.evaluator import *
from typing import Iterable, Dict, Any, List, Optional

class TrainLogitsFoldLoop:
    def __init__(self,
                 Ts,
                 warmup_T,
                 fold,
                 root_dir,
                 save_abs_dir,
                 log_abs_dir,
                 metrics_abs_dir,
                 ds_name,
                 ds,
                 ds_list,
                 num_classes,
                 batch,
                 accumulation_steps,
                 diffusion_name,
                 diffusion_dict,
                 model_name,
                 model_dict,
                 opt_name,
                 opt_dict,
                 lr_scheduler_name,
                 lr_dict,
                 criterion_name,
                 criterion_dict,
                 ema_decay,
                 log_interval,
                 val_interval,
                 device,
                 checkpoint_pth,
                 use_diffusion=False,
                 use_checkpoint=False,
                 use_timer=True,
                 use_earlystopper=True,
                 use_logger=True,
                 use_ema_model=True,
                 **kwargs
    ):
        self.step = 0
        self.Ts = Ts
        self.warmup_T = warmup_T
        self.fold = fold
        self.log_root_dir = os.path.join(root_dir, log_abs_dir)
        self.save_ckpt_pth = str(os.path.join(root_dir, save_abs_dir, f"fold_{fold+1}.pt"))
        self.metrics_pth = str(os.path.join(root_dir, metrics_abs_dir, f'fold_{fold+1}'))
        self.ds_name = ds_name
        self.ds = ds
        self.ds_list = ds_list
        self.proportions = self.ds.get_proportions(num_classes)
        self.num_classes = num_classes
        self.batch = batch
        self.accumulation_steps = accumulation_steps
        self.diffusion_name = diffusion_name
        self.diffusion_dict = diffusion_dict
        self.model_name = model_name
        self.model_dict = model_dict
        self.opt_name = opt_name
        self.opt_dict = opt_dict
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_dict = lr_dict
        self.criterion_name = criterion_name
        self.criterion_dict = criterion_dict
        self.ema_decay = ema_decay
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.device = device
        self.checkpoint_pth = checkpoint_pth
        self.lr = lr_dict["lr"]

        self.use_diffusion = use_diffusion
        self.use_checkpoint = use_checkpoint
        self.use_timer = use_timer
        self.use_earlystopper = use_earlystopper
        self.use_logger = use_logger
        self.use_ema_model = use_ema_model

        self.hyperparameters = warp_hyperparameters(
            self.diffusion_dict, True,
            self.opt_dict, self.lr_dict,
             proportions=self.proportions, num_classes=self.num_classes, batch=self.batch, accumulation_steps=self.accumulation_steps,
             ema_decay=self.ema_decay, log_interval=self.log_interval, val_interval=self.val_interval
        )

        make_dirs(self.metrics_pth)

    def _init_logger(self):
        assert self.use_logger, f"Strongly suggest using logger"
        self.logger = Logger(self.log_root_dir, f'fold_{self.fold+1}.log')

    def _init_diffusion(self):
        if self.use_diffusion:
            self.diffusion = get_noise_scheduler(
                self.diffusion_name, **self.diffusion_dict
            )

    def _init_model(self):
        self.model = get_model(self.model_name, **self.model_dict)

    def _init_opt(self):
        self.opt = get_optimizer(self.opt_name, self.model.parameters(), **self.opt_dict.update(lr=self.lr))

    def _init_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(
            self.lr_scheduler_name, self.opt, **self.lr_dict.update(
                warmup_T=self.warmup_T, Ts=self.Ts
            )
        )

    def _init_criterion(self):
        self.criterion = get_criterion(
            self.criterion_name, **self.criterion_dict.update(proportions=self.proportions, device=self.device)
        )

    def _init_timer(self):
        self.timer = Timer()

    def _init_early_stopper(self):
        self.early_stopper = EarlyStopping(patience=50)

    def _init_ema_model(self):
        self.ema_model = get_ema_model(self.use_ema_model, self.model) if self.use_ema_model else None

    def _update_ema(self):
        assert self.ema_model is not None, "ema_model class 'NoneType'"
        update_ema(self.model, self.ema_model, self.ema_decay)

    def _get_device_name(self):
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            self.device_name = torch.cuda.get_device_name(current_device)

    def _load_data(self):
        self.train_fold_ds, self.val_fold_ds = self.ds_list[self.fold][0], self.ds_list[self.fold][1]
        self.train_dl, self.val_dl = (DataLoader(self.train_fold_ds, self.batch, True, num_workers=4, pin_memory=True, drop_last=False),
                            DataLoader(self.val_fold_ds, self.batch, False, num_workers=4, pin_memory=True, drop_last=False))

    def _load_checkpoints(self):
        ckpt = torch.load(self.checkpoint_pth, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.opt.load_state_dict(ckpt['optimizer'])
        self.step = ckpt['step']
        self.Ts = ckpt['Ts']

        if ckpt['ema'] and self.use_ema_model:
            self.ema_model.load_state_dict(ckpt['ema'])
        if ckpt['lr_scheduler'] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    def _save_checkpoint(self,):
        torch.save({
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'step': self.step,
            'Ts': self.Ts
        }, self.save_ckpt_pth)

    def _val_classifier_loop(
            self, *args
    ):
        legend = []
        for item in args:
            legend.append(item)
        valid_metrics = Accumulator(legend)

        for batch, cond in self.val_dl:
            move2device(self.ema_model, batch, cond, device=self.device)
            logits = self.ema_model(batch)

            loss = self.criterion(logits, cond) if "val_loss" in legend else None
            accuracy = compute_accuracy(logits, cond, self.device, self.num_classes) if "accuracy" in legend else None
            precision = compute_precision(logits, cond, self.device, self.num_classes) if "precision" in legend else None
            recall = compute_recall(logits, cond, self.device, self.num_classes) if "recall" in legend else None
            f1 = compute_f1(logits, cond, self.device, self.num_classes) if "f1" in legend else None
            auc = compute_auc_roc(logits, cond, self.device, self.num_classes) if "auc" in legend else None
            confusion_matrix = compute_confusion_matrix(logits, cond, self.device, self.num_classes) if "confusion_matrix" in legend else None
            valid_metrics.update(
                loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1, confusion_matrix=confusion_matrix, auc=auc
            )
        return valid_metrics.mean_return()

    def _val_guide_loop(self, *args):
        legend = []
        for item in args:
            legend.append(item)
        valid_metrics = Accumulator(legend)

        for batch, cond in self.val_dl:
            move2device(self.ema_model, batch, cond, device=self.device)

            t_idx = torch.randint(0, self.diffusion_dict["num_timesteps"], size=[batch.shape[0], ], device=batch.device)
            xt, _ = self.diffusion.q_sample(batch, t_idx)
            logits = self.ema_model(xt, t_idx)

            loss = self.criterion(logits, cond) if "val_loss" in legend else None
            accuracy = compute_accuracy(logits, cond, self.device, self.num_classes) if "accuracy" in legend else None
            precision = compute_precision(logits, cond, self.device,
                                          self.num_classes) if "precision" in legend else None
            recall = compute_recall(logits, cond, self.device, self.num_classes) if "recall" in legend else None
            f1 = compute_f1(logits, cond, self.device, self.num_classes) if "f1" in legend else None
            confusion_matrix = compute_confusion_matrix(logits, cond, self.device,
                                                        self.num_classes) if "confusion_matrix" in legend else None
            valid_metrics.update(
                loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                confusion_matrix=confusion_matrix
            )
        return valid_metrics.mean_return()

    def run_classifier_loop(self):
        self._init_logger()
        self._init_timer()
        self._init_early_stopper()
        self.logger.log_user_info(f'----- {self.timer.start()} Experiment start -----')

        self.logger.log_user_info(f'Loading data ...')
        self._load_data()
        self.logger.log_user_info(f'Successfully load data: len train-ds = {len(self.train_fold_ds)} and len val-ds = {len(self.val_fold_ds)}')

        self.logger.log_user_info('Creating Model ...')
        self._init_model()

        self.logger.log_user_info('Creating optimizer & lr_scheduler & criterion ...')
        self._init_opt()
        self._init_lr_scheduler()
        self._init_criterion()

        if self.use_checkpoint:
            self.logger.log_user_info('Loading pretrained_weight ...')
            self._load_checkpoints()

        self.logger.log_base_info(self.model_name, self.hyperparameters, self.ds_name, self.device_name)

        running_loss = 0.
        self.opt.zero_grad()
        move2device(
            self.model, self.ema_model, device=self.device
        )

        train_iter = iter(self.train_dl)

        while (
                not self.early_stopper.early_stop or
                self.step != self.Ts
        ):
            self.model.train()
            try:
                batch, cond = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dl)
                batch, cond = next(train_iter)

            move2device(
                batch, cond, device=self.device
            )
            logits = self.model(batch)
            self.step += 1

            loss = self.criterion(logits, cond)
            running_loss += loss
            loss /= self.accumulation_steps

            loss.backward()

            if self.step % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                self._update_ema()
                self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if self.step % self.log_interval == 0:
                avg_loss = running_loss / self.step
                self.logger.log(train_step=self.step, train_loss=avg_loss)

            if self.step % self.val_interval == 0:
                _, curr_time = self.timer.stop()
                curr_lr = self.opt.param_groups[0]['lr']
                val_metrics = self._val_classifier_loop("val_loss", "accuracy", "precision", "recall", "f1",
                                             "confusion_matrix", "auc")
                self.logger.log(val_step=self.step, val_metrics=val_metrics, lr=curr_lr, time=curr_time)

                self.early_stopper(val_metrics["val_loss"])

        self._save_checkpoint()
        visualizer = LogVisualizer(self.logger.history, (3.5, 2.5))
        visualizer.plot_loss('train', os.path.join(self.metrics_pth, '_train.png'))
        visualizer.plot_loss('val', os.path.join(self.metrics_pth, '_val.png'))
        visualizer.plot_metrics(os.path.join(self.metrics_pth, '_metrics.png'))

    def run_guide_loop(self):
        self._init_logger()
        self._init_timer()
        self._init_early_stopper()
        self.logger.log_user_info(f'----- {self.timer.start()} Experiment start -----')

        self.logger.log_user_info(f'Loading data ...')
        self._load_data()
        self.logger.log_user_info(f'Successfully load data: len train-ds = {len(self.train_fold_ds)} and len val-ds = {len(self.val_fold_ds)}')

        self.logger.log_user_info('Creating Model ...')
        self._init_model()

        self.logger.log_user_info('Creating optimizer & lr_scheduler & criterion ...')
        self._init_opt()
        self._init_lr_scheduler()
        self._init_criterion()

        self.logger.log_user_info('Creating diffusion scheduler ...')
        self._init_diffusion()

        if self.use_checkpoint:
            self.logger.log_user_info('Loading pretrained_weight ...')
            self._load_checkpoints()
        self.logger.log_base_info(self.model_name, self.hyperparameters, self.ds_name, self.device_name)

        running_loss = 0.
        self.opt.zero_grad()
        move2device(
            self.model, self.ema_model, device=self.device
        )

        train_iter = iter(self.train_dl)

        while (
                not self.early_stopper.early_stop or
                self.step != self.Ts
        ):
            self.model.train()
            try:
                batch, cond = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dl)
                batch, cond = next(train_iter)

            move2device(
                batch, cond, device=self.device
            )
            t_idx = torch.randint(0, self.diffusion_dict["num_timesteps"], size=[batch.shape[0], ], device=batch.device)
            xt, _ = self.diffusion.q_sample(batch, t_idx)
            logits = self.model(xt, t_idx)
            self.step += 1

            loss = self.criterion(logits, cond)
            running_loss += loss
            loss /= self.accumulation_steps

            loss.backward()

            if self.step % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                self._update_ema()
                self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if self.step % self.log_interval == 0:
                avg_loss = running_loss / self.step
                self.logger.log(train_step=self.step, train_loss=avg_loss)

            if self.step % self.val_interval == 0:
                _, curr_time = self.timer.stop()
                curr_lr = self.opt.param_groups[0]['lr']
                val_metrics = self._val_guide_loop("val_loss", "accuracy", "precision", "recall", "f1",
                                             "confusion_matrix")
                self.logger.log(val_step=self.step, val_metrics=val_metrics, lr=curr_lr, time=curr_time)

                self.early_stopper(val_metrics["val_loss"])

        self._save_checkpoint()
        visualizer = LogVisualizer(self.logger.history, (3.5, 2.5))
        visualizer.plot_loss('train', os.path.join(self.metrics_pth, '_train.png'))
        visualizer.plot_loss('val', os.path.join(self.metrics_pth, '_val.png'))
        visualizer.plot_metrics(os.path.join(self.metrics_pth, '_metrics.png'))

class TrainLogitsKsLoop:
    def __init__(self,
                 Ks,
                 ds_name,
                 tf_type,
                 data_dir,
                 mapping_file_pth,
                 **kwargs):
        self.Ks = Ks
        self.ds_name = ds_name
        self.ds = get_dataset(ds_name, image_path=data_dir, annotation_path=mapping_file_pth)
        self.Ts = kwargs.get('Ts', 10_000)
        self.warmup_T = kwargs.get('warmup_T', 1_000)
        self.root_dir = kwargs.get('root_dir')
        self.save_abs_dir = kwargs.get('save_abst_dir')
        self.log_abs_dir = kwargs.get('log_abs_dir')
        self.metrics_abs_dir = kwargs.get('metrics_abs_dir')
        self.num_classes = kwargs.get('num_classes', 3)
        self.batch = kwargs.get('batch', 64)
        self.accumulation_steps = kwargs.get('accumulation_steps', 2)
        self.diffusion_name = kwargs.get('diffusion_name', 'linear')
        self.diffusion_dict = kwargs.get('diffusion_dict')
        self.model_name = kwargs.get('model_name')
        self.model_dict = kwargs.get('model_dict')
        self.opt_name = kwargs.get('opt_name')
        self.opt_dict = kwargs.get('opt_dict')
        self.lr_scheduler_name = kwargs.get('lr_scheduler_name', 'warmup_cosineannealing')
        self.lr_dict = kwargs.get('lr_dict')
        self.criterion_name = kwargs.get('criterion_name', 'cross_entropy')
        self.criterion_dict = kwargs.get('criterion_dict')
        self.ema_decay = kwargs.get('ema_decay', 0.9999)
        self.log_interval = kwargs.get('log_interval', 6)
        self.val_interval = kwargs.get('val_interval', 10)
        self.device = kwargs.get('device', 'cuda')
        self.checkpoint_pth = kwargs.get('checkpoint_pth')
        self.use_diffusion = kwargs.get('use_diffusion')
        self.use_checkpoint = kwargs.get('use_checkpoint')
        self.use_timer = kwargs.get('use_timer')
        self.use_earlystopper = kwargs.get('use_earlystopper')
        self.use_logger = kwargs.get('use_logger')
        self.use_ema_model = kwargs.get('use_ema_model')

        train_tf, val_tf = get_pipeline(tf_type)
        self.ds_list = get_fold_data(Ks, train_tf, val_tf, self.ds)

    def execute(self, name):
        for fold in range(self.Ks):
            train_fold_loop = TrainLogitsFoldLoop(
                self.Ts,
                self.warmup_T,
                fold,
                self.root_dir,
                self.save_abs_dir,
                self.log_abs_dir,
                self.metrics_abs_dir,
                self.ds_name,
                self.ds,
                self.ds_list,
                self.num_classes,
                self.batch,
                self.accumulation_steps,
                self.diffusion_name,
                self.diffusion_dict,
                self.model_name,
                self.model_dict,
                self.opt_name,
                self.opt_dict,
                self.lr_scheduler_name,
                self.lr_dict,
                self.criterion_name,
                self.criterion_dict,
                self.ema_decay,
                self.log_interval,
                self.val_interval,
                self.device,
                self.checkpoint_pth,
                self.use_diffusion,
                self.use_checkpoint,
                self.use_timer,
                self.use_earlystopper,
                self.use_logger,
                self.use_ema_model
            )
            if name == 'classifier':
                train_fold_loop.run_classifier_loop()
            else:
                train_fold_loop.run_guide_loop()

class TrainCRCFoldLoop:
    def __init__(self,
                 Ts,
                 warmup_T,
                 fold,
                 root_dir,
                 save_abs_dir,
                 log_abs_dir,
                 metrics_abs_dir,
                 ds_name,
                 ds,
                 ds_list,
                 num_classes,
                 batch,
                 accumulation_steps,
                 diffusion_name,
                 diffusion_dict,
                 model_name_dict,
                 model_dict,
                 opt_name,
                 opt_dict,
                 lr_scheduler_name,
                 lr_dict,
                 criterion_name,
                 criterion_dict,
                 ema_decay,
                 log_interval,
                 val_interval,
                 device,
                 checkpoint_pth_dict,
                 use_diffusion=False,
                 use_checkpoint=False,
                 use_timer=True,
                 use_earlystopper=True,
                 use_logger=True,
                 use_ema_model=True,
                 **kwargs
                 ):
        self.step = 0
        self.Ts = Ts
        self.warmup_T = warmup_T
        self.fold = fold
        self.root_dir = root_dir
        self.log_root_dir = os.path.join(root_dir, log_abs_dir)
        self.save_ckpt_pth = str(os.path.join(root_dir, save_abs_dir, f"fold_{fold+1}.pt"))
        self.metrics_pth = str(os.path.join(root_dir, metrics_abs_dir, f"fold_{fold+1}"))
        self.ds_name = ds_name
        self.ds = ds
        self.ds_list = ds_list
        self.proportions = self.ds.get_proportions(num_classes)
        self.num_classes = num_classes
        self.batch = batch
        self.accumulation_steps = accumulation_steps
        self.diffusion_name = diffusion_name
        self.diffusion_dict = diffusion_dict
        self.model_name_dict = model_name_dict
        self.model_dict = model_dict
        self.opt_name = opt_name
        self.opt_dict = opt_dict
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_dict = lr_dict
        self.criterion_name = criterion_name
        self.criterion_dict = criterion_dict
        self.ema_decay = ema_decay
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.device = device
        self.checkpoint_pth_dict = checkpoint_pth_dict
        self.use_diffusion = use_diffusion
        self.use_checkpoint = use_checkpoint
        self.use_timer = use_timer
        self.use_earlystopper = use_earlystopper
        self.use_logger = use_logger
        self.use_ema_model = use_ema_model
        self.lr = lr_dict["lr"]

        self.hyperparameters = warp_hyperparameters(
            diffusion_dict, True,
            model_name_dict, opt_dict, lr_dict,
            proportions=self.proportions, num_classes=self.num_classes, batch=self.batch, accumulation_steps=self.accumulation_steps,
            ema_decay=self.ema_decay, log_interval=self.log_interval, val_interval=self.val_interval
        )

        make_dirs(self.metrics_pth)
    def _init_logger(self):
        assert self.use_logger, f"Strongly suggest using logger"
        self.logger = Logger(self.log_root_dir, f'fold_{self.fold + 1}.log')

    def _init_diffusion(self):
        if self.use_diffusion:
            self.diffusion = get_noise_scheduler(
                self.diffusion_name, **self.diffusion_dict
            )

    def _init_model(self):
        self.models = {}
        for key, value in self.model_name_dict.items():
            self.models[key] = get_model(value, **self.model_dict)

    def _init_opt(self):
        self.opt = get_optimizer(
            self.opt_name, self.models['crcnet'].parameters(), **self.opt_dict.update(lr=self.lr)
        )

    def _init_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(
            self.lr_scheduler_name, self.opt, **self.lr_dict.update(
                warmup_T=self.warmup_T, Ts=self.Ts
            )
        )

    def _init_criterion(self):
        self.criterion = get_criterion(
            self.criterion_name, **self.criterion_dict.update(proportions=self.proportions, device=self.device)
        )

    def _init_timer(self):
        self.timer = Timer()

    def _init_early_stopper(self):
        self.early_stopper = EarlyStopping(patience=50)

    def _init_ema_model(self):
        self.ema_model = get_ema_model(self.use_ema_model, self.models["crcnet"]) if self.use_ema_model else None

    def _update_ema(self):
        assert self.ema_model is not None, "ema_model class 'NoneType'"
        update_ema(self.models["crcnet"], self.ema_model, self.ema_decay)

    def _get_device_name(self):
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            self.device_name = torch.cuda.get_device_name(current_device)

    def _load_data(self):
        self.train_fold_ds, self.val_fold_ds = self.ds_list[self.fold][0], self.ds_list[self.fold][1]
        self.train_dl, self.val_dl = (DataLoader(self.train_fold_ds, self.batch, True, num_workers=4, pin_memory=True, drop_last=False),
                            DataLoader(self.val_fold_ds, self.batch, False, num_workers=4, pin_memory=True, drop_last=False))

    def _load_checkpoints(self):
        classifier_ckpt_pth = self.checkpoint_pth_dict['classifier']
        guide_ckpt_pth = self.checkpoint_pth_dict['guide']
        crc_ckpt_pth = self.checkpoint_pth_dict['crcnet']

        classifier_ckpt = torch.load(classifier_ckpt_pth, map_location=self.device)
        guide_ckpt = torch.load(guide_ckpt_pth, map_location=self.device)
        crc_ckpt = torch.load(crc_ckpt_pth, map_location=self.device)

        self.models['classifier'].load_state_dict(classifier_ckpt['model'])
        self.models['guide'].load_state_dict(guide_ckpt['model'])

        self.models['crcnet'].load_state_dict(crc_ckpt['model'])
        self.opt.load_state_dict(crc_ckpt['optimizer'])
        self.step = crc_ckpt['step']
        self.Ts = crc_ckpt['Ts']
        if crc_ckpt['ema'] and self.use_ema_model:
            self.ema_model.load_state_dict(crc_ckpt['ema'])
        if crc_ckpt['lr_scheduler'] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(crc_ckpt['lr_scheduler'])

    def _save_checkpoint(self,):
        torch.save({
            'model': self.models['crcnet'].state_dict(),
            'ema': self.ema_model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'step': self.step,
            'Ts': self.Ts
        }, self.save_ckpt_pth)

    def _obtain_semantic_feats(self, batch, t_idx=None, target_layer=None):
        with torch.no_grad():
            if t_idx is None:
                act_extractor = ActivationExtractor(self.models['classifier'], [target_layer])
                _ = self.models['classifier'](batch)
                out = act_extractor.get(target_layer)
                act_extractor.remove()
                return out
            else:
                act_extractor = ActivationExtractor(self.models['guide'], [target_layer])
                _ = self.models['guide'](batch, t_idx)
                out = act_extractor.get(target_layer)
                act_extractor.remove()
                return out

    def _val_loop(self, *args):
        legend = []
        for item in args:
            legend.append(item)
        valid_metrics = Accumulator(legend)

        for batch, cond in self.val_dl:
            move2device(self.ema_model, batch, cond, device=self.device)

            t_idx = torch.randint(0, self.diffusion_dict["num_timesteps"], size=[batch.shape[0], ], device=batch.device)
            xt, r_noise = self.diffusion.q_sample(batch, t_idx)
            low_semantic = self._obtain_semantic_feats(batch, target_layer='feats.5')
            high_semantic = self._obtain_semantic_feats(xt, t_idx, target_layer='avgpool')
            logits, eps = self.models['crcnet'](xt, t_idx, low_semantic, high_semantic)

            loss = self.criterion(logits, cond, eps, r_noise) if "val_loss" in legend else None
            accuracy = compute_accuracy(logits, cond, self.device, self.num_classes) if "accuracy" in legend else None
            precision = compute_precision(logits, cond, self.device,
                                          self.num_classes) if "precision" in legend else None
            recall = compute_recall(logits, cond, self.device, self.num_classes) if "recall" in legend else None
            f1 = compute_f1(logits, cond, self.device, self.num_classes) if "f1" in legend else None
            confusion_matrix = compute_confusion_matrix(logits, cond, self.device,
                                                        self.num_classes) if "confusion_matrix" in legend else None
            valid_metrics.update(
                loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                confusion_matrix=confusion_matrix
            )
        return valid_metrics.mean_return()

    def run_loop(self):
        self._init_logger()
        self._init_timer()
        self._init_early_stopper()
        self.logger.log_user_info(f'----- {self.timer.start()} Experiment start -----')

        self.logger.log_user_info(f'Loading data ...')
        self._load_data()
        self.logger.log_user_info(
            f'Successfully load data: len train-ds = {len(self.train_fold_ds)} and len val-ds = {len(self.val_fold_ds)}')

        self.logger.log_user_info('Creating Model ...')
        self._init_model()

        self.logger.log_user_info('Creating optimizer & lr_scheduler & criterion ...')
        self._init_opt()
        self._init_lr_scheduler()
        self._init_criterion()

        if self.use_checkpoint:
            self.logger.log_user_info('Loading pretrained_weight ...')
            self._load_checkpoints()

        self.logger.log_base_info(self.model_name_dict['crcnet'], self.hyperparameters, self.ds_name, self.device_name)

        running_loss = 0.
        self.opt.zero_grad()
        move2device(
            self.models['classifier'], self.models['guide'], self.models['crcnet'] , self.ema_model, device=self.device
        )

        train_iter = iter(self.train_dl)

        while (
                not self.early_stopper.early_stop or
                self.step != self.Ts
        ):
            self.models['crcnet'].train()
            try:
                batch, rois, cond = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dl)
                batch, rois, cond = next(train_iter)

            move2device(
                batch, rois, cond, device=self.device
            )

            t_idx = torch.randint(0, self.diffusion_dict["num_timesteps"], size=[batch.shape[0], ], device=batch.device)
            xt, r_noise = self.diffusion.q_sample(batch, t_idx)

            low_semantic = self._obtain_semantic_feats(batch, target_layer='feats.5')
            high_semantic = self._obtain_semantic_feats(xt, t_idx, target_layer='avgpool')

            logits, eps = self.models['crcnet'](xt, t_idx, low_semantic, high_semantic)

            loss = self.criterion(logits, cond, eps, r_noise)
            running_loss += loss
            loss /= self.accumulation_steps

            loss.backward()

            if self.step % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.models['crcnet'].parameters(), max_norm=1.0)
                self.opt.step()
                self._update_ema()
                self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if self.step % self.log_interval == 0:
                avg_loss = running_loss / self.step
                self.logger.log(train_step=self.step, train_loss=avg_loss)

            if self.step % self.val_interval == 0:
                _, curr_time = self.timer.stop()
                curr_lr = self.opt.param_groups[0]['lr']
                val_metrics = self._val_loop("val_loss", "accuracy", "precision", "recall", "f1",
                                                   "confusion_matrix")
                self.logger.log(val_step=self.step, val_metrics=val_metrics, lr=curr_lr, time=curr_time)

                self.early_stopper(val_metrics["val_loss"])

        self._save_checkpoint()
        visualizer = LogVisualizer(self.logger.history, (3.5, 2.5))
        visualizer.plot_loss('train', os.path.join(self.metrics_pth, '_train.png'))
        visualizer.plot_loss('val', os.path.join(self.metrics_pth, '_val.png'))
        visualizer.plot_metrics(os.path.join(self.metrics_pth, '_metrics.png'))

class TrainCRCKsLoop:
    def __init__(self,
                 Ks,
                 ds_name,
                 tf_type,
                 data_dir,
                 mapping_file_pth,
                 **kwargs):
        self.Ks = Ks
        self.ds_name = ds_name
        self.ds = get_dataset(ds_name, image_path=data_dir, annotation_path=mapping_file_pth)
        self.Ts = kwargs.get('Ts', 300_000)
        self.warmup_T = kwargs.get('warmup_T', 30_000)
        self.root_dir = kwargs.get('root_dir')
        self.save_abs_dir = kwargs.get('save_abs_dir')
        self.log_abs_dir = kwargs.get('log_abs_dir')
        self.metrics_abs_dir = kwargs.get('metrics_abs_dir')
        self.num_classes = kwargs.get('num_classes', 3)
        self.batch = kwargs.get('batch', 64)
        self.accumulation_steps = kwargs.get('accumulation_steps', 2)
        self.diffusion_name = kwargs.get('diffusion_name', 'linear')
        self.diffusion_dict = kwargs.get('diffusion_dict')
        self.model_name_dict = kwargs.get('model_name_dict')
        self.model_dict = kwargs.get('model_dict')
        self.opt_name = kwargs.get('opt_name')
        self.opt_dict = kwargs.get('opt_dict')
        self.lr_scheduler_name = kwargs.get('lr_scheduler_name', 'warmup_cosineannealing')
        self.lr_dict = kwargs.get('lr_dict')
        self.criterion_name = kwargs.get('criterion_name', 'cross_entropy')
        self.criterion_dict = kwargs.get('criterion_dict')
        self.ema_decay = kwargs.get('ema_decay', 0.9999)
        self.log_interval = kwargs.get('log_interval', 6)
        self.val_interval = kwargs.get('val_interval', 10)
        self.device = kwargs.get('device', 'cuda')
        self.checkpoint_pth_dict = kwargs.get('checkpoint_pth_dict')
        self.use_diffusion = kwargs.get('use_diffusion')
        self.use_checkpoint = kwargs.get('use_checkpoint')
        self.use_timer = kwargs.get('use_timer')
        self.use_earlystopper = kwargs.get('use_earlystopper')
        self.use_logger = kwargs.get('use_logger')
        self.use_ema_model = kwargs.get('use_ema_model')

        train_tf, val_tf = get_pipeline(tf_type)
        self.ds_list = get_fold_data(Ks, train_tf, val_tf, self.ds)

    def execute(self):
        for fold in range(self.Ks):
            TrainCRCFoldLoop(
                self.Ts,
                self.warmup_T,
                fold,
                self.root_dir,
                self.save_abs_dir,
                self.log_abs_dir,
                self.metrics_abs_dir,
                self.ds_name,
                self.ds,
                self.ds_list,
                self.num_classes,
                self.batch,
                self.accumulation_steps,
                self.diffusion_name,
                self.diffusion_dict,
                self.model_name_dict,
                self.model_dict,
                self.opt_name,
                self.opt_dict,
                self.lr_scheduler_name,
                self.lr_dict,
                self.criterion_name,
                self.criterion_dict,
                self.ema_decay,
                self.log_interval,
                self.val_interval,
                self.device,
                self.checkpoint_pth_dict,
                self.use_diffusion,
                self.use_checkpoint,
                self.use_timer,
                self.use_earlystopper,
                self.use_logger,
                self.use_ema_model
            ).run_loop()

class ActivationExtractor:
    """
    from Model get activation
    support layer_names, module, layer_indices to register hook
    """

    def __init__(self, model: torch.nn.Module,
                 layer_names: Optional[Iterable[str]] = None,
                 modules: Optional[Iterable[torch.nn.Module]] = None,
                 layer_indices: Optional[Iterable[int]] = None,
                 detach: bool = True,
                 to_cpu: bool = True):
        self.model = model
        self.detach = detach
        self.to_cpu = to_cpu
        self.activations: Dict[str, Any] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register(layer_names, modules, layer_indices)

    def _register(self, layer_names, modules, layer_indices):
        named = list(self.model.named_modules())
        name2mod = dict(named)

        # helper hook
        def make_hook(name):
            def hook(module, input, output):
                out = output
                if self.detach:
                    out = out.detach()
                if self.to_cpu:
                    out = out.cpu()

                self.activations[name] = out
            return hook

        # register by names
        if layer_names:
            for name in layer_names:
                if name not in name2mod:
                    raise KeyError(f"Can not find: {name}. Can be used module e.g.: {[n for n,_ in named][:5]}...")
                handle = name2mod[name].register_forward_hook(make_hook(name))
                self.handles.append(handle)

        # register by modules
        if modules:
            for i, m in enumerate(modules):
                if not isinstance(m, torch.nn.Module):
                    raise TypeError("modules should be Iterable(torch.nn.Module)")
                # try to find a name for readability
                found_name = None
                for n, mm in named:
                    if mm is m:
                        found_name = n
                        break
                name = found_name or f"module_{i}"
                handle = m.register_forward_hook(make_hook(name))
                self.handles.append(handle)

        # register by indices
        if layer_indices:
            for idx in layer_indices:
                if idx < 0 or idx >= len(named):
                    raise IndexError(f"layer index {idx} out of bound，bound: 0..{len(named)-1}")
                name, module = named[idx]
                handle = module.register_forward_hook(make_hook(name or f"idx_{idx}"))
                self.handles.append(handle)

    def get(self, name: str):
        """return activation"""
        if name not in self.activations:
            raise KeyError(f"no activation saved in name {name}. current keys: {list(self.activations.keys())}")
        return self.activations[name]

    def clear(self):
        """clear activation but not remove hooks"""
        self.activations.clear()

    def remove(self):
        """remove hooks"""
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()

def move2device(*args, device):
    for item in args:
        assert hasattr(item, "to"), f"item {item} has no attr of .to()"
        item.to(device)

def make_dirs(*args):
    for item in args:
        if os.path.isdir(item):
            os.makedirs(item, exist_ok=True)

def warp_hyperparameters(base, override=True, *dicts, **kwargs):
    base = {} if base is None else dict(base)
    out = dict(base)

    for item in dicts:
        out.update(item)

    for k, v in kwargs.items():
        if k in out:
            if override:
                out[k] = v
        else:
            out[k] = v

    return out



if __name__ == "__main__":
    # from Models.MobileNetV1 import *
    # from Models.ResNet50FiLM import *
    # model = ResNet50FiLM(3, 256)
    # x = torch.randn(size=[2, 1, 224, 224], dtype=torch.float32)
    # t_idx = torch.randint(0, 1_000, size=[x.shape[0], ], device=x.device)
    # act_extractor = ActivationExtractor(model, layer_names=['avgpool'])
    # with torch.no_grad():
    #     out = model(x, t_idx)
    # act = act_extractor.get('avgpool')
    # print(act)
    # print(act.shape)
    pass




