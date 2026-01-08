from torch.utils.data import DataLoader
from torchmetrics import AUROC
from Models.logger import Logger, LogVisualizer
from Models.script_utils import logits_default, opt_dict_default, lr_dict_default, criterion_dict_default, \
    diffusion_default
from Models.train_utils import move2device, warp_hyperparameters
from Utils.evaluator import compute_accuracy, compute_precision, compute_recall, compute_f1, compute_confusion_matrix
from Utils.tools_setting import *

"""
hyperparameters
"""
K = 5
batch_size = 64
warmup_T=1_000
Ts = 10_000
num_classes = 3
accumulation_steps = 2
log_interval = 6
val_interval = 10
patience = 100
num_timesteps=1_000
model_name = 'fpn'
ds_name = 'raw'
data_dir = r'data/MRI/Images'
mapping_file_pth = r'data/MRI/fname2label.csv'
tf_type = 'strong'
root_dir = fr'Result/resnet50-compare'
log_root_dir = fr'Result/resnet50-compare/LOGs'
ckpt_root_dir = fr'Result/resnet50-compare/Parameters'
metric_root_dir = fr'Result/resnet50-compare/Metrics'
device = 'cuda'


#========== load data ==========#
ds = get_dataset(ds_name,
                 image_path=data_dir,
                 annotation_path=mapping_file_pth)
train_tf, val_tf = get_pipeline(tf_type)
ds_list = get_fold_data(K, train_tf, val_tf, ds)

proportion = ds.get_proportions(num_classes)

lr_dict = lr_dict_default()
lr_dict.update(warmup_T=warmup_T,
               Ts=Ts)

opt_dict = opt_dict_default()
opt_dict.update(lr=lr_dict['lr'])

criterion_dict = criterion_dict_default()
criterion_dict.update(proportions=proportion, device=device)

model_dict = logits_default(model_name)

diffusion_dict = diffusion_default()
for fold in range(K):
    os.makedirs(log_root_dir, exist_ok=True)
    os.makedirs(ckpt_root_dir, exist_ok=True)

    os.makedirs(os.path.join(metric_root_dir, f'fold_{fold + 1}'), exist_ok=True)
    metric_pth = os.path.join(metric_root_dir, f'fold_{fold + 1}')
    #========== init ==========#
    # logger

    diffusion = get_noise_scheduler(name='linear',
                                    num_timesteps=num_timesteps)

    step = 0

    hyperparameters = warp_hyperparameters(
        diffusion_dict, True,
        opt_dict, lr_dict,
        proportions=proportion, num_classes=num_classes, batch=batch_size,
        accumulation_steps=accumulation_steps,
        log_interval=log_interval, val_interval=val_interval
    )


    def _save_checkpoint():
        torch.save({
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'step': step,
            'Ts': Ts
        }, str(os.path.join(ckpt_root_dir, f"fold_{fold + 1}.pt")))

    def _val_classifier_loop(
            *args
    ):
        legend = list(args)
        valid_metrics = Accumulator(legend)

        model.eval()

        with torch.no_grad():
            if "auc" in legend:
                auroc_metric = AUROC(task='multiclass', num_classes=num_classes, average='macro').to(device)
                auroc_metric.reset()
            else:
                auroc_metric = None

            for batch, cond in val_dl:
                batch, cond = move2device(device, batch, cond)

                # logits = self.ema_model(batch)
                logits = model(batch)

                if "val_loss" in legend:
                    loss = criterion(logits, cond)
                    loss_val = float(loss.detach().cpu().item())
                else:
                    loss_val = None

                if "accuracy" in legend:
                    accuracy = compute_accuracy(logits, cond, device, num_classes)
                    accuracy_val = float(accuracy.detach().cpu().item()) if isinstance(accuracy,
                                                                                       torch.Tensor) else float(
                        accuracy)
                else:
                    accuracy_val = None

                if "precision" in legend:
                    precision = compute_precision(logits, cond, device, num_classes)
                    precision_val = float(precision.detach().cpu().item()) if isinstance(precision,
                                                                                         torch.Tensor) else float(
                        precision)
                else:
                    precision_val = None

                if "recall" in legend:
                    recall = compute_recall(logits, cond, device, num_classes)
                    recall_val = float(recall.detach().cpu().item()) if isinstance(recall, torch.Tensor) else float(
                        recall)
                else:
                    recall_val = None

                if "f1" in legend:
                    f1 = compute_f1(logits, cond, device, num_classes)
                    f1_val = float(f1.detach().cpu().item()) if isinstance(f1, torch.Tensor) else float(f1)
                else:
                    f1_val = None

                if "confusion_matrix" in legend:
                    cm = compute_confusion_matrix(logits, cond, device, num_classes)
                    if isinstance(cm, torch.Tensor):
                        cm = cm.detach().cpu().numpy()
                    cm_val = cm
                else:
                    cm_val = None

                valid_metrics.update(
                    val_loss=loss_val,
                    accuracy=accuracy_val,
                    precision=precision_val,
                    recall=recall_val,
                    f1=f1_val,
                    confusion_matrix=cm_val,
                    auc=None
                )

                if auroc_metric is not None:
                    probs = F.softmax(logits, dim=1)
                    auroc_metric.update(probs, cond.to(device))

            if auroc_metric is not None:
                try:
                    auc = auroc_metric.compute()
                    auc_val = float(auc.detach().cpu().item())
                except Exception as e:
                    print("AUROC compute failed:", e)
                    auc_val = float('nan')
                valid_metrics.update(auc=auc_val)

        return valid_metrics.mean_return()

    logger = Logger(log_root_dir, f'fold_{fold + 1}.log')
    timer = Timer()
    early_stopper = EarlyStopping(patience=patience)
    logger.log_user_info(f'----- {timer.start()} Experiment start -----')

    logger.log_user_info(f'Loading data ...')
    train_fold_ds, val_fold_ds = ds_list[fold][0], ds_list[fold][1]
    train_dl, val_dl = (
        DataLoader(train_fold_ds, batch_size, True, num_workers=4, pin_memory=True, drop_last=False),
        DataLoader(val_fold_ds, batch_size, False, num_workers=4, pin_memory=True, drop_last=False))
    logger.log_user_info(
        f'Successfully load data: len train-ds = {len(train_fold_ds)} and len val-ds = {len(val_fold_ds)}')

    logger.log_user_info('Creating Model ...')
    model = get_model(model_name, **model_dict)

    logger.log_user_info('Creating optimizer & lr_scheduler & criterion ...')
    opt = get_optimizer('adamw', model.parameters(), **opt_dict)
    lr_scheduler = get_lr_scheduler(
        'warmup_cosineannealing', opt, **lr_dict
    )
    criterion = get_criterion(
        'cross_entropy', **criterion_dict
    )

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

    logger.log_base_info(model_name, hyperparameters, ds_name, device_name)

    running_loss = 0.
    opt.zero_grad()
    model.to(device)

    train_iter = iter(train_dl)

    while (
            not early_stopper.early_stop and
            step <= Ts
    ):
        model.train()
        try:
            batch, cond = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch, cond = next(train_iter)

        batch, cond = move2device(
            device, batch, cond
        )
        logits = model(batch)
        step += 1

        loss = criterion(logits, cond)
        running_loss += loss
        loss /= accumulation_steps

        loss.backward()

        if step % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            opt.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        if step % log_interval == 0:
            avg_loss = running_loss / step
            logger.log(train_step=step, train_loss=avg_loss)

        if step % val_interval == 0:
            _, curr_time = timer.stop()
            curr_lr = opt.param_groups[0]['lr']
            val_metrics = _val_classifier_loop("val_loss", "accuracy", "precision", "recall", "f1",
                                                    "confusion_matrix", "auc")
            logger.log(val_step=step, val_metrics=val_metrics, lr=curr_lr, time=curr_time)

            early_stopper(val_metrics["val_loss"])

    _save_checkpoint()
    visualizer = LogVisualizer(logger.history, (3.5, 2.5))
    visualizer.plot_loss('train', os.path.join(metric_pth, '_train.png'))
    visualizer.plot_loss('val', os.path.join(metric_pth, '_val.png'))
    visualizer.plot_metrics(os.path.join(metric_pth, '_metrics.png'))
    logger.close_logger()

    if 'train_iter' in locals():
        del train_iter

    if 'train_dl' in locals():
        del train_dl
    if 'val_dl' in locals():
        del val_dl


    del model, opt, criterion

    gc.collect()
    torch.cuda.empty_cache()





