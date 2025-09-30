import argparse
import time
from Models.script_utils import (
    logits_default,
    add_dict_to_argparser,
    args_to_dict,
    diffusion_default,
    opt_dict_default,
    lr_dict_default,
    criterion_dict_default
)
from Models.train_utils import TrainLogitsKsLoop


def main():
    local_time = time.localtime()
    formatted_time = time.strftime("%Y%m%d_%H%M", local_time)
    parser, config = create_argparser(formatted_time)
    Train = TrainLogitsKsLoop(
        **(args_to_dict(parser.parse_args(), config.keys()))
    )
    Train.execute('classifier')

def create_argparser(current_time):
    defaults = dict(
        Ks=10,
        ds_name = 'roi',
        tf_type='strong',
        data_dir = r'data/MRI/Images',
        mapping_file_pth = r'data/MRI/fname2label.csv',
        Ts=10_000,
        warmup_T=1_000,
        root_dir = fr'Result/mobilenet_v1/Exp{current_time}', # Result/mobilenet_v1(resnet50)/Experiment20250824-1024
        save_abs_dir = r'Parameters',
        log_abs_dir = r'LOGs',
        metrics_abs_dir = r'Metrics',
        num_classes = 3,
        batch = 64,
        accumulation_steps = 2,
        diffusion_name = 'linear',
        model_name = 'mobilenet_v1',
        opt_name = 'adamw',
        lr_scheduler_name = 'warmup_cosineannealing',
        criterion_name = 'corss_entropy',
        ema_decay = 0.9999,
        log_interval = 6,
        val_interval = 10,
        device = 'cuda',
        checkpoint_pth = None,
        use_diffusion = False,
        use_checkpoint = False,
        use_timer = True,
        use_earlystopper = True,
        use_logger = True,
        use_ema_model = True
    )
    diffusion_config = diffusion_default()
    model_config = logits_default(defaults["model_name"])
    opt_config = opt_dict_default()
    lr_config = lr_dict_default()
    criterion_config = criterion_dict_default()

    defaults.update(
        diffusion_dict=diffusion_config, model_dict=model_config,
        opt_dict=opt_config, lr_dict=lr_config, criterion_dict=criterion_config
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser, defaults

if __name__ == "__main__":
    main()