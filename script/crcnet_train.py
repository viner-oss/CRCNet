import argparse
import time
from Models.script_utils import (
    logits_default,
    add_dict_to_argparser,
    args_to_dict,
    diffusion_default,
    opt_dict_default,
    lr_dict_default,
    criterion_dict_default, crcnet_default
)
from Models.train_utils import TrainCRCKsLoop


def main():
    local_time = time.localtime()
    formatted_time = time.strftime("%Y%m%d_%H%M", local_time)
    parser, config = create_argparser(formatted_time)
    Train = TrainCRCKsLoop(
        **(args_to_dict(parser.parse_args(), config.keys()))
    )
    Train.execute()

def create_argparser(current_time):
    defaults = dict(
        Ks=10,
        ds_name='mri',
        tf_type='strong',
        data_dir=r'data/Images',
        mapping_file_pth=r'data/fname2label.csv',
        Ts=15_000,
        warmup_T=1_000,
        root_dir=fr'Result/crcnet/Exp{current_time}',  # Result/mobilenet_v1(resnet50)/Experiment20250824-1024
        save_abs_dir=r'Parameters',
        log_abs_dir=r'LOGs',
        metrics_abs_dir=r'Metrics',
        num_classes=3,
        batch=8,
        accumulation_steps=8,
        diffusion_name='linear',
        opt_name='adamw',
        lr_scheduler_name='warmup_cosineannealing',
        criterion_name='jointloss',
        ema_decay=0.9999,
        log_interval=10,
        val_interval=100,
        device='cuda',
        checkpoint_pth_dict=None,
        use_diffusion=True,
        use_checkpoint=True,
        use_timer=True,
        use_earlystopper=True,
        use_logger=True,
        use_ema_model=True
    )
    ckpt_dict = dict(
        classifier=r'Result/mobilenet_v1/Exp20251001_2129/Parameters/fold_8.pt',
        guide=r'Result/resnet50film/Exp20251003_1342/Parameters/fold_8.pt',
        crcnet=None
    )
    
    diffusion_config = diffusion_default()

    model_name_config = dict(
        classifier='mobilenet_v1',
        guide='resnet50film',
        crcnet='crcnet'
    )
    model_config = crcnet_default()
    model_config.update(logits_default('mobilenet_v1'))
    model_config.update(logits_default('resnet50film'))
    opt_config = opt_dict_default()
    lr_config = lr_dict_default()
    criterion_config = criterion_dict_default()
    defaults.update(
        diffusion_dict=diffusion_config, model_dict=model_config, model_name_dict=model_name_config,
        opt_dict=opt_config, lr_dict=lr_config, criterion_dict=criterion_config, checkpoint_pth_dict=ckpt_dict
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser, defaults

if __name__ == "__main__":
    main()
