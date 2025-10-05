import os.path
import torch

def diffusion_default():
    return dict(
        timesteps=1_000,
        use_rescale_timesteps=False
    )

def logits_default(
        model_name: str,
        user_config = None,
):
    """
    Default for classifier
    choose specific model for classifier
    """
    base_config = dict(
        init_chs = 1,
        num_classes = 3,
        out_chs = 1,
        num_groups = 32,
        use_bias = False
    )

    model_specific = {
        "resnet50": dict(
            model_name = 'resnet50',
            depth = 4,
            base_chs = 64,
        ),
        "mobilenet_v1": dict(
            model_name = 'mobilenet_v1',
            width_mult = 1.0,
        ),
        "resnet50film": dict(
            num_classes = 3
        )
    }

    config = dict(base_config)
    config.update(model_specific.get(model_name, {}))
    if user_config:
        config.update(user_config)
    return config

def crcnet_default():
    """
    default for CRCnet
    include unet default and detect_head default
    """
    base_config = dict(
        norm = 'gn',
        act = 'relu',
        dropout = 0.0,
    )

    unet = dict(
        image_size = 224,
        init_ch = 1,
        unet_base_chs = 128,
        out_chs = 1,
        unet_ch_mult = (1, 1, 2, 2, 4, 4),
        low_in_chs = 1024,
        high_in_chs = 2048,
        attn_resolution = [],
        num_res_block = 2,
        num_attn_heads = 8,
        num_attn_head_chs = -1,
        num_heads_upsample = -1,
        conv_resample = True,
        use_fp16 = False,
        use_scale_shift_norm = True,
        use_resblock_updown = True,
        use_new_attn_order = False
    )

    detect_head = dict(
        detect_base_chs = 128,
        detect_ch_mult = (1, 2, 4),
        num_classes = 3,
        num_cbr = 5,
        detect_resolution = [112, 28, 7],
        num_detect_head = 3
    )

    config = dict(base_config)
    config.update(unet)
    config.update(detect_head)
    return config

def lr_dict_default():
    return dict(
        lr = 1e-3,
        lr_step = 5_000,
        lr_gamma = 0.1,
        lr_milestones = [3_000, 6_000, 9_000],
        lr_T_max = 10_000,
        warmup_start_lr = 1e-6,
        min_lr = 0.0
    )

def opt_dict_default():
    return dict(
        momentum=0.9,
        weight_decay=5e-4
    )

def criterion_dict_default():
    return dict(
        coef1 = 0.5,
        coef2 = 0.5
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def _load_state_dict(
        model,
        weight_path,
        map_location,
        strict,
        verbose
):
    assert os.path.isfile(weight_path), f"Can not Found weight file {weight_path}"
    ckpt = torch.load(weight_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        new_state[new_k] = v

    load_res = model.load_state_dict(new_state, strict=strict)

    missing = getattr(load_res, 'missing_keys', None)
    unexpected = getattr(load_res, 'unexpected_keys', None)
    if verbose:
        print(f"[load weights] from: {weight_path}")
        if missing is not None or unexpected is not None:
            print(f"  missing_keys: {missing}")
            print(f"  unexpected_keys: {unexpected}")
    return model
