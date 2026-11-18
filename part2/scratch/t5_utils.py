import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    pass

def initialize_model(args):
    from load_data import get_tokenizer

    assert not args.finetune, "Extra credit 必须从随机权重开始"
    tokenizer = get_tokenizer()

    config = T5Config.from_pretrained(os.environ.get("T5_CONFIG_PATH", "google-t5/t5-small"))
    config.vocab_size = len(tokenizer)
    assert config.vocab_size == len(tokenizer)

    # 构造函数随机初始化；不调用模型的 from_pretrained。
    config.dropout_rate = args.dropout_rate
    model = T5ForConditionalGeneration(config)
    print("Dropout:", config.dropout_rate, flush=True)
    assert model.config.decoder_start_token_id == tokenizer.pad_token_id

    print("INITIALIZATION: RANDOM WEIGHTS", flush=True)
    print("Vocabulary size:", config.vocab_size, flush=True)
    print("Encoder layers:", config.num_layers, flush=True)
    print("Decoder layers:", config.num_decoder_layers, flush=True)
    return model.to(DEVICE)

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    from pathlib import Path
    from load_data import get_tokenizer

    destination = Path(checkpoint_dir) / ("best" if best else "latest")
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    get_tokenizer().save_pretrained(destination)

def load_model_from_checkpoint(args, best):
    from pathlib import Path

    model_type = "ft" if args.finetune else "scr"
    destination = (
        Path("checkpoints")
        / f"{model_type}_experiments"
        / args.experiment_name
        / ("best" if best else "latest")
    )
    if not (destination / "config.json").exists():
        raise FileNotFoundError(f"Checkpoint not found: {destination}")

    return T5ForConditionalGeneration.from_pretrained(
        destination
    ).to(DEVICE)

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

