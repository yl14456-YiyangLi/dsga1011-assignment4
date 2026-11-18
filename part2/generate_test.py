"""Generate final test SQL/record files using a selected local checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import T5ForConditionalGeneration
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--extra_credit', action='store_true')
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()
assert torch.cuda.is_available(), 'Use a GPU for generation'

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / ('scratch' if args.extra_credit else 'finetuned')))
import load_data
import utils
from train_t5 import save_predictions

utils.DB_PATH = str(root / 'data/flight_database.db')
load_data.TOKENIZER_PATH = str(args.checkpoint.resolve())
load_data.get_tokenizer.cache_clear()
tokenizer = load_data.get_tokenizer()
loader = load_data.get_dataloader(args.batch_size, 'test')
assert len(loader.dataset) == 432
model = T5ForConditionalGeneration.from_pretrained(
    args.checkpoint, local_files_only=True
).to('cuda')
assert model.config.vocab_size == len(tokenizer)
model.eval()
queries = []
with torch.inference_mode():
    for enc, mask, initial in tqdm(loader, desc='Test'):
        generated = model.generate(
            input_ids=enc.to('cuda'), attention_mask=mask.to('cuda'),
            decoder_input_ids=initial.to('cuda'), max_new_tokens=512,
            num_beams=1, do_sample=False, use_cache=True,
        )
        queries.extend(q.strip() for q in tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))
assert len(queries) == 432
args.output_dir.mkdir(parents=True, exist_ok=True)
stem = 't5_ft_experiment_ec_test' if args.extra_credit else 't5_ft_experiment_test'
save_predictions(queries, args.output_dir / (stem + '.sql'), args.output_dir / (stem + '.pkl'))
(args.output_dir / 'generation_config_used.json').write_text(json.dumps({
    'checkpoint': str(args.checkpoint.resolve()), 'number_of_predictions': 432,
    'num_beams': 1, 'do_sample': False, 'max_new_tokens': 512,
    'query_timeout_seconds': 10, 'execution_workers': 4,
}, indent=2))
print('Saved:', args.output_dir / (stem + '.sql'))
print('Saved:', args.output_dir / (stem + '.pkl'))
