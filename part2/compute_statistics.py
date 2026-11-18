"""Recompute Q4 and extra-credit token statistics without EOS or padding."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, default=root / 'data')
parser.add_argument('--tokenizer', type=Path, default=root / 'tokenizer_sql')
parser.add_argument('--output_dir', type=Path, default=root.parent / 'runs/q4_recomputed')
args = parser.parse_args()

def summarize(tokenizer, texts):
    ids = tokenizer(texts, add_special_tokens=False, padding=False,
                    truncation=False, verbose=False)['input_ids']
    lengths = [len(seq) for seq in ids]
    return {
        'mean_length': float(np.mean(lengths)),
        'vocabulary_size': len({token for seq in ids for token in seq}),
        'max_length': max(lengths),
        'p95_length': int(np.percentile(lengths, 95)),
        'unknown_token_count': sum(seq.count(tokenizer.unk_token_id) for seq in ids),
        'examples_with_unknown': sum(tokenizer.unk_token_id in seq for seq in ids),
    }

stats, rows = {}, []
for stage, tokenizer_path in [('before', 'google-t5/t5-small'), ('after', str(args.tokenizer))]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    stats[stage] = {}
    for split in ['train', 'dev']:
        texts = [x.strip() for x in (args.data_dir / (split + '.nl')).read_text().splitlines()]
        queries = [x.strip() for x in (args.data_dir / (split + '.sql')).read_text().splitlines()]
        assert len(texts) == len(queries)
        nl, sql = summarize(tokenizer, texts), summarize(tokenizer, queries)
        stats[stage][split] = {'number_of_examples': len(texts), 'natural_language': nl, 'sql': sql}
        rows.append({'stage': stage, 'split': split, 'number_of_examples': len(texts),
                     'mean_sentence_length': nl['mean_length'], 'mean_sql_length': sql['mean_length'],
                     'nl_vocabulary_size': nl['vocabulary_size'], 'sql_vocabulary_size': sql['vocabulary_size']})
        if stage == 'after':
            assert sql['unknown_token_count'] == 0
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / 'q4_statistics.json').write_text(json.dumps({
    'length_includes_special_tokens': False, 'truncation': False, 'statistics': stats}, indent=2))
with (args.output_dir / 'q4_tables.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
