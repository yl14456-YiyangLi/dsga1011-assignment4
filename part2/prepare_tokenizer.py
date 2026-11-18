"""Add ordinary SQL operator tokens; never register them as special tokens."""
from pathlib import Path
from transformers import AutoTokenizer, AddedToken

tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')
added = tokenizer.add_tokens([
    AddedToken(op, normalized=False, special=False) for op in ['<', '<=', '<>']
])
assert len(tokenizer) == 32103
for op in ['<', '<=', '<>']:
    text = f'SELECT flight_id FROM flight WHERE departure_time {op} 1200'
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    assert tokenizer.unk_token_id not in ids
    assert op in tokenizer.decode(ids, skip_special_tokens=True)
destination = Path(__file__).resolve().parent / 'tokenizer_sql'
tokenizer.save_pretrained(destination)
print('Added regular tokens:', added, 'Saved:', destination)
