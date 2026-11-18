
import os
from pathlib import Path
from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

PAD_IDX = 0
TOKENIZER_PATH = os.environ.get(
    "T5_TOKENIZER_PATH", str(Path(__file__).resolve().parents[1] / "tokenizer_sql")
)
DATA_FOLDER = Path(os.environ.get(
    "T5_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")
))

@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def load_lines(path):
    return Path(path).read_text().splitlines()

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split
        self.process_data(data_folder, split, get_tokenizer())

    def process_data(self, data_folder, split, tokenizer):
        folder = Path(data_folder)
        self.texts = [s.strip() for s in load_lines(folder / f"{split}.nl")]
        assert all(self.texts), "发现空的自然语言输入"

        self.encoder_ids = tokenizer(
            self.texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            verbose=False,
        )["input_ids"]

        self.target_ids = None
        if split != "test":
            self.sqls = [
                s.strip() for s in load_lines(folder / f"{split}.sql")
            ]
            assert len(self.texts) == len(self.sqls)
            assert all(self.sqls)

            self.target_ids = tokenizer(
                self.sqls,
                padding=False,
                truncation=False,
                add_special_tokens=True,
                verbose=False,
            )["input_ids"]

            for ids in self.target_ids:
                assert ids[-1] == tokenizer.eos_token_id
                assert tokenizer.unk_token_id not in ids, "SQL 含未知 token"

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        encoder = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        if self.target_ids is None:
            return encoder
        target = torch.tensor(self.target_ids[idx], dtype=torch.long)
        return encoder, target

def normal_collate_fn(batch):
    encoder_sequences, target_sequences = zip(*batch)

    encoder_ids = pad_sequence(
        encoder_sequences, batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # T5 的 decoder 起始 token 是 pad，即 0
    decoder_sequences = [
        torch.cat([torch.tensor([PAD_IDX]), target[:-1]])
        for target in target_sequences
    ]
    decoder_inputs = pad_sequence(
        decoder_sequences, batch_first=True, padding_value=PAD_IDX
    )
    decoder_targets = pad_sequence(
        target_sequences, batch_first=True, padding_value=PAD_IDX
    )
    initial_decoder_inputs = torch.full(
        (len(batch), 1), PAD_IDX, dtype=torch.long
    )

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )

def test_collate_fn(batch):
    encoder_ids = pad_sequence(
        batch, batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full(
        (len(batch), 1), PAD_IDX, dtype=torch.long
    )
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    return DataLoader(
        T5Dataset(DATA_FOLDER, split),
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=test_collate_fn if split == "test" else normal_collate_fn,
        num_workers=0,
    )

def load_t5_data(batch_size, test_batch_size):
    return (
        get_dataloader(batch_size, "train"),
        get_dataloader(test_batch_size, "dev"),
        get_dataloader(test_batch_size, "test"),
    )
