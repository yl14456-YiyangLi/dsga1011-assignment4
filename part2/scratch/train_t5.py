
import argparse
import json
import pickle
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import utils
from load_data import get_tokenizer, load_t5_data
from t5_utils import initialize_model, initialize_optimizer_and_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def execute_query(item):
    """Read-only execution, with an actual per-query time limit."""
    index, query = item
    database = Path(utils.DB_PATH).resolve()
    conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    deadline = time.monotonic() + 10.0
    conn.set_progress_handler(
        lambda: int(time.monotonic() > deadline), 10000
    )
    try:
        records = conn.execute(query).fetchall()
        error = ""
    except Exception as exc:
        records = []
        error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    return index, records, error


def save_predictions(queries, sql_path, records_path):
    sql_path = Path(sql_path)
    records_path = Path(records_path)
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve one prediction per example.
    assert all("\n" not in q and "\r" not in q for q in queries)
    sql_path.write_text(
        "\n".join(queries) + "\n", encoding="utf-8"
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        executed = list(tqdm(
            pool.map(execute_query, enumerate(queries)),
            total=len(queries),
            desc="Execute SQL"
        ))

    assert [x[0] for x in executed] == list(range(len(queries)))
    records = [x[1] for x in executed]
    errors = [x[2] for x in executed]
    with records_path.open("wb") as handle:
        pickle.dump((records, errors), handle)


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, total_tokens = 0.0, 0

    progress = tqdm(loader, desc="Train")
    for enc, mask, dec, targets, _ in progress:
        enc, mask, dec, targets = [
            tensor.to(DEVICE) for tensor in (enc, mask, dec, targets)
        ]
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            input_ids=enc,
            attention_mask=mask,
            decoder_input_ids=dec,
            use_cache=False,
        ).logits

        valid = targets.ne(0)
        loss = F.cross_entropy(logits[valid], targets[valid])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        count = valid.sum().item()
        total_loss += loss.item() * count
        total_tokens += count
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_tokens


@torch.inference_mode()
def eval_epoch(model, loader, tokenizer, output_dir, max_new_tokens):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    queries = []

    for enc, mask, dec, targets, initial in tqdm(loader, desc="Dev"):
        enc, mask, dec, targets, initial = [
            tensor.to(DEVICE)
            for tensor in (enc, mask, dec, targets, initial)
        ]

        logits = model(
            input_ids=enc,
            attention_mask=mask,
            decoder_input_ids=dec,
            use_cache=False,
        ).logits

        valid = targets.ne(0)
        loss_sum = F.cross_entropy(
            logits[valid], targets[valid], reduction="sum"
        )
        total_loss += loss_sum.item()
        total_tokens += valid.sum().item()
        del logits, loss_sum

        generated = model.generate(
            input_ids=enc,
            attention_mask=mask,
            decoder_input_ids=initial,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
        )
        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # Remove outer whitespace only; preserve SQL content.
        queries.extend(query.strip() for query in decoded)

    gold_path = Path("data/dev.sql")
    gold_queries = gold_path.read_text().splitlines()
    assert len(queries) == len(loader.dataset) == len(gold_queries), (
        len(queries), len(loader.dataset), len(gold_queries)
    )

    sql_path = output_dir / "dev.sql"
    records_path = output_dir / "dev.pkl"
    save_predictions(queries, sql_path, records_path)

    sql_em, record_em, record_f1, errors = utils.compute_metrics(
        str(gold_path),
        str(sql_path),
        "records/ground_truth_dev.pkl",
        str(records_path),
    )

    return {
        "dev_loss": total_loss / total_tokens,
        "record_f1": float(record_f1),
        "record_em": float(record_em),
        "sql_em": float(sql_em),
        # Includes syntax errors, execution errors and timeouts.
        "execution_error_rate": sum(bool(e) for e in errors) / len(errors),
    }


@torch.inference_mode()
def test_inference(model, loader, tokenizer, output_dir, max_new_tokens):
    """Available for final inference after selecting the configuration."""
    model.eval()
    queries = []

    for enc, mask, initial in tqdm(loader, desc="Test"):
        generated = model.generate(
            input_ids=enc.to(DEVICE),
            attention_mask=mask.to(DEVICE),
            decoder_input_ids=initial.to(DEVICE),
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
        )
        queries.extend(query.strip() for query in tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ))

    assert len(queries) == len(loader.dataset)
    save_predictions(
        queries,
        output_dir / "t5_ft_experiment_test.sql",
        output_dir / "t5_ft_experiment_test.pkl",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--max_n_epochs", type=int, default=20)
    parser.add_argument("--patience_epochs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer_type", default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler_type", default="linear")
    parser.add_argument("--num_warmup_epochs", type=int, default=1)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU is not available"
    utils.set_random_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert not (output_dir / "history.json").exists(), (
        "This directory already contains a run. Use a new directory."
    )

    config = {
        **vars(args),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "precision": "float32",
        "decoding": "greedy",
        "gradient_clip_norm": 1.0,
        "query_timeout_seconds": 10,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    tokenizer = get_tokenizer()
    train_loader, dev_loader, _ = load_t5_data(
        args.batch_size, args.test_batch_size
    )
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args, model, len(train_loader)
    )

    print("GPU:", config["gpu"], flush=True)
    print("Train examples:", len(train_loader.dataset), flush=True)
    print("Dev examples:", len(dev_loader.dataset), flush=True)
    print("Output directory:", output_dir, flush=True)

    best_f1, stale_epochs = -1.0, 0
    history = []

    for epoch in range(1, args.max_n_epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.max_n_epochs} ==========",
              flush=True)
        started = time.monotonic()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)

        epoch_dir = output_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(exist_ok=True)
        metrics = eval_epoch(
            model, dev_loader, tokenizer, epoch_dir, args.max_new_tokens
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)

        if metrics["record_f1"] > best_f1:
            best_f1 = metrics["record_f1"]
            stale_epochs = 0
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            shutil.copy2(epoch_dir / "dev.sql", output_dir / "best_dev.sql")
            shutil.copy2(epoch_dir / "dev.pkl", output_dir / "best_dev.pkl")
            (output_dir / "best_metrics.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            print(f"Saved best model: Record F1 = {best_f1:.4%}", flush=True)
        else:
            stale_epochs += 1

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        if epoch >= 20 and stale_epochs >= args.patience_epochs:
            print("Early stopping.", flush=True)
            break

    print(f"\nBest dev Record F1: {best_f1:.4%}", flush=True)
    print("Best checkpoint:", output_dir / "best", flush=True)
    print("PASS: T5 training and dev evaluation completed", flush=True)


if __name__ == "__main__":
    main()
