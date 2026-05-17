import argparse
import glob
import logging
import os
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import wandb

from src.model import MiniGPT
from src.config import ModelConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TextDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, max_seq_len: int):
        self.token_ids = token_ids
        self.max_seq_len = max_seq_len

    def __len__(self):
        return (len(self.token_ids) - 1) // self.max_seq_len

    def __getitem__(self, idx: int):
        start_idx = idx * self.max_seq_len
        chunk = self.token_ids[start_idx : start_idx + self.max_seq_len + 1]
        if len(chunk) < self.max_seq_len + 1:
            chunk = self.token_ids[-(self.max_seq_len + 1) :]
        return chunk[:-1], chunk[1:]


class Early_Stopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.best_loss = float("inf")
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss * (1 - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm_type", type=str, default="rmsnorm")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--data_path", type=str, default="data/python_code.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run_name", type=str, default="minigpt-run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.001)
    return parser.parse_args()


def get_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


@torch.no_grad()
def estimate_loss(model, dataloader, device, num_batches=20):
    model.eval()
    total_loss = 0.0
    for i, (x, y) in enumerate(dataloader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / min(num_batches, len(dataloader))


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    early_stopping,
    config,
    epoch,
    step,
    batch_idx,
    path,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "early_stopping_best_loss": early_stopping.best_loss,
            "early_stopping_counter": early_stopping.counter,
            "config": config,
            "epoch": epoch,
            "step": step,
            "batch_idx": batch_idx,
            "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        },
        path,
    )
    logging.info(f"Saved checkpoint: {path}")


def find_latest_checkpoint(checkpoint_dir: str, run_name: str):
    pattern = re.compile(
        rf"^{re.escape(run_name)}(?:_epoch(?P<epoch>\d+))?_s(?P<step>\d+)\.pt$"
    )
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, f"{run_name}_*.pt"))
    valid_ckpts = []
    for path in ckpt_files:
        match = pattern.match(os.path.basename(path))
        if match:
            valid_ckpts.append((int(match.group("step")), path))
    if not valid_ckpts:
        return None
    return max(valid_ckpts, key=lambda x: x[0])[1]


def create_train_loader(token_ids: torch.Tensor, args, epoch: int):
    return DataLoader(
        TextDataset(token_ids, args.max_seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed + epoch),
    )


def load_and_tokenize(data_path: str, tokenizer, chunk_size: int = 100000):
    all_ids = []
    buffer = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            buffer.append(line)
            if len(buffer) >= chunk_size:
                chunk_text = "".join(buffer)
                ids = tokenizer.encode(chunk_text, add_special_tokens=False)
                all_ids.extend(ids)
                buffer = []
                logging.info(f"Tokenized {len(all_ids):,} tokens so far...")
        if buffer:
            chunk_text = "".join(buffer)
            ids = tokenizer.encode(chunk_text, add_special_tokens=False)
            all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    logging.info(f"Using device: {device}")

    latest_ckpt = None
    ckpt_metadata = None
    wandb_id = None
    wandb_resume = None
    if args.resume:
        latest_ckpt = find_latest_checkpoint(args.checkpoint_dir, args.run_name)
        if latest_ckpt:
            ckpt_metadata = torch.load(latest_ckpt, map_location="cpu")
            wandb_id = ckpt_metadata.get("wandb_run_id")
            wandb_resume = "must" if wandb_id else "allow"

    wandb_kwargs = {
        "project": "minigpt-training",
        "name": args.run_name,
        "config": vars(args),
    }
    if wandb_resume is not None:
        wandb_kwargs["resume"] = wandb_resume
    if wandb_id is not None:
        wandb_kwargs["id"] = wandb_id
    wandb.init(**wandb_kwargs)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = load_and_tokenize(args.data_path, tokenizer)
    split_idx = int(len(token_ids) * (1 - args.val_split))

    train_loader = DataLoader(
        TextDataset(token_ids[:split_idx], args.max_seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TextDataset(token_ids[split_idx:], args.max_seq_len),
        batch_size=args.batch_size,
        pin_memory=True,
    )

    config = ModelConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        norm_type=args.norm_type,
        activation=args.activation,
    )

    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    total_steps = (len(train_loader) // args.grad_accum) * args.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device.type else "cpu")
    early_stopping = Early_Stopping(patience=args.patience, min_delta=args.min_delta)

    start_epoch = 0
    start_batch_idx = 0
    global_step = 0

    if args.resume:
        latest_ckpt = find_latest_checkpoint(args.checkpoint_dir, args.run_name)
        if latest_ckpt:
            checkpoint = torch.load(
                latest_ckpt, map_location=device, weights_only=False
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

            early_stopping.best_loss = checkpoint.get(
                "early_stopping_best_loss", float("inf")
            )
            early_stopping.counter = checkpoint.get("early_stopping_counter", 0)

            start_epoch = checkpoint.get("epoch", 0)
            start_batch_idx = checkpoint.get("batch_idx", 0)
            global_step = checkpoint.get("step", 0)
            if start_batch_idx >= len(train_loader):
                start_epoch += 1
                start_batch_idx = 0
            logging.info(
                f"Successfully resumed from {latest_ckpt} at step {global_step} "
                f"(Epoch {start_epoch}, batch {start_batch_idx})"
            )
        else:
            logging.warning("No checkpoints found. Starting training from scratch.")

    if start_epoch >= args.max_epochs:
        logging.info(
            "Checkpoint already completed all requested epochs. Nothing to resume."
        )
        return

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        train_loader = create_train_loader(token_ids[:split_idx], args, epoch)
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.max_epochs}",
            initial=start_batch_idx if epoch == start_epoch else 0,
            total=len(train_loader),
        )
        optimizer.zero_grad()

        for i, (x, y) in enumerate(pbar):
            if epoch == start_epoch and i < start_batch_idx:
                continue

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda" if "cuda" in device.type else "cpu"
            ):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "train/loss": loss.item() * args.grad_accum,
                        "train/lr": current_lr,
                        "global_step": global_step,
                    }
                )
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item() * args.grad_accum:.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

                if global_step % args.eval_every == 0:
                    val_loss = estimate_loss(model, val_loader, device)
                    wandb.log({"val/loss": val_loss, "global_step": global_step})
                    logging.info(f"Step {global_step}: Val Loss = {val_loss:.4f}")

                    if early_stopping.step(val_loss):
                        logging.info("Early stopping triggered. Training stopped.")
                        break

                if global_step % args.save_every == 0:
                    ckpt_path = (
                        f"{args.checkpoint_dir}/{args.run_name}_s{global_step}.pt"
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        early_stopping,
                        config,
                        epoch,
                        global_step,
                        i + 1,
                        ckpt_path,
                    )

        if early_stopping.stop:
            break

        epoch_ckpt_path = (
            f"{args.checkpoint_dir}/{args.run_name}_epoch{epoch + 1}_s{global_step}.pt"
        )
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            early_stopping,
            config,
            epoch + 1,
            global_step,
            0,
            epoch_ckpt_path,
        )

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
