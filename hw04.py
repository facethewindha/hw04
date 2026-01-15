# train_speaker.py
import os
import json
import math
import random
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 87) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------------
# Dataset
# -----------------------------
class MyDataset(Dataset):
    def __init__(self, data_dir: str, segment_len: int = 128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # mapping.json: speaker name -> speaker id
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open("r", encoding="utf-8"))
        self.speaker2id = mapping["speaker2id"]

        # metadata.json
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(metadata_path.open("r", encoding="utf-8"))["speakers"]

        self.speaker_num = len(metadata.keys())

        # list of [feature_path, speaker_id]
        self.data = []
        for speaker in metadata.keys():
            for utt in metadata[speaker]:
                self.data.append([utt["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]

        # Load mel-spectrogram tensor saved in .pt
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Random segment for efficiency + augmentation
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start : start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)

        # speaker id for CrossEntropyLoss should be LongTensor scalar
        speaker = torch.tensor(speaker, dtype=torch.long)
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


def collate_batch(batch):
    """Make a batch with padding."""
    mels, speakers = zip(*batch)

    # pad along time dimension -> (B, T, 40)
    mels = pad_sequence(mels, batch_first=True, padding_value=-20)

    # speakers: tuple of scalar LongTensor -> (B,)
    speakers = torch.stack(speakers, dim=0).long()
    return mels, speakers


def get_dataloader(data_dir: str, batch_size: int, n_workers: int):
    dataset = MyDataset(data_dir)
    speaker_num = dataset.get_speaker_number()

    train_len = int(0.9 * len(dataset))
    lengths = [train_len, len(dataset) - train_len]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return train_loader, valid_loader, speaker_num


# -----------------------------
# Model
# -----------------------------
class Classifier(nn.Module):
    def __init__(self, d_model=512, n_spks=600, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(40, d_model)

        # Transformer encoder layer (default batch_first=False)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=16,
            dim_feedforward=256,
            dropout=dropout,
            activation="relu",
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.Sigmoid(),
            nn.Linear(2*d_model, n_spks),
        )

    def forward(self, mels):
        # mels: (B, T, 40)
        out = self.prenet(mels)          # (B, T, d_model)
        out = out.permute(1, 0, 2)       # (T, B, d_model) for TransformerEncoderLayer
        out = self.encoder_layer(out)    # (T, B, d_model)
        out = out.transpose(0, 1)        # (B, T, d_model)

        # mean pooling over time
        stats = out.mean(dim=1)          # (B, d_model)
        out = self.pred_layer(stats)     # (B, n_spks)
        return out


# -----------------------------
# LR Scheduler
# -----------------------------
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step: int):
        # warmup: 0 -> 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # cosine decay: 1 -> 0
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# -----------------------------
# Train/Eval helpers
# -----------------------------
def model_fn(batch, model, criterion, device):
    mels, labels = batch
    mels = mels.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    outs = model(mels)
    loss = criterion(outs, labels)

    preds = outs.argmax(dim=1)
    accuracy = (preds == labels).float().mean()
    return loss, accuracy


@torch.no_grad()
def valid(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")
    seen = 0

    for i, batch in enumerate(dataloader):
        loss, acc = model_fn(batch, model, criterion, device)
        running_loss += loss.item()
        running_acc += acc.item()

        bs = batch[0].size(0)
        seen += bs
        pbar.update(bs)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{running_acc / (i + 1):.2f}",
        )

    pbar.close()
    model.train()
    return running_acc / max(1, len(dataloader))


class InferenceDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open("r", encoding="utf-8"))
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        mel = torch.FloatTensor(mel)  # 保证是 float tensor
        return feat_path, mel
def inference_collate_batch(batch):
    """Collate a batch of data for inference."""
    feat_paths, mels = zip(*batch)
    # 保险起见：推理也做 padding（batch_size=1 时无影响，batch_size>1 时需要）
    mels = pad_sequence(mels, batch_first=True, padding_value=-20)
    return feat_paths, mels

@torch.no_grad()
def run_inference(data_dir: str, model_path: str, output_path: str, batch_size: int = 1, n_workers: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open("r", encoding="utf-8"))

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        collate_fn=inference_collate_batch,
    )
    print("[Info]: Finish loading inference data!", flush=True)

    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("[Info]: Finish loading model!", flush=True)

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader, desc="Infer", ncols=0):
        mels = mels.to(device)
        outs = model(mels)
        preds = outs.argmax(1).cpu().numpy()
        for feat_path, pred in zip(feat_paths, preds):
            results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    print(f"[Info]: Inference done. CSV saved to {output_path}")


# -----------------------------
# Config + Main
# -----------------------------
def parse_args():
    """
    你可以直接改这里的参数：
    mode: "train" or "infer"
    """
    config = {
        "mode": "train",          # 改成 "infer" 就走推理
        "data_dir": "./Dataset",
        "save_path": "model.ckpt",

        # train
        "batch_size": 32,
        "n_workers": 4,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 70000,
        "seed": 87,

        # infer
        "model_path": "model.ckpt",
        "output_path": "./output.csv",
        "infer_batch_size": 1,
        "infer_workers": 0,
    }
    return config



def main(
    mode,
    data_dir,
    save_path,
    batch_size,
    n_workers,
    valid_steps,
    warmup_steps,
    total_steps,
    save_steps,
    seed,
    model_path,
    output_path,
    infer_batch_size,
    infer_workers,
):
    if mode == "infer":
        run_inference(
            data_dir=data_dir,
            model_path=model_path,
            output_path=output_path,
            batch_size=infer_batch_size,
            n_workers=infer_workers,
        )
        return

    # ---- train ----
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Use device: {device}")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print("[Info] Finish loading data!", flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print("[Info] Finish creating model!", flush=True)

    best_acc = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, acc = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_acc = acc.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pbar.update(1)
        pbar.set_postfix(loss=f"{batch_loss:.2f}", accuracy=f"{batch_acc:.2f}", step=step + 1)

        if (step + 1) % valid_steps == 0:
            pbar.close()
            val_acc = valid(valid_loader, model, criterion, device)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_acc:.4f})")

    pbar.close()
    if best_state_dict is not None:
        torch.save(best_state_dict, save_path)
        print(f"[Info] Training done. Best model saved to {save_path} (acc={best_acc:.4f})")



if __name__ == "__main__":
    main(**parse_args())
