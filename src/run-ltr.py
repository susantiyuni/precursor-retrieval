import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_loss import RankLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import utils as u
u.set_seed(66)
u.set_deterministic()

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="runs_ce/", help="Output directory")
parser.add_argument("--model_name", default="distilbert-base-uncased")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument('--loss_type', type=str, default='lambda_loss') #list_net #rank_net,
# pointwise_rmse
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("data/candidate-pool.jsonl")
FEATURE_PATH = Path("data/sparql_feats.jsonl")
TOP_K = 50
DATA_LIMIT = 100

print(f"\n===== {args.model_name} {args.loss_type} =====")

# ================================
# DATASET
# ================================
class ListwiseDataset(Dataset):
    def __init__(self, entries, leave_out_idx, tokenizer):
        self.entries = []
        self.tokenizer = tokenizer

        for i, entry in enumerate(entries):
            if i == leave_out_idx:
                continue
            self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        q = u.build_query_text(entry["query_paper"])
        # print (q)
        candidates = entry["candidates"]
        candidates = sorted(
            entry["candidates"],
            key=lambda c: c["llm_score"],
            reverse=True
        )[:DATA_LIMIT]

        # print (u.build_candidate_text(candidates[0]))
        # sys.exit()

        texts = [u.build_candidate_text(c) for c in candidates]        
        scores = [c["llm_score"] for c in candidates]
        # labels = normalize_labels(scores) #for ranknet better?
        labels = u.quantize_labels(scores)

        enc = self.tokenizer(
            [q] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"],             # [n, L]
            "attention_mask": enc["attention_mask"],   # [n, L]
            "labels": torch.tensor(labels, dtype=torch.float)
        }

# ================================
# TRAIN
# ================================
def train_model(dataset, model, tokenizer):

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_function = getattr(RankLoss, args.loss_type)

    model.train()

    for epoch in range(args.epochs):
        losses = []

        for batch in tqdm(loader, desc=f"Epoch {epoch}"):

            input_ids = batch["input_ids"].to(device)        # [N, L]
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)              # [N]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits.squeeze(-1)              # [N]

            mask = (labels != -100)
            logits = logits.masked_fill(~mask, -1e9)

            loss = loss_function(
                logits.unsqueeze(0),   # NOW correct: [1, N]
                labels.unsqueeze(0)
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        print(f"Epoch {epoch} loss: {np.mean(losses):.4f}")

    return model

# ================================
# RANK
# ================================
def rank(model, tokenizer, query, candidates):

    model.eval()

    q = u.build_query_text(query)
    candidates = sorted(
        candidates,
        key=lambda c: c["llm_score"],
        reverse=True
    )[:DATA_LIMIT]

    texts = [u.build_candidate_text(c) for c in candidates]

    enc = tokenizer(
        [q] * len(texts),
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        scores = model(**enc).logits.squeeze(-1).cpu().numpy()

    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])

    return [
        {
            "paper": c["paper"],
            "score": float(s),
            "rank": i + 1
        }
        for i, (c, s) in enumerate(ranked[:TOP_K])
    ]

# ================================
# LOO EVAL
# ================================
def run_loo():
    entries = u.load_data(DATA_PATH, FEATURE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    all_results = []

    for leave_out in range(len(entries)):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=1
        ).to(device)

        print(f"\n===== LOO {leave_out} =====")

        dataset = ListwiseDataset(entries, leave_out, tokenizer)

        model = train_model(dataset, model, tokenizer)

        test_entry = entries[leave_out]

        ranked = rank(
            model,
            tokenizer,
            test_entry["query_paper"],
            test_entry["candidates"]
        )

        print("\nTop results:")
        for r in ranked[:5]:
            print(r)

        output = {
            "query_paper": test_entry["query_paper"],
            "ranked_candidates": ranked
        }

        all_results.append(output)

        # # ---------------- SAVE PER QUERY ----------------
        # out_path = OUT_DIR / f"run_{leave_out}_ce.jsonl"
        # with out_path.open("w") as f:
        #     f.write(json.dumps(output) + "\n")

    global_path = OUT_DIR / f"run_global_{args.loss_type[:5]}_{args.model_name[:7]}.jsonl"
    with global_path.open("w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n===== Saved: {global_path} =====")
    return all_results    

if __name__ == "__main__":
    run_loo()
    
