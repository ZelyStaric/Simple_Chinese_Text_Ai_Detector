#!/usr/bin/env python3
"""训练中文 AI 文本检测器（基线：chinese-roberta）。

用法:
  ./env/bin/python scripts/train.py --base hfl/chinese-roberta-wwm-ext --epochs 3 --bs 32 --max_len 512
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_jsonl(path):
    return load_dataset("json", data_files=path, split="train")


def make_tok(tok):
    def f(b):
        return tok(b["text"], truncation=True, max_length=MAXLEN)
    return f


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    p = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    pred = (p >= 0.5).astype(int)
    out = {
        "acc": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred, average="macro"),
    }
    try:
        out["auroc"] = roc_auc_score(labels, p)
    except ValueError:
        out["auroc"] = float("nan")
    # ECE (10 bins): confidence = max(p,1-p)
    conf = np.maximum(p, 1 - p)
    correct = (pred == labels)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    out["ece"] = ece
    return out


def fit_temperature(dev_logits, dev_labels):
    logits = torch.tensor(dev_logits, dtype=torch.float32)
    labels = torch.tensor(dev_labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    lossf = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach())


def main():
    global MAXLEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "baseline"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="小样本冒烟测试")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU")
    a = ap.parse_args()
    MAXLEN = a.max_len
    torch.manual_seed(a.seed)
    use_gpu = torch.cuda.is_available() and not a.cpu

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=2)

    train = load_jsonl(os.path.join(a.data_dir, "train.jsonl"))
    dev = load_jsonl(os.path.join(a.data_dir, "dev.jsonl"))
    if a.smoke:
        train = train.select(range(min(2000, len(train))))
        dev = dev.select(range(min(500, len(dev))))
    train = train.map(make_tok(tok), batched=True)
    dev = dev.map(make_tok(tok), batched=True)
    coll = DataCollatorWithPadding(tok)

    steps_per_epoch = max(1, len(train) // a.bs)
    warmup = int(0.1 * steps_per_epoch * a.epochs)
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs * 2,
        learning_rate=a.lr, eval_strategy="epoch", save_strategy="no",
        logging_steps=50, bf16=use_gpu, use_cpu=not use_gpu, report_to=[], seed=a.seed,
        warmup_steps=warmup, weight_decay=0.01,
    )
    tkw = dict(model=model, args=args, train_dataset=train, eval_dataset=dev,
               data_collator=coll, compute_metrics=compute_metrics)
    try:
        t = Trainer(processing_class=tok, **tkw)
    except TypeError:
        t = Trainer(tokenizer=tok, **tkw)
    t.train()

    os.makedirs(a.out, exist_ok=True)
    t.save_model(a.out)
    tok.save_pretrained(a.out)

    # dev 预测 -> 拟合温度
    pred = t.predict(dev)
    T = fit_temperature(pred.predictions, pred.label_ids)
    json.dump({"temperature": T, "base": a.base, "max_len": a.max_len},
              open(os.path.join(a.out, "temperature.json"), "w"), indent=2)
    eval_metrics = compute_metrics((pred.predictions / T, pred.label_ids))
    print("\n=== dev (temperature-calibrated) ===")
    print(json.dumps({k: round(float(v), 4) for k, v in eval_metrics.items()}, ensure_ascii=False))
    print("temperature =", round(T, 4))


if __name__ == "__main__":
    main()
