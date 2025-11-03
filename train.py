# train_lstm_ner.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse

# ---------------------------
# 1) HÀM ĐỌC FILE CoNLL
# ---------------------------
def load_conll(path):
    sentences, labels = [], []
    sent, lab = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    labels.append(lab)
                    sent, lab = [], []
            else:
                # mong định dạng: token <space> tag
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[-1]
                else:
                    continue
                sent.append(token)
                lab.append(tag)
    # nếu file không kết thúc bằng dòng trống
    if sent:
        sentences.append(sent)
        labels.append(lab)
    return sentences, labels

# ---------------------------
# 2) XÂY VOCAB
# ---------------------------
def build_vocab(sentences, min_freq=1):
    freq = {}
    for s in sentences:
        for w in s:
            freq[w] = freq.get(w, 0) + 1
    # reserve 0: PAD, 1: UNK
    word2idx = {"<PAD>":0, "<UNK>":1}
    for w, c in freq.items():
        if c >= min_freq:
            word2idx[w] = len(word2idx)
    return word2idx

def build_tag_map(labels):
    tag2idx = {"<PAD>":0}   # PAD tag mapped to 0 for ignore_index
    for seq in labels:
        for t in seq:
            if t not in tag2idx:
                tag2idx[t] = len(tag2idx)
    return tag2idx

# ---------------------------
# 3) DATASET & COLLATE
# ---------------------------
class NERDataset(Dataset):
    def __init__(self, sents, tags, word2idx, tag2idx):
        self.sents = sents
        self.tags = tags
        self.w2i = word2idx
        self.t2i = tag2idx

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words = [self.w2i.get(w, self.w2i["<UNK>"]) for w in self.sents[idx]]
        labs  = [self.t2i[t] for t in self.tags[idx]]
        return torch.tensor(words, dtype=torch.long), torch.tensor(labs, dtype=torch.long)

def collate_fn(batch):
    words, labels = zip(*batch)
    lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    words_padded = pad_sequence(words, batch_first=True, padding_value=0)   # PAD idx = 0
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0) # PAD tag idx = 0
    return words_padded, labels_padded, lengths

# ---------------------------
# 4) MÔ HÌNH LSTM 1-CHIỀU
# ---------------------------
class LSTM_NER(nn.Module):
    def __init__(self, vocab_size, tag_size, emb_dim=128, hidden_dim=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # **UNI-DIRECTIONAL LSTM** (bidirectional=False)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, tag_size)

    def forward(self, x, lengths):
        # x: (B, L)
        emb = self.embedding(x)                  # (B, L, E)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)       # process sequence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, L, H)
        logits = self.fc(out)                   # (B, L, tag_size)
        return logits

# ---------------------------
# 5) TRAIN / EVAL HELPERS
# ---------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for words, labels, lengths in dataloader:
        words = words.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(words, lengths)   # (B, L, C)
        B, L, C = logits.shape
        loss = criterion(logits.view(-1, C), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, idx2tag, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for words, labels, lengths in dataloader:
            words = words.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits = model(words, lengths)          # (B, L, C)
            preds = logits.argmax(-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            lengths_np = lengths.cpu().numpy()
            for i in range(len(lengths_np)):
                L_i = lengths_np[i]
                for j in range(L_i):
                    true_idx = labels_np[i][j]
                    pred_idx = preds[i][j]
                    # skip PAD tag (index 0)
                    if true_idx == 0:
                        continue
                    y_true.append(idx2tag[true_idx])
                    y_pred.append(idx2tag[pred_idx])
    # classification_report requires list of labels; we print token-level report
    report = classification_report(y_true, y_pred, zero_division=0)
    return report

# ---------------------------
# 6) MAIN: load data, build, train
# ---------------------------
def main(args):
    # load conll
    sentences, tags = load_conll("D:/Code/Week_3/test_word.conll")
    print(f"Loaded {len(sentences)} sentences.")

    # build vocabs
    word2idx = build_vocab(sentences)
    tag2idx = build_tag_map(tags)
    idx2tag = {v:k for k,v in tag2idx.items()}

    print("Vocab size:", len(word2idx), "Tag size:", len(tag2idx))

    # split
    s_train, s_test, t_train, t_test = train_test_split(sentences, tags, test_size=args.test_size, random_state=42)

    train_ds = NERDataset(s_train, t_train, word2idx, tag2idx)
    test_ds  = NERDataset(s_test, t_test, word2idx, tag2idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_NER(vocab_size=len(word2idx), tag_size=len(tag2idx),
                     emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, dropout=args.dropout).to(device)

    # criterion: ignore_index = PAD tag index (0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}  Train loss: {loss:.4f}")
        if epoch % args.eval_every == 0 or epoch==args.epochs:
            print("Evaluating on test set...")
            rep = evaluate(model, test_loader, idx2tag, device)
            print(rep)

    # save model & vocabs
    torch.save({
        "model_state": model.state_dict(),
        "word2idx": word2idx,
        "tag2idx": tag2idx
    }, args.save_path)
    print("Saved model to", args.save_path)

    # demo predict: take first 2 sentences from test set
    model.eval()
    print("\nDemo predictions on a few test sentences:")
    for sent, tg in list(zip(s_test, t_test))[:3]:
        with torch.no_grad():
            words_idx = torch.tensor([[word2idx.get(w, word2idx["<UNK>"]) for w in sent]], dtype=torch.long).to(device)
            lengths = torch.tensor([len(sent)]).to(device)
            out = model(words_idx, lengths)  # (1, L, C)
            preds = out.argmax(-1).cpu().numpy()[0]
            print("SENT:", " ".join(sent))
            print("PRED:", [idx2tag[p] for p in preds[:len(sent)]])
            print("TRUE:", tg)
            print("-"*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train.txt", help="path to CoNLL file")
    parser.add_argument("--save_path", type=str, default="lstm_ner.pt", help="where to save model")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--eval_every", type=int, default=1)
    args = parser.parse_args()
    main(args)
