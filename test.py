from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("./bert_article_bias/final_model")  # or "./bert_article_bias/best_model"
LABELS_FILE = Path("./bert_article_bias/label_encoder_classes.txt")
MODEL_NAME_FALLBACK = "bert-base-uncased"  # only used if tokenizer isn't saved in folder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load labels (same order as LabelEncoder classes)
labels = [line.strip() for line in LABELS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

# Load model + tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FALLBACK)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

def predict_bias(text: str, max_length: int = 128):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).detach().cpu()

    pred_id = int(torch.argmax(probs).item())
    pred_label = labels[pred_id] if pred_id < len(labels) else str(pred_id)

    return pred_label, {labels[i]: float(probs[i]) for i in range(min(len(labels), probs.numel()))}

if __name__ == "__main__":
    text = input("Paste article text:\n")
    label, probs = predict_bias(text)
    print("\nPrediction:", label)
    print("Probabilities:")
    for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v:.4f}")
