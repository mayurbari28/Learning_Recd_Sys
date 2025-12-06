import torch
from model_sasrec import SASRec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def recommend(model, seq, k=10):
    model.eval()
    seq = torch.tensor([seq]).to(device)

    with torch.no_grad():
        logits = model(seq)
        top_k = torch.topk(logits, k=k).indices[0].cpu().tolist()

    return top_k
