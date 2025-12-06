import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import SASRecDataset
from model_sasrec import SASRec

def train_sasrec(df):
    dataset = SASRecDataset(df)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_items = df["item"].nunique()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SASRec(num_items).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(10):
        for seq, target in loader:
            seq, target = seq.to(device), target.to(device)

            logits = model(seq)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

    torch.save(model.state_dict(), "models/sasrec.pt")
    return model
