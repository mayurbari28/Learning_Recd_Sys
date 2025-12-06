import torch
from torch.utils.data import Dataset

class SASRecDataset(Dataset):
    def __init__(self, df, max_len=50):
        self.users = df["user"].unique()
        self.df = df
        self.max_len = max_len

        self.user_sequences = self.build_sequences()

    def build_sequences(self):
        sequences = {}
        for u in self.users:
            seq = self.df[self.df.user == u].item.tolist()
            sequences[u] = seq
        return sequences

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_sequences[user]

        seq_input = seq[:-1][-self.max_len:]
        target = seq[-1]

        pad_len = self.max_len - len(seq_input)
        seq_input = [0]*pad_len + seq_input

        return torch.tensor(seq_input), torch.tensor(target)
