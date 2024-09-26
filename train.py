import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import digitRecog

class AudioDataSet(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir 
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(int(file_path[len(self.data_dir)+1: len(self.data_dir)+2]), dtype=torch.int64)

def load_batches(batch_size):
    data_dir = "dataset/forModel/train"
    dataset = AudioDataSet(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

BATCH_SIZE = 64
loader = load_batches(batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classifier = digitRecog().to(device)

params = list(num_classifier.parameters())
optim = torch.optim.Adam(params,lr = 1e-3)

EPOCH = 25

for i in range(EPOCH):
    for x,y in loader:
        X,Y = x.to(device), y.to(device)
        # not goot pratice ik
        if X.shape[0] < BATCH_SIZE:
            missing = BATCH_SIZE - X.shape[0]
            X = torch.cat((prev_X[-missing:], X), dim=0)
            Y = torch.cat((prev_Y[-missing:], Y), dim=0)
        prev_X, prev_Y = X, Y
        X = X.view(BATCH_SIZE,1,40,99)

        y_hat = num_classifier(X)
        loss = F.cross_entropy(y_hat, Y)

        optim.zero_grad()
        loss.backward()

        optim.step()
    print(f"Epoch {i}/{EPOCH}, Loss {loss.item()}")

# -----------------------------------------------------------------------------------------------------------
torch.save(num_classifier.state_dict(), "numClassifier.pth")


