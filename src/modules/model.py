import torch.nn as nn
import torch

class MLP(nn.Module):

    def __init__(self, num_class=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(180, 2050),
            nn.ReLU(),
            nn.Linear(2050, 1280),
            nn.ReLU(),
            nn.Linear(1280, 560),
            nn.ReLU(),
            nn.Linear(560, 120),
            nn.ReLU(),
            nn.Linear(120, num_class)
        )

    def init_weights(self, f_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        checkpoint = torch.load(f_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        x = self.fc(x)
        return x
