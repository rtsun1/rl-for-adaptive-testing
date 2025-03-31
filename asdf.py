import torch
import numpy as np
import torch.nn as nn

class SingleActionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # print("DEBUG single_observation_space length: ", len(env.single_observation_space))
        # print("DEBUG action_space length: ", env.action_space.n)

        input_size = 244 + 2
        # e.g. + [0 0 0 0 1 0 0 0 0] for action 4
        self.network = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 240),
            nn.ReLU(),
            nn.Linear(240, 1),
        )

    def forward(self, x):
        return self.network(
            x
        )

obs = np.random.rand(10, 2)  # (n_features,)

one_hot = torch.eye(244)  # (n_actions, n_actions)
one_hot = one_hot.unsqueeze(0)  # (1, n_actions, n_actions)
one_hot = one_hot.repeat(10, 1, 1)  # (batch_size, n_actions, n_actions)
x = torch.Tensor(obs).unsqueeze(1).expand(-1, 244, -1)  # (batch_size, n_actions, n_features)

x = torch.cat((x, one_hot), dim=2)  # (batch_size, n_actions, n_features + n_actions)
net = SingleActionNetwork()
print(x.shape)

print(net(x).shape)
