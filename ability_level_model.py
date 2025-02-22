import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    # data in size [num_samples, num_questions, num_data]
    def __init__(self, num_questions: int=244,
                 num_data: int=4, # discrimination, difficulty, response, response time
                 ) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        num_entries = num_questions * num_data
        self.fc1 = nn.Linear(num_entries, num_entries*2)
        self.fc2 = nn.Linear(num_entries*2, num_entries*4)
        self.fc3 = nn.Linear(num_entries*4, num_entries//6)
        self.fc4 = nn.Linear(num_entries//6, 10)
        self.fc5 = nn.Linear(10, 2)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: embed time
        input_data = x.flatten(1).float()
        out = self.relu(self.fc1(input_data))
        out = self.relu(self.dropout(self.fc2(out)))
        out = self.relu(self.dropout(self.fc3(out)))
        out = self.relu(self.dropout(self.fc4(out)))
        out = self.fc5(out)

        return out