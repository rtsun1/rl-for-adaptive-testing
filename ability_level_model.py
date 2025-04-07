import torch
import torch.nn as nn

# Define constants for integration
QUADRATURE_POINTS = 41
ABILITY_RANGE = (-4, 4)

def calculate_eap_ability(responses, discrimination, difficulty):
    # Ensure everything is already on the same device (GPU or CPU)
    device = responses.device

    # Define quadrature points and weights for numerical integration
    abilities = torch.linspace(ABILITY_RANGE[0], ABILITY_RANGE[1], QUADRATURE_POINTS, device=device) 

    # Broadcast the abilities array to allow vectorized computation
    abilities_broadcasted = abilities.view(1, 1, -1)  # Shape: (1, 1, QUADRATURE_POINTS)
    
    # Compute the probability of correct response for all samples and items simultaneously
    prob_correct = 1 / (1 + torch.exp(-discrimination.unsqueeze(2) * (abilities_broadcasted - difficulty.unsqueeze(2))))

    # Compute the probability of incorrect response
    prob_incorrect = 1 - prob_correct

    # Compute the log likelihoods for correct and incorrect responses
    log_likelihood = responses.unsqueeze(2) * torch.log(prob_correct) + (1 - responses.unsqueeze(2)) * torch.log(prob_incorrect)

    # Sum log likelihoods over all items for each sample
    posterior_numerator = log_likelihood.sum(dim=1)  # Shape: (sample_size, QUADRATURE_POINTS)

    # Calculate posterior denominator with stability adjustment
    max_numerator = torch.max(posterior_numerator, dim=1, keepdim=True)[0]  # Stability trick to avoid overflow
    posterior_denominator = torch.exp(posterior_numerator - max_numerator)
    posterior_denominator /= posterior_denominator.sum(dim=1, keepdim=True)

    # Compute EAP ability for each sample
    eap_ability = torch.sum(abilities * posterior_denominator, dim=1)

    # Compute the standard error of the ability estimate for each sample
    eap_variance = torch.sum((abilities - eap_ability.unsqueeze(1)) ** 2 * posterior_denominator, dim=1)
    eap_standard_error = torch.sqrt(eap_variance)

    return eap_ability, eap_standard_error
    

class RegressionModel(nn.Module):
    # data in size [num_samples, num_questions, num_data]
    def __init__(self, num_questions: int=244,
                 num_data: int=4, # discrimination, difficulty, response, response time
                 ) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        num_entries = num_questions * num_data
        self.fc1 = nn.Linear(num_entries + 2, num_entries*2)
        self.fc2 = nn.Linear(num_entries*2, num_entries*1)
        #self.fc3 = nn.Linear(num_entries*1 + 2, num_entries//6)
        self.fc3 = nn.Linear(num_entries*1, num_entries//6)
        self.fc4 = nn.Linear(num_entries//6, num_entries//6)
        self.fc5 = nn.Linear(num_entries//6, 10)
        self.fc6 = nn.Linear(10, 1)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: embed time
        #"""
        x_clean = x[torch.any(x != torch.tensor([-1, -1, -1, -1]).to(x.device), dim=-1)]
        x_clean = x_clean.view(x.shape[0], -1, x.shape[2])

        eap, eap_se = calculate_eap_ability(x_clean[:,:,2], x_clean[:,:,0], x_clean[:,:,1])
        eap = eap.unsqueeze(-1).float()
        eap_se = eap_se.unsqueeze(-1).float()
        #"""
        input_data = x.flatten(1).float()
        input_data = torch.cat([input_data, eap, eap_se], dim=1)
        out = self.relu(self.fc1(input_data))
        out = self.relu(self.fc2(out))
        
        out = self.relu(self.fc3(out))
        #out = self.relu(self.dropout(self.fc2(out)))
        #out = self.relu(self.dropout(self.fc3(out)))
        #out = torch.cat([out, eap, eap_se], dim=1)
        #out = self.relu(self.dropout(self.fc4(out)))
        out = self.relu(self.fc4(out))
        out = self.relu(self.fc5(out))
        out = self.fc6(out)

        return out