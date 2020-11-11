import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class FactorisedNoisyLayer(nn.Module):
    def __init__(self, input_features, output_features, sigma=0.5):
        super(FactorisedNoisyLayer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.sigma = sigma
        self.bound = input_features**(-0.5)

        self.mu_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.mu_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))

        self.epsilon_input = None
        self.epsilon_output = None
        self.register_buffer('epsilon_input', torch.FloatTensor(input_features))
        self.register_buffer('epsilon_output', torch.FloatTensor(output_features))

        self.parameter_initialization()
        self.sample_noise()

    def parameter_initialization(self):
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.sigma_bias.data.fill_(self.sigma * self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)
        self.sigma_weight.data.fill_(self.sigma * self.bound)

    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        if not self.training:
            return F.linear(x, weight=self.mu_weight, bias=self.mu_bias)

        if sample_noise:
            self.sample_noise()

        weight = self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight
        bias = self.sigma_bias * self.epsilon_output + self.mu_bias
        return F.linear(x, weight=weight, bias=bias)

    def sample_noise(self):
        self.epsilon_input = self.get_noise_tensor(self.input_features)
        self.epsilon_output = self.get_noise_tensor(self.output_features)

    def get_noise_tensor(self, features: int) -> torch.Tensor:
        noise = torch.FloatTensor(features).uniform_(-self.bound, self.bound)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))


class Dueling_DQN(nn.Module):
    def __init__(self, state_inputs, actions_num):
        super(Dueling_DQN, self).__init__()
        self.linear_input_size = state_inputs
        self.feature_layers = nn.Sequential(
            nn.Linear(self.linear_input_size, 1024),
            Swish(inplace=True),
            # FactorisedNoisyLayer(1024, 512),
            nn.Linear(1024, 512),
            Swish(inplace=True),
            # FactorisedNoisyLayer(512, 256),
            nn.Linear(512, 256),
            Swish(inplace=True),

        )
        self.value_fc = nn.Sequential(
            # FactorisedNoisyLayer(256, 256),
            nn.Linear(256, 256),
            Swish(inplace=True),
            nn.Linear(256, 256),
            Swish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.advantage_fc = nn.Sequential(
            # FactorisedNoisyLayer(256, 256),
            nn.Linear(256, 256),
            Swish(inplace=True),
            nn.Linear(256, 256),
            Swish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, actions_num),
            # nn.BatchNorm1d(actions_num)
        )

    def forward(self, x):
        x = x.view(-1, self.linear_input_size)
        x = self.feature_layers(x)
        # x = self.goal_fc(x)
        v = self.value_fc(x)
        advantage = self.advantage_fc(x)
        advantage -= advantage.mean()
        return v + advantage
