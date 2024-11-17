import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.norm5 = nn.LayerNorm(hidden_size)
        
        self.layer6 = nn.Linear(hidden_size, hidden_size)
        self.norm6 = nn.LayerNorm(hidden_size)
        
        self.layer7 = nn.Linear(hidden_size, hidden_size)
        self.norm7 = nn.LayerNorm(hidden_size)
        
        self.layer8 = nn.Linear(hidden_size, hidden_size)
        self.norm8 = nn.LayerNorm(hidden_size)
        
        self.layer9 = nn.Linear(hidden_size, output_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        nn.init.xavier_uniform_(self.layer5.weight)
        nn.init.xavier_uniform_(self.layer6.weight)
        nn.init.xavier_uniform_(self.layer7.weight)
        nn.init.xavier_uniform_(self.layer8.weight)
        nn.init.xavier_uniform_(self.layer9.weight)


        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
        nn.init.zeros_(self.layer4.bias)
        nn.init.zeros_(self.layer5.bias)
        nn.init.zeros_(self.layer6.bias)
        nn.init.zeros_(self.layer7.bias)
        nn.init.zeros_(self.layer8.bias)
        nn.init.zeros_(self.layer9.bias)

        
    def forward(self, x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = nn.Tanh()(x)
        
        x = self.layer2(x)
        x = self.norm2(x)
        x = nn.Tanh()(x)
        
        x = self.layer3(x)
        x = self.norm3(x)
        x = nn.Tanh()(x)
        
        x = self.layer4(x)
        x = self.norm4(x)
        x = nn.Tanh()(x)
        
        x = self.layer5(x)
        x = self.norm5(x)
        x = nn.Tanh()(x)
        
        x = self.layer6(x)
        x = self.norm6(x)
        x = nn.Tanh()(x)
        
        x = self.layer7(x)
        x = self.norm7(x)
        x = nn.Tanh()(x)
        
        x = self.layer8(x)
        x = self.norm8(x)
        x = nn.Tanh()(x)
        
        x = self.layer9(x)
        return x


#FUNZIONI TRAINING
def local_monotonicity_loss(rates, rud): #misure velocità angolari e predizioni del rudder della rete
    delta_rates = torch.tanh(5*(rates[1:] - rates[:-1]))  
    delta_rudder = torch.tanh(5*(rud[1:] - rud[:-1]))
    sign_diff = delta_rates * delta_rudder[:, None]
    loss = torch.mean(1 - sign_diff)
    return loss

# Physics-Informed Loss combining MSE and Local Monotonicity Loss
def physics_informed_loss(pred, target, r, lambda_lm):
    mse_loss = nn.MSELoss()(pred, target)
    monotonicity_loss = local_monotonicity_loss(r, pred) #pred devono essere i valori del rudder predetti dalla rete, mentre r sono le misure di velocità angolari
    #print("MSE", mse_loss, "Informed", monotonicity_loss)
    total_loss = mse_loss + lambda_lm * monotonicity_loss
    return total_loss
    
def cyclical_annealing_scheduler(epoch, num_epochs, lambda_max, num_cycles=5, r=0.5):
    cycle_length = num_epochs // num_cycles
    epoch_in_cycle = epoch % cycle_length
    beta = epoch_in_cycle / cycle_length
    if beta > r:
        return lambda_max * (1 - beta) / (1 - r)
    else:
        return lambda_max