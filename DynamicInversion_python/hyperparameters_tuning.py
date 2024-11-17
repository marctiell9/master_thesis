import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from feedforward_nn import*

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#device="cpu"

# Load the DataSet
mat_data = loadmat('NODEdataset_rudfiltered.mat')

ail_sp = torch.tensor(mat_data["ail_sp"],dtype=torch.float32).to(device)
elev_sp = torch.tensor(mat_data["elev_sp"],dtype=torch.float32).to(device)
rud_sp = torch.tensor(mat_data["rud_sp"],dtype=torch.float32).to(device)
flaps_sp = torch.tensor(mat_data["flaps_eq"],dtype=torch.float32).to(device)
load_meas = torch.tensor(mat_data["load_meas"],dtype=torch.float32).to(device)
length_meas = torch.tensor(mat_data["length_meas"],dtype=torch.float32).to(device)
time_vector = torch.tensor(mat_data["time_meas_sec_validation"],dtype=torch.float32).to(device) #ho preso tutto il data set

N_meas = torch.tensor(mat_data["N_meas"],dtype=torch.float32).to(device)
E_meas = torch.tensor(mat_data["E_meas"],dtype=torch.float32).to(device)
h_meas = torch.tensor(mat_data["h_meas"],dtype=torch.float32).to(device)
u_meas = torch.tensor(mat_data["u_meas"],dtype=torch.float32).to(device)
v_meas = torch.tensor(mat_data["v_meas"],dtype=torch.float32).to(device)
w_meas = torch.tensor(mat_data["w_meas"],dtype=torch.float32).to(device)
phi_meas = torch.tensor(mat_data["phi_meas"],dtype=torch.float32).to(device) #attenzione alle grandezze di misura
theta_meas = torch.tensor(mat_data["theta_meas"],dtype=torch.float32).to(device)
psi_meas = torch.tensor(mat_data["psi_meas"],dtype=torch.float32).to(device)
p_meas = torch.tensor(mat_data["p_meas"],dtype=torch.float32).to(device)
q_meas = torch.tensor(mat_data["q_meas"],dtype=torch.float32).to(device)
r_meas = torch.tensor(mat_data["r_meas"],dtype=torch.float32).to(device)

#identification dataset
identification_index = 66666
#identification_index = 114000
ail_sp_train = ail_sp[0:identification_index]
elev_sp_train = elev_sp[0:identification_index]
rud_sp_train = rud_sp[0:identification_index]
flaps_sp_train = flaps_sp[0:identification_index]
load_meas_train = load_meas[0:identification_index]
length_meas_train = length_meas[0:identification_index]
time_id_train = time_vector[0:identification_index]

N_meas_train = N_meas[0:identification_index]
E_meas_train = E_meas[0:identification_index]
h_meas_train = h_meas[0:identification_index]
u_meas_train = u_meas[0:identification_index]
v_meas_train = v_meas[0:identification_index]
w_meas_train = w_meas[0:identification_index]
phi_meas_train = phi_meas[0:identification_index]
theta_meas_train = theta_meas[0:identification_index]
psi_meas_train = psi_meas[0:identification_index]
p_meas_train = p_meas[0:identification_index]
q_meas_train = q_meas[0:identification_index]
r_meas_train = r_meas[0:identification_index]

train_inputs = torch.stack([N_meas_train, E_meas_train, h_meas_train, u_meas_train, v_meas_train, w_meas_train,
                         phi_meas_train*torch.pi/180, theta_meas_train*torch.pi/180, psi_meas_train*torch.pi/180,
                         p_meas_train*torch.pi/180, q_meas_train*torch.pi/180, r_meas_train*torch.pi/180]).squeeze(-1).T
train_target = rud_sp_train*torch.pi/180

#validation dataset 
validation_index = 88867
#validation_index = 128250 #se utilizzo lo split 80-10-10

ail_sp_val = ail_sp[identification_index:validation_index]
elev_sp_val = elev_sp[identification_index:validation_index]
rud_sp_val = rud_sp[identification_index:validation_index]
flaps_sp_val = flaps_sp[identification_index:validation_index]
load_meas_val = load_meas[identification_index:validation_index]
length_meas_val = length_meas[identification_index:validation_index]
time_id_val = time_vector[identification_index:validation_index]

N_meas_val = N_meas[identification_index:validation_index]
E_meas_val = E_meas[identification_index:validation_index]
h_meas_val = h_meas[identification_index:validation_index]
u_meas_val = u_meas[identification_index:validation_index]
v_meas_val = v_meas[identification_index:validation_index]
w_meas_val = w_meas[identification_index:validation_index]
phi_meas_val = phi_meas[identification_index:validation_index]
theta_meas_val = theta_meas[identification_index:validation_index]
psi_meas_val = psi_meas[identification_index:validation_index]
p_meas_val = p_meas[identification_index:validation_index]
q_meas_val = q_meas[identification_index:validation_index]
r_meas_val = r_meas[identification_index:validation_index]

validation_inputs = torch.stack([N_meas_val, E_meas_val, h_meas_val, u_meas_val, v_meas_val, w_meas_val,
                                phi_meas_val*torch.pi/180, theta_meas_val*torch.pi/180, psi_meas_val*torch.pi/180*torch.pi/180,
                                p_meas_val*torch.pi/180, q_meas_val*torch.pi/180, r_meas_val*torch.pi/180]).squeeze(-1).T
validation_target = rud_sp_val*torch.pi/180

#Dataset
train_dataset = TensorDataset(train_inputs, train_target)
val_dataset = TensorDataset(validation_inputs, validation_target)

#class with conditional initialization depending on the activation function
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size, activation_fn, initialization_fn):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size)) 
        activation_layer = activation_fn()
        layers.append(activation_layer)

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size)) 
            layers.append(activation_fn())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
        # Apply weight initialization based on the activation function
        self.model.apply(initialization_fn)

    def forward(self, x):
        return self.model(x)
    


def init_weights_he(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)
        if layer.bias is not None:  
            nn.init.zeros_(layer.bias)

def init_weights_xavier(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:  
            nn.init.zeros_(layer.bias)

def objective(trial):
    # Define the search space
    activation_choice = trial.suggest_categorical("activation", ["ReLU", "Tanh"])
    num_layers = trial.suggest_int("num_layers", 2, 12) #prende un intero nel range 2-12
    hidden_size = trial.suggest_int("hidden_size", 20, 64)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    #learning_rate = trial.suggest_loguniform("lr", 1e-6, 1e-4) #learning rate sampled logrithmically between these two values

    if activation_choice == "ReLU":
        activation_fn = nn.ReLU
        initialization_fn = init_weights_he
    else:
        activation_fn = nn.Tanh
        initialization_fn = init_weights_xavier

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) 
    
    model = FeedForwardNN(input_size=12, num_layers=num_layers, hidden_size=hidden_size, 
                          output_size=1, activation_fn=activation_fn, initialization_fn=initialization_fn).to(device)
    num_epochs = 300
    num_cycles = 5
    lambda_max = 0.01  # Maximum value for the local monotonicity loss weight
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), 1e-5)
    val_loss_criterion = nn.MSELoss()



    for epoch in range(num_epochs):
        lambda_lm = cyclical_annealing_scheduler(epoch, num_epochs, lambda_max, num_cycles, 0.5)
        model.train()
        for batch_idx, (batch_input, batch_target) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            pred = model(batch_input)
            rates = batch_input[:,10:13] #caso in cui consideriamo p,q,r per la monotonicity loss, indice 13 non incluso
            train_loss = physics_informed_loss(pred, batch_target, rates, lambda_lm)
            train_loss.backward()
            optimizer.step()
        print("EPOCH", epoch)

    model.eval()
    with torch.no_grad(): #dopo il training loop valutiamo il modello
        val_loss = 0.0    
        val_outputs = model(validation_inputs)
        loss = val_loss_criterion(val_outputs, validation_target)
        val_loss += loss.item()
    
    return val_loss 


# Create an Optuna study
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())

# Run optimization
study.optimize(objective, n_trials=50)

# Get the best trial
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

