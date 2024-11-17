from scipy.io import loadmat
from feedforward_nn import*
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#device="cpu" # with gpu is faster
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

#Identification data set 60%=66666 (index)
identification_index=66666
#identification_index=114000 #80% sbagliato
ail_sp = ail_sp[0:identification_index]
elev_sp = elev_sp[0:identification_index]
rud_sp = rud_sp[0:identification_index]
flaps_sp = flaps_sp[0:identification_index]
load_meas = load_meas[0:identification_index]
length_meas = length_meas[0:identification_index]
time_id = time_vector[0:identification_index]

N_meas = N_meas[0:identification_index]
E_meas = E_meas[0:identification_index]
h_meas = h_meas[0:identification_index]
u_meas = u_meas[0:identification_index]
v_meas = v_meas[0:identification_index]
w_meas = w_meas[0:identification_index]
phi_meas = phi_meas[0:identification_index]
theta_meas = theta_meas[0:identification_index]
psi_meas = psi_meas[0:identification_index]
p_meas = p_meas[0:identification_index]
q_meas = q_meas[0:identification_index]
r_meas = r_meas[0:identification_index]

#metto tutti gli angoli degli stati e delle superfici di controllo in gradi durante il training
input_train=torch.stack([N_meas, E_meas, h_meas, u_meas, v_meas, w_meas,
                         phi_meas, theta_meas, psi_meas,
                         p_meas, q_meas, r_meas,
                        ail_sp, elev_sp, flaps_sp]).squeeze(-1).T

input_train=torch.stack([N_meas, E_meas, h_meas, u_meas, v_meas, w_meas,
                         phi_meas, theta_meas, psi_meas,
                         p_meas, q_meas, r_meas,]).squeeze(-1).T #senza input di controllo
target_train = rud_sp

#inizializzazione rete neurale
input_size=12  #stati + aileron, elevator and flaps (discrete control surfaces that are constant)
hidden_size=32
output_size=1 # the output is the rudder only
dynamic_inversion=MLP(input_size, hidden_size, output_size).to(device) #Neural Network initialization

optimizer = torch.optim.Adam(dynamic_inversion.parameters(), lr=0.0001) # weight_decay=1e-4)
loss_validation = nn.MSELoss()
batch_size = 256
#dataset = TensorDataset(input_normalized, target_normalized)
dataset = TensorDataset(input_train, target_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#training loop Physics_informed
loss_history = []
num_epochs = 100
num_cycles = 10
lambda_max = 10  # Maximum value for the local monotonicity loss weight
# Assume you have your training data loader ready
for epoch in range(num_epochs):
    dynamic_inversion.train()  
    total_loss = 0.0  
    lambda_lm = cyclical_annealing_scheduler(epoch, num_epochs, lambda_max, num_cycles, 0.5)

    for batch_idx, (batch_input, batch_target) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        pred = dynamic_inversion(batch_input)
        #r = batch_input[:, 11]
        rates = batch_input[:,11] #caso in cui consideriamo p,q,r per la monotonicity loss, indice 12 non incluso
        #print(rates.shape) #dovrebbero essere vettori di 256x3
        loss = physics_informed_loss(pred, batch_target, rates, 0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_history.append(total_loss)
    print("LOSS VALUE", total_loss, "EPOCH", epoch)
    #ad ogni epoch potrei mettere anche il valore della validazione
    torch.save(dynamic_inversion.state_dict(), 'dynamic_inversion.pth')
    #torch.save(dynamic_inversion.state_dict(), 'dynamic_inversion.pt')
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()
