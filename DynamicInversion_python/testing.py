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

device="cpu"

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

input_train=torch.stack([N_meas_train, E_meas_train, h_meas_train, u_meas_train, v_meas_train, w_meas_train,
                         phi_meas_train, theta_meas_train, psi_meas_train,
                         p_meas_train, q_meas_train, r_meas_train,
                        ail_sp_train, elev_sp_train, flaps_sp_train]).squeeze(-1).T
input_train=torch.stack([N_meas_train, E_meas_train, h_meas_train, u_meas_train, v_meas_train, w_meas_train,
                         phi_meas_train, theta_meas_train, psi_meas_train,
                         p_meas_train, q_meas_train, r_meas_train]).squeeze(-1).T
target_train = rud_sp_train

#validation
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
                                phi_meas_val, theta_meas_val, psi_meas_val,
                                p_meas_val, q_meas_val, r_meas_val]).squeeze(-1).T
validation_target = rud_sp_val

#inizializzazione rete neurale
input_size=12  #stati + aileron, elevator and flaps (discrete control surfaces that are constant)
hidden_size=32
output_size=1 # the output is the rudder only
dynamic_inversion=MLP(input_size, hidden_size, output_size).to(device)
loss = nn.MSELoss()
dynamic_inversion.load_state_dict(torch.load('dynamic_inversion.pth', weights_only=True))
#validation
dynamic_inversion.eval()
with torch.no_grad():
    rud_sim = dynamic_inversion(validation_inputs) #sto usando i validation input normalizzati
val_loss = loss(rud_sim, validation_target)
print("validation loss", val_loss)
time = np.arange(0, len(rud_sp_val) * 0.045, 0.045)
plt.figure()
plt.plot(time, rud_sim  , label='prediction')
plt.plot(time, rud_sp_val, label='measure')
plt.legend()
plt.title("VALIDATION SET")
plt.show()

#Test set
ail_sp_test = ail_sp[validation_index:]
elev_sp_test = elev_sp[validation_index:]
rud_sp_test = rud_sp[validation_index:]
flaps_sp_test = flaps_sp[validation_index:]
load_meas_test = load_meas[validation_index:]
length_meas_test = length_meas[validation_index:]
time_id_test = time_vector[validation_index:]

N_meas_test = N_meas[validation_index:]
E_meas_test = E_meas[validation_index:]
h_meas_test = h_meas[validation_index:]
u_meas_test = u_meas[validation_index:]
v_meas_test = v_meas[validation_index:]
w_meas_test = w_meas[validation_index:]
phi_meas_test = phi_meas[validation_index:]
theta_meas_test = theta_meas[validation_index:]
psi_meas_test = psi_meas[validation_index:]
p_meas_test = p_meas[validation_index:]
q_meas_test = q_meas[validation_index:]
r_meas_test = r_meas[validation_index:]

test_inputs = torch.stack([N_meas_test, E_meas_test, h_meas_test, u_meas_test, v_meas_test, w_meas_test,
                                phi_meas_test, theta_meas_test, psi_meas_test,
                                p_meas_test, q_meas_test, r_meas_test]).squeeze(-1).T
test_target = rud_sp_test

dynamic_inversion.eval()
with torch.no_grad():  
    rud_sim = dynamic_inversion(test_inputs) #sto usando i validation input normalizzati
test_loss = loss(rud_sim, test_target)
print("test loss", test_loss)
time = np.arange(0, len(rud_sp_test) * 0.045, 0.045)
plt.figure()
plt.plot(time, rud_sim   , label='prediction')
plt.plot(time, rud_sp_test, label='measure')
plt.legend()
plt.title("TEST SET")
plt.show()