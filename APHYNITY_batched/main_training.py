import torch.utils
from HybridModel import* 
from scipy.io import loadmat
from KM1 import KM1kite
import time
import os

device='cpu'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the DataSet
mat_data = loadmat('NODEdataset_filtered.mat')

ail_sp = torch.tensor(mat_data["ail_filtered"],dtype=torch.float32).to(device)
elev_sp = torch.tensor(mat_data["elev_filtered"],dtype=torch.float32).to(device)
rud_sp = torch.tensor(mat_data["rud_filtered"],dtype=torch.float32).to(device)
flaps_sp = torch.tensor(mat_data["flaps_filtered"],dtype=torch.float32).to(device)
load_meas = torch.tensor(mat_data["load_meas"],dtype=torch.float32).to(device)
length_meas = torch.tensor(mat_data["length_meas"],dtype=torch.float32).to(device)
time_vector = torch.tensor(mat_data["time_vector"],dtype=torch.float32).to(device) #ho preso tutto il data set

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
 #numero di sample training dataset, 4200
batch_number = 1400 #number of batches
N = 50 #prediction horizon
identification_index=N*batch_number
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

input_meas=torch.stack([ail_sp, elev_sp, rud_sp, flaps_sp, load_meas, length_meas, time_id]).squeeze(-1).T.to(device)
state_meas=torch.stack([ u_meas, v_meas, w_meas ,
                         phi_meas* torch.pi / 180, theta_meas* torch.pi / 180, psi_meas* torch.pi / 180,
                           p_meas* torch.pi / 180, q_meas* torch.pi / 180, r_meas* torch.pi / 180,
                             N_meas, E_meas, h_meas,]).squeeze(-1).T.to(device)

#controller inizializzazione
input_size_controller=12  #stati 
hidden_size_controller=32
output_size_controller=1 # the output is the rudder only
dynamic_inversion=MLP_controller(input_size_controller, hidden_size_controller, output_size_controller).to(device)
dynamic_inversion.load_state_dict(torch.load('dynamic_inversion.pth', weights_only=True))

physical_params = nn.ParameterDict({
    'CYp': nn.Parameter(torch.tensor(-0.05433,dtype=torch.float32)),
    'Clp': nn.Parameter(torch.tensor(-0.63669,dtype=torch.float32)),
    'Cnp': nn.Parameter(torch.tensor(-0.18801,dtype=torch.float32)),
    'CLq': nn.Parameter(torch.tensor(8.18191,dtype=torch.float32)),
    'Cmq': nn.Parameter(torch.tensor(-18.65100,dtype=torch.float32)),
    'CYr': nn.Parameter(torch.tensor(0.21502,dtype=torch.float32)),
    'Clr': nn.Parameter(torch.tensor(0.31084,dtype=torch.float32)),
    'Cnr': nn.Parameter(torch.tensor(-0.04912,dtype=torch.float32)),
    'T_cg_x': nn.Parameter(torch.tensor(6/1000,dtype=torch.float32)),
    'T_cg_z': nn.Parameter(torch.tensor(145/1000,dtype=torch.float32)),
    'Ixx': nn.Parameter(torch.tensor(33.69508442,dtype=torch.float32)),
    'Iyy': nn.Parameter(torch.tensor(23.75486667,dtype=torch.float32)),
    'Izz': nn.Parameter(torch.tensor(66.33728998,dtype=torch.float32)),
    'Ixz': nn.Parameter(torch.tensor(3.0,dtype=torch.float32)),
    'Cdt': nn.Parameter(torch.tensor(1.0,dtype=torch.float32)),
    'CD_0': nn.Parameter(torch.tensor(0.060140723114519,dtype=torch.float32)),
    'CD_alfa': nn.Parameter(torch.tensor(0.002522363025499,dtype=torch.float32)),
    'CL_0': nn.Parameter(torch.tensor(0.731242284649101,dtype=torch.float32)),
    'CL_alfa': nn.Parameter(torch.tensor(0.098245353054389,dtype=torch.float32)),
    'Cm_0': nn.Parameter(torch.tensor(0.011962094587529,dtype=torch.float32)),
    'Cm_alfa': nn.Parameter(torch.tensor(-0.010259723614798,dtype=torch.float32)),
    'Cn_rud': nn.Parameter(torch.tensor(-0.001137018696165,dtype=torch.float32)),
}).to(device)

#inzializzazione hybrid model
input_size=20 #stati + control input +alpha, beta, va, tether load
hidden_size= 256
output_size=6 #stati
NN=MLP(input_size, hidden_size, output_size).to(device) #Neural Network initialization
kite=KM1kite(physical_params, device, input_meas).to(device) #kite passa direttamente i parametri fisici all hybrid model
hybrid_model=HybridModel(kite, NN, dynamic_inversion, input_meas, state_meas, device, N, batch_number)
#optimizer = torch.optim.RMSprop(hybrid_model.parameters(), lr=0.001, alpha=0.9)
optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.0001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7) #ogni step_size epoch il learning rate viene ridotto di gamma
#optimizer = torch.optim.LBFGS(hybrid_model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
#creazione batch stati iniziali 
initial_state_list = []
for i in range(batch_number):
    index = i * N
    initial_state_sample = torch.tensor([u_meas[index],v_meas[index],w_meas[index],phi_meas[index]*torch.pi/180 ,
            theta_meas[index]*torch.pi/180 ,psi_meas[index]*torch.pi/180,
            p_meas[index]*torch.pi/180,q_meas[index]*torch.pi/180,r_meas[index]*torch.pi/180,
            N_meas[index],E_meas[index],h_meas[index]],dtype=torch.float32, requires_grad=True).to(device)
    initial_state_list.append(initial_state_sample)
    initial_state_batch = torch.stack(initial_state_list, dim=0) #shape (10,12)

# w scaling computaion 
'''
max = torch.tensor([60.4487,   28.3286,   29.1759,
                      101.9335*torch.pi/180,   55.6728*torch.pi/180,  179.9200*torch.pi/180 ,
                          30.7203*torch.pi/180,  267.4908*torch.pi/180 , 261.1460*torch.pi/180,
                        -177.4984 , 551.5260,  323.3980,])
min = torch.tensor([-12.5718,  -39.4945,  -24.2744,
                      -93.4235*torch.pi/180,  -79.1208*torch.pi/180, -179.9450*torch.pi/180,
                          -72.2445*torch.pi/180,  -69.2943*torch.pi/180, -266.0327*torch.pi/180,
                            -413.4147,  205.8565,  157.6271])
w = 1.0 / (max - min)
w = w.view(1, 1, -1)
'''
# definizione con massimi presi sui valori assoluti
# w calcolato su dataset con 70000 dati
#w = torch.tensor([0.0165429456221055,	0.0253199629725467,	0.0342748907638177,	0.00981031751092624,	0.0126389095148871,	0.00555802578923966,	0.0138418841572715,	0.0138508711332263,	0.00382927518487792,	0.00241887886577157,	0.00181315114790599,	0.00309216507214021])

w = torch.ones(1,12)
state_meas = state_meas.view(batch_number, N, 12).permute(1, 0, 2) #rendiamo il vettore delle misure conforme al vettore delle predizioni, sono stati trasformati in radianti
time_id=time_id.view(-1).to(device)

epochs=3000
eta2 = 0.0001 #nel paper lo metteva a 1000, eta2 definisce quanto velocemente il peso della MSE LOSS diventa più importante della norma della predizione della rete neurale
eta1 = 1
for epoch in range(epochs):
    start_time = time.time()
    y_sim = odeint(hybrid_model, initial_state_batch, time_id[0:N], method='rk4', options={'step_size': 0.045})
    #y_sim = odeint(hybrid_model, initial_state_batch, time_id[0:N], method='dopri5')
    loss = eta1*(((y_sim - state_meas) * w) **2).mean()
    nn_regularization = hybrid_model.loss_nn_compute()
    #loss += nn_regularization
    print("Norma rete neurale", nn_regularization)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=0.1) #gradient clipping
    #for param in NN.parameters():
    #        print(param.grad, "GRADIENTS NN")
    optimizer.step()
    '''
    #eta1 update
    
    with torch.no_grad():
      y_sim = odeint(hybrid_model, initial_state_batch, time_id[0:N], method='rk4', options={'step_size': 0.045}) #ricalcolo la predizione della traiettoria dopo aver aggiornato i parametri
      loss_eta1_update = (((y_sim - state_meas) * w) **2).mean()*N*batch_number
      eta1 = eta1 + eta2*loss_eta1_update
      print(eta1, "ETA1")
    '''

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds", "LOSS", loss, "EPOCH", epoch)
    # Save the model every 100 epochs
    if (epoch + 1) % 100 == 0:
        save_path = os.path.join('model_folder', f"hybrid_model_epoch_{epoch + 1}.pth")
        torch.save(hybrid_model.state_dict(), save_path)
        print(f"Model saved at {save_path}")
torch.save(hybrid_model.state_dict(), 'hybrid_model.pth') # salvo anche il modello finale

