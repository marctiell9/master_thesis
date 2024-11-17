import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
from TorchDiffEqPack.odesolver import odesolve

class MLP_controller(nn.Module):
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
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.BatchNorm1d(hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)

        '''
        self.norm4 = nn.BatchNorm1d(hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.norm5 = nn.BatchNorm1d(hidden_size)
        self.layer6 = nn.Linear(hidden_size, hidden_size)
        self.norm6 = nn.BatchNorm1d(hidden_size)
        self.layer7 = nn.Linear(hidden_size, hidden_size)
        self.norm7 = nn.BatchNorm1d(hidden_size)
        self.layer8 = nn.Linear(hidden_size, hidden_size)
        self.norm8 = nn.BatchNorm1d(hidden_size)
        self.layer9 = nn.Linear(hidden_size, output_size)
        self._initialize_weights()
        '''
    def _initialize_weights(self):
        
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        '''
        nn.init.xavier_uniform_(self.layer5.weight)
        nn.init.xavier_uniform_(self.layer6.weight)
        nn.init.xavier_uniform_(self.layer7.weight)
        nn.init.xavier_uniform_(self.layer8.weight)
        nn.init.zeros_(self.layer9.weight)
        '''
        
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
        nn.init.zeros_(self.layer4.bias)
        '''
        nn.init.zeros_(self.layer5.bias)
        nn.init.zeros_(self.layer6.bias)
        nn.init.zeros_(self.layer7.bias)
        nn.init.zeros_(self.layer8.bias)
        nn.init.zeros_(self.layer9.bias)
        '''
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
        '''
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
        '''
        return x
    
class HybridModel(nn.Module):
    def __init__(self, model_phy, model_nn, controller, input_meas, state_meas, device, N, bacth_number):
        super().__init__()
        self.model_phy = model_phy
        self.model_nn = model_nn
        self.controller = controller
        self.input_meas = input_meas
        self.state_meas = state_meas
        self.device = device
        self.N = N #orizzonte di predizione
        self.bacth_number = bacth_number
    def forward(self, t, state):
        state = state.to(self.device)
        # t: shape (batch_size,)
        # state: shape (batch_size, state_dim)
        #t_batch = torch.stack([t, self.input_meas[self.N,6]+t ])

        t_batch = create_t_batch(t, self.input_meas, self.N, self.bacth_number)
        # Get batched inputs
        u = self.get_input(t_batch)  # u: shape (batch_size, input_dim)
        # Extract inputs from u
        u=u.unsqueeze(1)
        ail_sp = u[:,0, 0]  # Shape: (batch_size,)
        elev_sp = -u[:,0, 1]
        rud_sp = -u[:, 0, 2]
        flaps_sp = u[:, 0, 3]
        load_meas = u[:, 0, 4]
        length_meas = u[:, 0, 5]
        
        # Get rudder setpoint from the controller
        '''
        self.controller.eval()
        with torch.no_grad():
            rud_sp = -self.controller(state).squeeze(1).to(self.device)  # Shape: (batch_size,)
        '''
        # Compute va, alfa, beta, and v_kite_relative_b in batched fashion
        va, alfa, beta, v_kite_relative_b = self.va_computation(state)

        
        # Physical model prediction
        model_pred = self.model_phy(
            state, ail_sp, elev_sp, rud_sp, flaps_sp,
            load_meas, length_meas, va, alfa, beta, v_kite_relative_b
        )

        # Prepare input for neural network
        state_angles_deg = state[:, 3:6] * 180 / torch.pi
        state_rates_deg = state[:, 6:9] * 180 / torch.pi

        nn_input = torch.cat([
            state[:, 0:3],              # u, v, w
            state_angles_deg,           # phi, theta, psi in degrees
            state_rates_deg,            # P, Q, R in degrees
            state[:, 9:12],             # N, E, h
            ail_sp.unsqueeze(1),
            elev_sp.unsqueeze(1),
            rud_sp.unsqueeze(1),
            flaps_sp.unsqueeze(1),
            load_meas.unsqueeze(1),
            va.unsqueeze(1),
            (alfa * 180 / torch.pi).unsqueeze(1),
            (beta * 180 / torch.pi).unsqueeze(1)
        ], dim=1)

        
        # Neural network prediction
        nn_pred = self.model_nn(nn_input)  # Shape: (batch_size, nn_output_dim)
        pred = model_pred.clone()  # Create a copy of model_pred that requires gradients
        pred[:, 0:3] = pred[:, 0:3] + nn_pred[:, 0:3]  # Correct velocity derivatives
        pred[:, 6:9] = pred[:, 6:9] + nn_pred[:, 3:6]
        return pred  # Shape: (batch_size, state_dim)
    
    def get_input(self, t_batch):
        # t_batch: shape (batch_size,)
        # Ensure t_batch is on the same device and type as input_meas
        #t_batch = t_batch.to(self.input_meas.device).type(self.input_meas.dtype)

        # Perform batched searchsorted
        indices = torch.searchsorted(self.input_meas[:, 6], t_batch, right=True) - 1
        indices = indices.clamp(min=0, max=self.input_meas.shape[0] - 1)  # Ensure indices are valid
        # Fetch inputs corresponding to the indices
        inputs = self.input_meas[indices, :]  # Shape: (batch_size, input_dim)
        return inputs  
    
    def va_computation(self, states):
        # states: shape (batch_size, state_dim)
        batch_size = states.shape[0]
        device = states.device

        # Extract state variables
        phi = states[:, 3]    # Shape: (batch_size,)
        theta = states[:, 4]
        psi = states[:, 5]
        v_kite_b = states[:, 0:3]  # Shape: (batch_size, 3)

        # WIND profile (interpolation based on state height)
        wind_profile_h = torch.tensor([0, 100, 200, 300, 400, 600], dtype=states.dtype, device=device)  # Shape: (6,)
        wind_profile_v = torch.tensor([13, 14.32, 13.82, 10.63, 9.18, 10], dtype=states.dtype, device=device)  # Shape: (6,)

        h = states[:, 11]  # Shape: (batch_size,)

        # Clamp h to the range of wind_profile_h
        h_clamped = h.clamp(min=wind_profile_h[0], max=wind_profile_h[-1])

        # Perform linear interpolation to compute vw for each sample in the batch
        indices = torch.searchsorted(wind_profile_h, h_clamped, right=True)
        indices = indices.clamp(max=wind_profile_h.size(0) - 1)
        i_left = (indices - 1).clamp(min=0)
        i_right = indices

        x0 = wind_profile_h[i_left]  # Shape: (batch_size,)
        x1 = wind_profile_h[i_right]
        y0 = wind_profile_v[i_left]
        y1 = wind_profile_v[i_right]

        dx = x1 - x0
        dx[dx == 0] = 1e-6  # Prevent division by zero

        frac = (h_clamped - x0) / dx
        vw = y0 + frac * (y1 - y0)  # Shape: (batch_size,)

        # Wind direction
        wind_dir = torch.tensor(-54.5939, dtype=states.dtype, device=device)

        # Compute wind vector in NED frame
        wind_dir_rad = torch.deg2rad(wind_dir)
        cos_wind_dir = torch.cos(wind_dir_rad)
        sin_wind_dir = torch.sin(wind_dir_rad)

        v_w_n = torch.stack([
            -vw * cos_wind_dir,
            -vw * sin_wind_dir,
            torch.zeros_like(vw)
        ], dim=1)  # Shape: (batch_size, 3)

        # Compute trigonometric functions
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_psi = torch.sin(psi)
        cos_psi = torch.cos(psi)

        # Compute Direction Cosine Matrix (DCM) for each sample
        DCM = torch.zeros((batch_size, 3, 3), dtype=states.dtype, device=device)

        # First row
        DCM[:, 0, 0] = cos_theta * cos_psi
        DCM[:, 0, 1] = cos_theta * sin_psi
        DCM[:, 0, 2] = -sin_theta

        # Second row
        DCM[:, 1, 0] = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
        DCM[:, 1, 1] = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
        DCM[:, 1, 2] = sin_phi * cos_theta

        # Third row
        DCM[:, 2, 0] = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
        DCM[:, 2, 1] = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
        DCM[:, 2, 2] = cos_phi * cos_theta

        # Compute v_kite_n
        v_kite_b_expanded = v_kite_b.unsqueeze(2)  # Shape: (batch_size, 3, 1)
        DCM_T = DCM.transpose(1, 2)
        v_kite_n = torch.bmm(DCM_T, v_kite_b_expanded).squeeze(2)  # Shape: (batch_size, 3)

        # Compute v_kite_relative_n
        v_kite_relative_n = v_kite_n - v_w_n  # Shape: (batch_size, 3)

        # Compute v_kite_relative_b
        v_kite_relative_n_expanded = v_kite_relative_n.unsqueeze(2)
        v_kite_relative_b = torch.bmm(DCM, v_kite_relative_n_expanded).squeeze(2).to(self.device)  # Shape: (batch_size, 3)

        # Compute va
        va = torch.norm(v_kite_relative_b, dim=1).to(self.device)  # Shape: (batch_size,)
        va = va + (va == 0).float() * torch.finfo(states.dtype).eps  # Prevent division by zero

        # Compute angle of attack (alfa) and sideslip angle (beta)
        alfa = torch.atan2(v_kite_relative_b[:, 2], v_kite_relative_b[:, 0]).to(self.device)  # Shape: (batch_size,)
        beta = torch.asin(v_kite_relative_b[:, 1] / va).to(self.device)

        return va, alfa, beta, v_kite_relative_b

    def loss_nn_compute(self):
        #gli stati misurati sono in radianti
        va, alfa, beta, v_kite_relative_b= self.va_computation(self.state_meas)
        nn_pred = self.model_nn(torch.stack([
            self.state_meas[:,0],
            self.state_meas[:,1],  
            self.state_meas[:,2],  
            self.state_meas[:,3]*180 / torch.pi,  
            self.state_meas[:,4]*180 / torch.pi,  
            self.state_meas[:,5]*180 / torch.pi,  
            self.state_meas[:,6]*180 / torch.pi,            
            self.state_meas[:,7]*180 / torch.pi,  
            self.state_meas[:,8]*180 / torch.pi,  
            self.state_meas[:,9],  
            self.state_meas[:,10],  
            self.state_meas[:,11],  
            self.input_meas[:,0],
            -1*self.input_meas[:,1],
            -1*self.input_meas[:,2],
            self.input_meas[:,3],
            self.input_meas[:,4],
            va,
            (alfa * 180 / torch.pi),
            (beta * 180 / torch.pi)]).T)
        loss_nn = (torch.norm(nn_pred, dim=1) ** 2) .mean()
        return loss_nn
    
def h_poly(t):
    # t: shape (batch_size,)
    batch_size = t.shape[0]
    device = t.device
    dtype = t.dtype

    # Compute tt: shape (batch_size, 4)
    exponents = torch.arange(4, device=device, dtype=dtype).unsqueeze(0)  # Shape: (1, 4)
    t_expanded = t.unsqueeze(1)  # Shape: (batch_size, 1)
    tt = t_expanded ** exponents  # Shape: (batch_size, 4)

    # Transpose tt to match original dimensions (if necessary)
    tt = tt.transpose(0, 1)  # Shape: (4, batch_size)

    # Define A matrix
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=dtype, device=device)  # Shape: (4, 4)

    # Compute hh: Shape: (4, batch_size)
    hh = torch.matmul(A, tt)  # Matrix multiplication

    # Transpose hh to get shape (batch_size, 4)
    hh = hh.transpose(0, 1)  # Shape: (batch_size, 4)

    return hh



def interp(x, y, xs):
    # x: shape (n_points,)
    # y: shape (n_points,)
    # xs: shape (batch_size,)

    device = xs.device
    dtype = xs.dtype
    n_points = x.size(0)
    batch_size = xs.size(0)

    # Compute slopes m between points
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])  # Shape: (n_points - 1,)
    m = torch.cat([
        m[[0]],                        # First slope
        (m[1:] + m[:-1]) / 2,          # Average of adjacent slopes
        m[[-1]]                        # Last slope
    ])  # Shape: (n_points,)

    # Search for indices
    idxs = torch.searchsorted(x[1:], xs)
    idxs = torch.clamp(idxs, 0, n_points - 2)  # Ensure indices are valid

    # Compute dx
    x_idx = x[idxs]          # Shape: (batch_size,)
    x_idx1 = x[idxs + 1]     # Shape: (batch_size,)
    dx = x_idx1 - x_idx      # Shape: (batch_size,)

    # Avoid division by zero
    dx_nonzero = dx.clone()
    dx_nonzero[dx_nonzero == 0] = 1e-6

    # Compute normalized t for h_poly
    t = (xs - x_idx) / dx_nonzero  # Shape: (batch_size,)

    # Compute hh using h_poly
    hh = h_poly(t)  # Shape: (batch_size, 4)

    # Get y and m at idxs and idxs + 1
    y_idx = y[idxs]          # Shape: (batch_size,)
    y_idx1 = y[idxs + 1]     # Shape: (batch_size,)
    m_idx = m[idxs]
    m_idx1 = m[idxs + 1]

    # Reshape for broadcasting
    y_idx = y_idx.unsqueeze(1)        # Shape: (batch_size, 1)
    y_idx1 = y_idx1.unsqueeze(1)
    m_idx = m_idx.unsqueeze(1)
    m_idx1 = m_idx1.unsqueeze(1)
    dx = dx.unsqueeze(1)              # Shape: (batch_size, 1)

    # Compute interpolated values
    interpolated = (
        hh[:, 0].unsqueeze(1) * y_idx +
        hh[:, 1].unsqueeze(1) * m_idx * dx +
        hh[:, 2].unsqueeze(1) * y_idx1 +
        hh[:, 3].unsqueeze(1) * m_idx1 * dx
    )  # Shape: (batch_size, 1)

    return interpolated.squeeze(1)  # Shape: (batch_size,)


def create_t_batch(t, input_meas, N, num_batches):
    t_list = []
    for i in range(num_batches):
        index = i * N
        offset = input_meas[index, 6]  # in 6 ho time_id
        t_shifted = t + offset
        t_list.append(t_shifted)
    
    # Stack the list of time vectors along a new dimension
    t_batch = torch.stack(t_list, dim=0)
    return t_batch




