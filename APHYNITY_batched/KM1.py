import torch
import torch.nn as nn
from scipy.interpolate import interp1d

class KM1kite(nn.Module):

    def __init__(self, physical_params, device, input_meas):
        super().__init__()
        self.device = device
        self.input_meas = input_meas
        self.physical_params = physical_params

        Ixx = self.physical_params['Ixx']
        Iyy = self.physical_params['Iyy']
        Izz = self.physical_params['Izz']
        Ixz = self.physical_params['Ixz']

        #costanti che posso calcolare prima
        self.m = 54.00  # Mass of the kite (kg)
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
    
        # Precompute other constants
        self.S_w = 2.982  # Wing area (m^2)
        self.rho = 1.225  # Air density (kg/m^3)
        self.span = 7.4   # Wingspan (m)
        self.MAC = 0.431  # Mean Aerodynamic Chord (m)
        self.Weight_NED = torch.tensor([0.0, 0.0, self.m * self.g], dtype=torch.float32).to(device)  

    def forward(self, state, ail_sp, elev_sp, rud_sp, flaps_sp, load_meas, length_meas, va, alfa, beta, v_kite_relative_b ):
        states_dot = self.KM1_kite(state, ail_sp, elev_sp, rud_sp, flaps_sp, load_meas, length_meas, va, alfa, beta, v_kite_relative_b)
        return states_dot
        

    def KM1_kite(self, states, ail_sp, elev_sp, rud_sp, flaps_sp, load_meas, length_meas, va, alfa, beta, v_kite_relative_b):
        u = states[:, 0]
        v = states[:, 1]
        w = states[:, 2]
        phi = states[:, 3]
        theta = states[:, 4]
        psi = states[:, 5]
        P = states[:, 6]
        Q = states[:, 7]
        R = states[:, 8]
        N = states[:, 9]
        E = states[:, 10]
        h = states[:, 11]

        load_meas = load_meas.unsqueeze(1)
        length_meas = length_meas.unsqueeze(1)
        
        v_kite_b = torch.stack([u, v, w], dim=1)

        diameter = 3.5e-3  # scalar
        rho = 1.2  # scalar
        effective_length = length_meas / 2  # length_meas shape: (batch_size,)
        Ad = torch.pi * diameter / 2 * effective_length  # Shape: (batch_size,)

        v_kite_relative_b_norm = torch.norm(v_kite_relative_b, dim=1).squeeze(0).to(self.device)  # Shape: (batch_size,)
        tether_drag = 0.5 *self.physical_params['Cdt'] * Ad.squeeze(1) * rho * (v_kite_relative_b_norm**2) # Shape: (batch_size,)
        theta_t_rad = torch.atan2(-tether_drag.unsqueeze(1), load_meas.squeeze(0)).squeeze(1)  # Shape: (batch_size,)
        # Forces and moments due to the tether (in body frame)
        NEh = torch.stack([N, E, -h], dim=1).to(self.device)  # Shape: (batch_size, 3)
        NEh_norm = torch.norm(NEh, dim=1, keepdim=True) # Shape: (batch_size, 1)
        Ft_N = (-NEh / NEh_norm * load_meas).unsqueeze(2) # Shape: (batch_size, 3)
        
        # Precompute trigonometric functions
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_psi = torch.sin(psi)
        cos_psi = torch.cos(psi)
        DCM = self.compute_DCM(sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi)

        F_t_b = torch.bmm(DCM, Ft_N).squeeze(2)  # Shape: (batch_size, 3)
       
        # Rotation matrix for tether
        cos_theta_t = torch.cos(theta_t_rad)
        sin_theta_t = torch.sin(theta_t_rad)
        rotation_matrix = self.compute_rotation_matrix(cos_theta_t, sin_theta_t)
        # rotation_matrix shape: (batch_size, 3, 3)
        # Apply rotation matrix to F_t_b
        F_t_b = torch.bmm(rotation_matrix, F_t_b.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, 3)
        # Compute moments due to tether
        T_cg = torch.stack([self.physical_params['T_cg_x'], torch.tensor(0.0, dtype=torch.float32), self.physical_params['T_cg_z']])
        M_t_b = torch.cross(T_cg.unsqueeze(0).expand(F_t_b.shape[0], -1), F_t_b, dim=1)  # Shape: (batch_size, 3)

        ##################################################CALCOLO COEFFICIENTI AERODINAMICI
        #La definisco prima di portare alfa e beta in gradi
        sin_alfa = torch.sin(alfa)
        cos_alfa = torch.cos(alfa)
        sin_beta = torch.sin(beta)
        cos_beta = torch.cos(beta)

        zero_tensor = torch.zeros_like(sin_beta)  

        Wind2Body = torch.stack([
    torch.stack([cos_beta * cos_alfa, -sin_beta * cos_alfa, -sin_alfa], dim=-1),
    torch.stack([sin_beta, cos_beta, zero_tensor], dim=-1),
    torch.stack([cos_beta * sin_alfa, -sin_beta * sin_alfa, cos_alfa], dim=-1)
], dim=-2)

        alfa = alfa*180/torch.pi
        beta = beta*180/torch.pi;# positive when blow from right

        #matrice con coefficienti AERODINAMICI
        alfa2 = alfa**2
        alfa3 = alfa**3
        alfa4 = alfa**4
        beta_abs = torch.abs(beta)
        beta2 = beta**2
        beta3 = beta**3
        alfa_beta = alfa * beta
        flap2 = flaps_sp**2
        alfa_flap = alfa * flaps_sp
        alfa2_flap = alfa2 * flaps_sp
        alfa3_flap = alfa3 * flaps_sp
        flap_ail_abs = torch.abs(ail_sp) * flaps_sp
        ail2 = ail_sp**2
        ail_abs = torch.abs(ail_sp)
        ail_alfa = ail_sp * alfa
        ail_alfa2 = ail_sp * alfa2
        ail_alfa3 = ail_sp * alfa3
        elev2 = elev_sp**2
        elev_alfa = elev_sp * alfa
        rud_beta = rud_sp * beta
        rud_beta2 = rud_sp * beta2

        batch_size = alfa.shape[0]  # Assuming alfa, beta, etc., are batched

        # Create batched versions of constants in col
        ones = torch.ones(batch_size, dtype=torch.float32).to(self.device)

        # Stack each entry in col along the batch dimension
        col = torch.stack([
            ones, alfa, alfa2, alfa3, alfa4, beta, beta_abs, beta2, beta3,
            alfa_beta, flaps_sp, flap2, alfa_flap, alfa2_flap, alfa3_flap,
            flap_ail_abs, ail_sp, ail2, ail_abs, ail_alfa, ail_alfa2,
            ail_alfa3, elev_sp, elev2, elev_alfa, rud_sp, rud_beta, rud_beta2
        ], dim=1)
        
        Aero_Mat = torch.stack([
            torch.stack([
                self.physical_params['CD_0'],
                torch.tensor(0.0),
                self.physical_params['CL_0'],
                torch.tensor(0.000564472365244),
                self.physical_params['Cm_0'],
                torch.tensor(0.0)
            ]),
            torch.stack([
                self.physical_params['CD_alfa'],
                torch.tensor(-0.000114407083448),
                self.physical_params['CL_alfa'],
                torch.tensor(-0.000112565466713),
                self.physical_params['Cm_alfa'],
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.000633096909836),
                torch.tensor(0.0),
                torch.tensor(-0.001172087843517),
                torch.tensor(-0.000008559892380),
                torch.tensor(-0.000215542230679),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-0.000155525989384),
                torch.tensor(0.000001334492880),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.000003673362687),
                torch.tensor(-0.000000030912852),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(-0.004097380801405),
                torch.tensor(0.0),
                torch.tensor(-0.000788952517328),
                torch.tensor(0.0),
                torch.tensor(0.001303408142587)
            ]),
            torch.stack([
                torch.tensor(-0.000224091824682),
                torch.tensor(0.0),
                torch.tensor(-0.000614550894367),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.000061819395501),
                torch.tensor(-0.000003199214929),
                torch.tensor(-0.000237847677679),
                torch.tensor(0.000004680727100),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.000000316778457),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.000118247415372),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.002326616033398),
                torch.tensor(0.0),
                torch.tensor(0.041084815618762),
                torch.tensor(0.0),
                torch.tensor(0.001205810996722),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.000076542367624),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.000032572976850),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-0.000117456086622),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-0.000129584415794),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.000002602117206),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.005541490333061),
                torch.tensor(0.0),
                torch.tensor(-0.000318079447660)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.000028884716251),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-0.006981445469420),
                torch.tensor(0.0),
                torch.tensor(-0.000013246875501),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-0.000015135734314),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-0.000019063405150),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.000000439095862),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.000001545451775),
                torch.tensor(0.0),
                torch.tensor(0.004563921758182),
                torch.tensor(0.0),
                torch.tensor(-0.018973755153161),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.000017189016131),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.000010527743305),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.003225382114448),
                torch.tensor(0.0),
                torch.tensor(0.000162800776073),
                torch.tensor(0.0),
                self.physical_params['Cn_rud']
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-0.000008413169249),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
            torch.stack([
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-0.000000883141625),
                torch.tensor(0.0),
                torch.tensor(0.0)
            ]),
        ])  # Shape: (28, 6)
        # Repeat Aero_Mat for each batch
        Aero_Mat_batched = Aero_Mat.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, 28, 6)
        col = (col.unsqueeze(2)) # Shape: (batch_size, 28, 1)
        # Perform batch matrix multiplication
        Coef = torch.bmm(Aero_Mat_batched.transpose(1, 2), col).squeeze(2)
        qbar=.5*rho*(va**2)
        CD = Coef[:, 0]  # Shape: (batch_size,)
        CY = Coef[:, 1] + (self.span / 2) / va * (self.physical_params['CYp'] * states[:, 6] + self.physical_params['CYr'] * states[:, 8])  # Shape: (batch_size,)
        CL = Coef[:, 2] + (self.MAC / 2) / va * self.physical_params['CLq'] * states[:, 7]
        Cl = Coef[:, 3] + (self.span / 2) / va * (self.physical_params['Clp'] * states[:, 6] + self.physical_params['Clr'] * states[:, 8])
        Cm = Coef[:, 4] + (self.MAC / 2) / va * self.physical_params['Cmq'] * states[:, 7]
        Cn = Coef[:, 5] + (self.span / 2) / va * (self.physical_params['Cnp'] * states[:, 6] + self.physical_params['Cnr'] * states[:, 8])

        D = qbar * self.S_w * CD
        L = qbar * self.S_w * CL
        Y = qbar * self.S_w * CY

        l = qbar * self.S_w * self.span * Cl
        m = qbar * self.S_w * self.MAC * Cm
        n = qbar * self.S_w * self.span * Cn

        # Stack aerodynamic forces to prepare for body frame transformation
        # Shape: (batch_size, 3, 1)
        aero_forces = torch.stack([-D, Y, -L], dim=1).unsqueeze(2)

        # Compute aerodynamic forces in the body frame
        # aero_forces shape: (batch_size, 3, 1)
        F_a_b = torch.bmm(Wind2Body, aero_forces).squeeze(2)  # Result shape: (batch_size, 3)

        # Stack aerodynamic moments for body frame
        # Shape: (batch_size, 3, 1)
        M_a_b = torch.stack([l, m, n], dim=1).unsqueeze(2)

        # Compute total forces and moments in the body frame
        # Ensure F_t_b and M_t_b have shapes (batch_size, 3)
        F_b = F_a_b + F_t_b  # Shape: (batch_size, 3)
        M_b = M_a_b.squeeze(2) + M_t_b  # Shape: (batch_size, 3)
        
        # Compute state derivatives (6-DOF equations)
        states_dot = self.DOF6(F_b, M_b, states, DCM)
        return states_dot

    def DOF6(self, F_b, M_b, states, DCM):
        # Extract state variables
        u = states[:, 0]
        v = states[:, 1]
        w = states[:, 2]
        phi = states[:, 3]
        theta = states[:, 4]
        psi = states[:, 5]
        P = states[:, 6]
        Q = states[:, 7]
        R = states[:, 8]

        # Compute Euler angle derivatives
        phi_dot = P + Q * torch.sin(phi) * torch.tan(theta) + R * torch.cos(phi) * torch.tan(theta)
        theta_dot = Q * torch.cos(phi) - R * torch.sin(phi)
        psi_dot = (Q * torch.sin(phi) + R * torch.cos(phi)) / torch.cos(theta)
        # Compute body forces
        Weight_NED_expanded = self.Weight_NED.unsqueeze(0).expand(states.shape[0], -1)  # Shape: (batch_size, 3)

        Weight_b = torch.bmm(DCM, Weight_NED_expanded.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, 3)
        v_b = torch.stack([u, v, w], dim=1)  # Shape: (batch_size, 3)
        omega_b = torch.stack([P, Q, R], dim=1)  # Shape: (batch_size, 3)

        # Compute v_b_dot
        cross_prod = torch.cross(omega_b, v_b, dim=1)  # Shape: (batch_size, 3)
        v_b_dot = (F_b + Weight_b) / self.m - cross_prod  # Shape: (batch_size, 3)

        # Moment of inertia matrix I (constant)
        I = torch.stack([ torch.stack([self.physical_params['Ixx'], torch.tensor(0), -1*self.physical_params['Ixz']])
                       , torch.stack([torch.tensor(0), self.physical_params['Iyy'],torch.tensor(0)]), 
                       torch.stack([-1*self.physical_params['Ixz'], torch.tensor(0), self.physical_params['Izz']])])
        # Expand I to match batch size
        I_expanded = I.unsqueeze(0).expand(states.shape[0], -1, -1)  # Shape: (batch_size, 3, 3)
        # Compute omega_dot
        Iomega = torch.bmm(I_expanded, omega_b.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, 3)
        omega_cross_Iomega = torch.cross(omega_b, Iomega, dim=1)  # Shape: (batch_size, 3)
        M_minus_cross = M_b - omega_cross_Iomega  # Shape: (batch_size, 3)
        # Solve I * omega_dot = M_b - omega_cross_Iomega for omega_dot
        omega_dot = torch.linalg.solve(I_expanded, M_minus_cross.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, 3)

        # Navigation equations
        DCM_transpose = DCM.transpose(1, 2)  # Shape: (batch_size, 3, 3)
        xyhdot = torch.bmm(DCM_transpose, v_b.unsqueeze(2)).squeeze(2) * torch.tensor([1.0, 1.0, -1.0]).to(self.device)  # Shape: (batch_size, 3)

        # Stack all derivatives
        states_dot = torch.cat([
            v_b_dot,  # Shape: (batch_size, 3)
            phi_dot.unsqueeze(1),  # Shape: (batch_size, 1)
            theta_dot.unsqueeze(1),
            psi_dot.unsqueeze(1),
            omega_dot,  # Shape: (batch_size, 3)
            xyhdot  # Shape: (batch_size, 3)
        ], dim=1)  # Shape: (batch_size, state_dim)
        return states_dot
    
    def compute_DCM(self, sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi):
        batch_size = sin_phi.shape[0]
        DCM = torch.zeros((batch_size, 3, 3), dtype=torch.float32).to(self.device)
        DCM[:, 0, 0] = cos_theta * cos_psi
        DCM[:, 0, 1] = cos_theta * sin_psi
        DCM[:, 0, 2] = -sin_theta
        DCM[:, 1, 0] = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
        DCM[:, 1, 1] = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
        DCM[:, 1, 2] = sin_phi * cos_theta
        DCM[:, 2, 0] = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
        DCM[:, 2, 1] = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
        DCM[:, 2, 2] = cos_phi * cos_theta
        return DCM
    
    def compute_rotation_matrix(self, cos_theta_t, sin_theta_t):
        batch_size = cos_theta_t.shape[0]
        rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float32).to(self.device)
        rotation_matrix[:, 0, 0] = cos_theta_t
        rotation_matrix[:, 0, 2] = sin_theta_t
        rotation_matrix[:, 1, 1] = 1.0
        rotation_matrix[:, 2, 0] = -sin_theta_t
        rotation_matrix[:, 2, 2] = cos_theta_t
        return rotation_matrix

    
    


   

