'''
Author: Jiajia HUANG
Date: 2025-12-07 21:43:03
LastEditors: Jiajia HUANG
LastEditTime: 2025-12-09 12:58:59
Description: fill the seacription
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
np.random.seed(42)

class LinearGaussianSSM:
    def __init__(self, A, B, C, D, Sigma_init):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Sigma_init = Sigma_init
        
        # Derived Covariances based on Doucet(09) definitions
        # Q = B * I * B^T
        self.Q = self.B @ self.B.T
        # R = D * I * D^T
        self.R = self.D @ self.D.T
        
        self.nx = A.shape[0]
        self.ny = C.shape[0]

    def simulate(self, T):
        """
        Generates synthetic data based on Example 2.
        """
        x_true = np.zeros((T, self.nx))
        y_obs = np.zeros((T, self.ny))
        
        # Initialization X1 ~ N(0, Sigma)
        x_true[0] = np.random.multivariate_normal(np.zeros(self.nx), self.Sigma_init)
        print(f'The initial state is: x_0 = {x_true[0][0]:.2f}, y_0 = {x_true[0][1]:.2f}, vx_0 = {x_true[0][2]:.2f}, vy_0 = {x_true[0][3]:.2f}')
        # Generate noise vectors V and W
        # V ~ N(0, I_nv), W ~ N(0, I_nw)
        # Dimensions of V match columns of B, W match columns of D
        nv = self.B.shape[1]
        nw = self.D.shape[1]
        
        for n in range(1, T):
            vn = np.random.randn(nv)
            # X_n = A X_{n-1} + B V_n
            x_true[n] = self.A @ x_true[n-1] + self.B @ vn
            
        for n in range(T):
            wn = np.random.randn(nw)
            # Y_n = C X_n + D W_n
            y_obs[n] = self.C @ x_true[n] + self.D @ wn
            
        return x_true, y_obs

def run_kalman_filter(ssm, y_obs, P_updation='standard'):
    """
    Joseph Stabilized Kalman Filter
    Uses: P = (I - KC)P_pred(I - KC)^T + KRK^T
    This form maintains symmetry and positive definiteness better.
    """
    T = y_obs.shape[0]
    nx = ssm.nx
    
    # Storage
    x_filt = np.zeros((T, nx))
    P_filt = np.zeros((T, nx, nx))
    
    x_pred = np.zeros(nx) 
    P_pred = ssm.Sigma_init
    condition_numbers = np.zeros(T)
    ## Xn = A * X_{n-1} + B * V_n
    ## Yn = C * Xn + D * Wn
    for k in range(T):

        y_pred = ssm.C @ x_pred ## The iniitial x_pred is 0
        innovation = y_obs[k] - y_pred
        
        S = ssm.C @ P_pred @ ssm.C.T + ssm.R
        K = P_pred @ ssm.C.T @ np.linalg.inv(S)
        
        x_now = x_pred + K @ innovation
        
        I = np.eye(nx)
        if P_updation == 'joseph':
            P_now = (I - K @ ssm.C) @ P_pred @ (I - K @ ssm.C).T + K @ ssm.R @ K.T
        elif P_updation == 'standard':
            P_now = (I - K @ ssm.C) @ P_pred 
        else:
            raise ValueError(f"Invalid P_updation method: {P_updation}")
        
        x_filt[k] = x_now
        P_filt[k] = P_now
        condition_numbers[k] = np.linalg.cond(P_now)
        
        x_pred = ssm.A @ x_now
        P_pred = ssm.A @ P_now @ ssm.A.T + ssm.Q 
            
    return x_filt, P_filt, condition_numbers

def check_positive_definite(P):
    """Check if matrix is positive definite"""
    try:
        np.linalg.cholesky(P)
        return True
    except np.linalg.LinAlgError:
        return False



if __name__ == "__main__":

    dt = 1 # Time step
    sigma_var = 5 # Initial variance
    b = 0.5 #0.5 # Scaling factor for process noise， here is for vx and vy
    d = 10.0 # Scaling factor for measurement noise, here is for x and y
    T_steps = 100

    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    B = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]) * b

    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]) 

    D = np.eye(2) * d # Measurement noise 

    Sigma_init = np.eye(4) * sigma_var

    lgssm = LinearGaussianSSM(A, B, C, D, Sigma_init)
    x_true, y_obs = lgssm.simulate(T_steps)
    
    x_est, P_est, condition_numbers = run_kalman_filter(lgssm, y_obs, P_updation='standard')
    x_est_j, P_est_j, condition_numbers = run_kalman_filter(lgssm, y_obs, P_updation='joseph')

    plt.figure(figsize=(5, 8))
    plt.suptitle(f'$vx_0$={x_true[0][2]:.2f}, $vy_0$={x_true[0][3]:.2f}, b={b}, d={d}')
    plt.subplot(2,1 , 1)
    plt.plot(x_true[:, 0], label='True ($x_t$)', color='black')
    plt.scatter(range(T_steps), y_obs[:, 0], label='Observations ($x_t$)', color='red', s=10, alpha=0.5)
    plt.plot(x_est[:, 0], label='KF Estimate ($\hat{x}_1$)', color='blue', )
    plt.fill_between(range(T_steps), 
                    x_est[:, 0] - 1.96 * np.sqrt(P_est[:, 0, 0]), 
                    x_est[:, 0] + 1.96 * np.sqrt(P_est[:, 0, 0]), 
                    color='blue', alpha=0.2, label='95% CI')
    plt.xlabel('Time Step')
    plt.ylabel('Position X')
    plt.legend()
    plt.grid(True)
    # plt.savefig('Kalman_x.png', dpi=200)
    # Plot Trajectory (2D)
    # plt.figure()
    plt.subplot(2,1 , 2)
    plt.plot(x_true[:, 0], x_true[:, 1], label='True Trajectory', color='black')
    plt.scatter(y_obs[:, 0], y_obs[:, 1], label='Observations', color='red', s=10, alpha=0.5)
    plt.plot(x_est[:, 0], x_est[:, 1], label='KF Estimated Trajectory', color='blue')
    # plt.title('2D Trajectory (Latent State)')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Kalman.png', dpi=200)
    plt.show()