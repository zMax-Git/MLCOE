'''
Author: Jiajia HUANG
Date: 2025-12-08 23:17:40
LastEditors: Jiajia HUANG
LastEditTime: 2025-12-14 21:33:49
Description: fill the seacription
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        # Generate noise vecktors V and W
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

def run_kalman_filter(ssm, y_obs):
    """
    Standard Kalman Filter Implementation.
    
    Args:
        ssm: Instance of LinearGaussianSSM
        y_obs: Observations (T, ny)
        
    Returns:
        x_est: Filtered state estimates (T, nx)
        P_est: Filtered error covariance matrices (T, nx, nx)
    """
    T = y_obs.shape[0]
    nx = ssm.nx
    
    # Storage
    x_before_correction = np.zeros((T, nx))
    P_before_correction = np.zeros((T, nx, nx))
    x_filt = np.zeros((T, nx))
    P_filt = np.zeros((T, nx, nx))
    kalman_gain = np.zeros((T, ssm.C.shape[1], ssm.C.shape[0]))
    
    # Initialization (At time 0)
    # We assume we start with the prior N(0, Sigma_init)
    x_pred = np.zeros(nx) 
    P_pred = ssm.Sigma_init
    ## Xn = A * X_{n-1} + B * V_n
    ## Yn = C * Xn + D * Wn
    for k in range(T):
        # --- Update Step (Correction) ---
        # Innovation (Residual): y - C * x_pred
        y_pred = ssm.C @ x_pred
        innovation = y_obs[k] - y_pred
        
        # Innovation Covariance: S = C P C^T + R
        S = ssm.C @ P_pred @ ssm.C.T + ssm.R
        
        # Kalman Gain: K = P C^T S^-1
        K = P_pred @ ssm.C.T @ np.linalg.inv(S)
        kalman_gain[k] = K
        # State Update: x = x_pred + K * innovation
        x_now = x_pred + K @ innovation
        
        # Covariance Update: P = (I - K C) P_pred
        I = np.eye(nx)
        P_now = (I - K @ ssm.C) @ P_pred 
        
        # Store results
        x_before_correction[k] = x_pred
        P_before_correction[k] = P_pred
        x_filt[k] = x_now
        P_filt[k] = P_now
        kalman_gain[k] = K

        x_pred = ssm.A @ x_now
        P_pred = ssm.A @ P_now @ ssm.A.T + ssm.Q 
            
    return x_filt, P_filt, x_before_correction, P_before_correction, kalman_gain



def run_without_filter(ssm, y_obs):

    T = y_obs.shape[0]
    nx = ssm.nx
    
    # Storage
    x_filt = np.zeros((T, nx))
    P_filt = np.zeros((T, nx, nx))

    # Initialization (At time 0)
    # We assume we start with the prior N(0, Sigma_init)
    x_pred = np.zeros(nx) 
    P_pred = ssm.Sigma_init
    ## Xn = A * X_{n-1} + B * V_n
    ## Yn = C * Xn + D * Wn
    ### using the first observation to initialize the filter
    x_pred[0]= y_obs[0][0]
    x_pred[1] = y_obs[0][1]
    P_pred[0][0] = ssm.R[0][0]
    P_pred[1][1] = ssm.R[1][1]
    for k in range(T):
      
        x_now = x_pred
        P_now = P_pred
    
        x_filt[k] = x_now
        P_filt[k] = P_now

        x_pred = ssm.A @ x_now
        P_pred = ssm.A @ P_now @ ssm.A.T + ssm.Q 
            
    return x_filt, P_filt


def plot_pyplot():
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=['(a) Position X vs Time', '(b) Error of Position X vs Time', '(c) Kalman Gain and posterior covariance'],
                        vertical_spacing=0.12,
                        specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}]])
    time = list(range(T_steps))
    fig.add_trace(go.Scatter(x=time, y=x_true[:, 0], 
                            mode='lines', name='$x_t$', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=y_obs[:, 0], 
                            mode='markers', name='$x_t^{obs}$)', 
                            marker=dict(color='red', size=5, opacity=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=x_before_correction[:, 0], 
                            mode='lines', name='$\hat{x}_{t|t-1}$', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=x_est[:, 0], 
                            mode='lines', name='$\hat{x}_{t|t}$', line=dict(color='blue')), row=1, col=1)

    fig.add_trace(go.Scatter(x=time, y=abs(x_true[:, 0] - x_before_correction[:, 0]), 
                            mode='lines', name='$|x_t - \hat{x}_{t|t-1}|$', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=abs(x_true[:, 0] - x_est[:, 0]), 
                            mode='lines', name='$|x_t - \hat{x}_{t|t}|$', line=dict(color='blue')), row=2, col=1)


    fig.add_trace(go.Scatter(x=time, y=kalman_gain[:, 0, 0], 
                            mode='lines', name='Kalman Gain $K_t$', line=dict(color='red')), row=3 , col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=time, y=P_before_correction[:, 0, 0], 
                            mode='lines', name='$P_{t|t-1}$', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=time, y=P_est[:, 0, 0], 
                            mode='lines', name='$P_{t|t}$', line=dict(color='blue')), row=3, col=1)




    # fig.write_html(f'Kalman-b-{b}-d-{d}-2.html')
    fig.write_image(f'Kalman-b-{b}-d-{d}-2.png', scale=2, width=1000, height=1000)
    fig.show()

def plot_matplotlib():
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    # fig.suptitle('Kalman Filter Results', fontsize=14)
    time = np.arange(T_steps)
    
    # Subplot 1: Position X vs Time
    ax1 = axes[0]
    ax1.plot(time, x_true[:, 0], 'k-', label='$x_t$', linewidth=1.5)

    ax1.scatter(time, y_obs[:, 0], c='red', s=5, alpha=0.5, label='$x_t^{obs}$')
    ax1.plot(time, x_before_correction[:, 0], 'g-', label='$\hat{x}_{t|t-1}$', linewidth=1.5)
    ax1.plot(time, x_est[:, 0], 'b-', label='$\hat{x}_{t|t}$', linewidth=1.5)
    ax1.fill_between(time, 
                    x_est[:, 0] - 1.96 * np.sqrt(P_est[:, 0, 0]), 
                    x_est[:, 0] + 1.96 * np.sqrt(P_est[:, 0, 0]), 
                    color='blue', alpha=0.2, label='95% CI of $\hat{x}_{t|t}$')
    # ax1.set_title('(a) Position X vs Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position X')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Subplot 2: Error of Position X vs Time
    ax2 = axes[1]
    ax2.plot(time, abs(x_true[:, 0] - x_before_correction[:, 0]), 'g-', 
            label='$|x_t - \hat{x}_{t|t-1}|$', linewidth=1.5)
    ax2.plot(time, abs(x_true[:, 0] - x_est[:, 0]), 'b-', 
            label='$|x_t - \hat{x}_{t|t}|$', linewidth=1.5)
    # ax2.set_title('(b) Error of Position X vs Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Subplot 3: Kalman Gain and posterior covariance (with dual y-axis)
    ax3 = axes[2]
    ax3.plot(time, P_before_correction[:, 0, 0], 'g-', label='$P_{t|t-1}$', linewidth=1.5)
    ax3.plot(time, P_est[:, 0, 0], 'b-', label='$P_{t|t}$', linewidth=1.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Covariance', color='black')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.legend(loc='upper center')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # ax3.set_ylim(0, 70)
    ax3.set_ylim(0, 500)
    # Secondary y-axis for Kalman Gain
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time, kalman_gain[:, 0, 0], 'r-', label='Kalman Gain $K_t$', linewidth=1.5)
    ax3_twin.set_ylabel('Kalman Gain $K_t$', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3_twin.legend(loc='upper right')
    # ax3_twin.set_ylim(0, 1)
    ax3_twin.set_ylim(0, 0.25)
    
    # ax3.set_title('(c) Kalman Gain and posterior covariance')
    
    plt.tight_layout()
    plt.savefig(f'Kalman-b-{b}-d-{d}-2.png', dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    dt = 1 # Time step
    sigma_var = 5 # Initial variance
    b = 0.5 #0.5 # Scaling factor for process noise， here is for vx and vy
    d = 50.0 # Scaling factor for measurement noise, here is for x and y
    T_steps = 200

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
    x_est, P_est, x_before_correction, P_before_correction, kalman_gain = run_kalman_filter(lgssm, y_obs)
    x_without_filter, P_without_filter = run_without_filter(lgssm, y_obs)
    # data = {
    #     'x_true': x_true[:, 0],
    #     'x_obs': y_obs[:, 0],
    #     'kalman_gain': kalman_gain[:, 0, 0],
    #     'x_before_correction': x_before_correction[:, 0],
    #     'x_est': x_est[:, 0],
    #     'y_true': x_true[:, 1],
    #     'y_obs': y_obs[:, 1],
    #     'y_before_correction': x_before_correction[:, 1],
    #     'y_est': x_est[:, 1],

    # }
    # df = pd.DataFrame(data)
    # df.round(2).to_excel('Kalman-recursive-means.xlsx', index=False)
    # plot_pyplot()
    plot_matplotlib()