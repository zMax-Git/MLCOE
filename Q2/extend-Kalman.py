'''
Author: Jiajia HUANG
Date: 2025-12-09 13:01:51
LastEditors: Jiajia HUANG
LastEditTime: 2026-01-12 14:19:45
Description: fill the seacription
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
np.random.seed(42)

class StochasticVolatilityModel:
    """
    Stochastic Volatility Model (Example 4 from the paper)
    
    State equation: X_n = alpha * X_{n-1} + sigma * V_n
    Observation equation: Y_n = beta * exp(X_n / 2) * W_n
    
    where V_n, W_n ~ N(0, 1)
    """
    
    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        
        # Initial state distribution
        self.x0_mean = 0
        self.x0_var = sigma**2 / (1 - alpha**2)
    
    def simulate(self, T):
        """
        Generate synthetic data based on Example 4.
        """
        x = np.zeros(T)
        y = np.zeros(T)
        
        # Initial state
        x[0] = np.random.normal(self.x0_mean, np.sqrt(self.x0_var))
        print(f'The initial state is: x_0 = {x[0]:.2f}')
        y[0] = self.beta * np.exp(x[0] / 2) * np.random.normal(0, 1)
        print(f'The initial observation is: y_0 = {y[0]:.2f}')
        for t in range(1, T):
            # Process noise
            x[t] = self.alpha * x[t-1] + self.sigma * np.random.normal(0, 1)
            y[t] = self.beta * np.exp(x[t] / 2) * np.random.normal(0, 1)
        return x, y


def run_kalman_filter_for_sv(Y, alpha, beta, sigma):
    """
    Extended Kalman Filter for SV model
    log(Y^2) = log(beta^2) + X + log(epsilon^2)
    """
    n = len(Y)

    Y_transformed = np.log(Y**2 + 1e-10) ## avoid log(0)

    mean_log_eps2 = -1.27
    var_log_eps2 = np.pi**2 / 2

    c = np.log(beta**2) + mean_log_eps2
  
    x_hat = np.zeros(n)      # 状态估计 (后验)
    P = np.zeros(n)          # 估计协方差 (后验)
    x_pred = np.zeros(n)     # 状态预测 (先验)
    P_pred = np.zeros(n)     # 预测协方差 (先验)
  
    x_hat[0] = 0
    P[0] = sigma**2 / (1 - alpha**2)

    for t in range(1, n):

        x_pred[t] = alpha * x_hat[t-1]
        P_pred[t] = alpha**2 * P[t-1] + sigma**2
        
        z_t = Y_transformed[t]
        y_tilde = z_t - (x_pred[t] + c)
        
        S = P_pred[t] + var_log_eps2
        K = P_pred[t] / S
        
        x_hat[t] = x_pred[t] + K * y_tilde
        P[t] = (1 - K) * P_pred[t]
    return x_hat, P



if __name__ == "__main__":
    T = 500
    sv = StochasticVolatilityModel()
    x, y = sv.simulate(T)
    x_hat, P = run_kalman_filter_for_sv(y, 0.91, 0.5, 1.0)
    
    extrem_points = [55, 189]
    # plt.figure(figsize=(10, 5))
    # plt.plot(x, label='True', color='blue')
    # plt.plot(x_hat, label='Estimated', color='black')
    # plt.scatter(range(T), y, label='Observations', color='red', s=10, alpha=0.5)
    # plt.xlabel('Time')
    # plt.ylabel('Volatility')
    # plt.title('Stochastic Volatility Model')
    # plt.legend()
    # plt.savefig('extend-Kalman.png', dpi=200)
    # plt.show()
    plt.style.use('seaborn-v0_8-darkgrid')  # 使用更现代的样式
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1 = axes[0]
    # ax1.scatter(range(T), y, label='Observations, $Y_n$', color='#F18F01', s=15, alpha=0.6, edgecolors='none')
    ax1.plot(range(T), y, label='Observations, $Y_n$', color='#F18F01', alpha=0.6)
    ax1.set_ylabel('Observation Value', fontsize=12, fontweight='bold')
    ax1.set_title('Observations', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    for point in extrem_points:
        ax1.scatter([point], [y[point]], color='red', s=100, alpha=0.5, edgecolors='none')
    # 第二个子图：状态估计对比
    ax2 = axes[1]
    ax2.plot(x, label='True State', color='#2E86AB', linewidth=2, alpha=0.8)
    ax2.plot(x_hat, label='EKF Estimated State', color='#A23B72', linewidth=2, alpha=0.8)
    
    # 添加不确定性区间（±2标准差）
    uncertainty = 2 * np.sqrt(P)
    ax2.fill_between(range(T), x_hat - uncertainty, x_hat + uncertainty, 
                     color='#A23B72', alpha=0.2, label='95% Confidence Interval')
    
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Volatility State, $X_n$', fontsize=12, fontweight='bold')
    ax2.set_title('Stochastic Volatility Model: State Estimation', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    for point in extrem_points:
        ax2.scatter([point], [x_hat[point]], color='red', s=100, alpha=0.8, edgecolors='none')
    # ax2.scatter([55],[] )
    plt.tight_layout()
    plt.savefig('extend-Kalman.png', dpi=300, bbox_inches='tight')
    plt.show()
    print()