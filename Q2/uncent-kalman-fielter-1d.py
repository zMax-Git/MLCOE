'''
Author: Jiajia HUANG
Date: 2026-01-12 21:17:10
LastEditors: Jiajia HUANG
LastEditTime: 2026-01-12 22:13:30
Description: fill the seacription
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
np.random.seed(42)

def generate_data(n_obs, alpha=0.91, sigma=1.0, beta=0.5):
    """
    Stochastic Volatility Model (Example 4 from the paper)
    
    State equation: X_n = alpha * X_{n-1} + sigma * V_n
    Observation equation: Y_n = beta * exp(X_n / 2) * W_n
    
    where V_n, W_n ~ N(0, 1)
    """
    
    X_true = np.zeros(n_obs)
    Y = np.zeros(n_obs)
    
    x0_mean = 0
    x0_var = sigma**2 / (1 - alpha**2)
    
    X_true[0] = np.random.normal(x0_mean, np.sqrt(x0_var))
    Y[0] = beta * np.exp(X_true[0]/2) * np.random.normal()
    
    for t in range(1, n_obs):
        X_true[t] = alpha * X_true[t-1] + sigma * np.random.normal()
        Y[t] = beta * np.exp(X_true[t]/2) * np.random.normal()

    return X_true, Y

# =========================
# Unscented Transform tools
# =========================

def run_ukf_sv(Y, alpha_sv=0.91, sigma_sv=1.0, beta_sv=0.5,
                     alpha=1e-3, beta=2.0, kappa=0.0,
                     ):
    """
    UKF for stochastic volatility model
    log(Y^2) = X + log(beta^2) + log(epsilon^2)
    """
    
    T = len(y)
    
    mean_log_eps2 = -1.27
    var_log_eps2 = np.pi**2 / 2
    Y_transformed = np.log(Y**2 + 1e-10)
    c = np.log(beta_sv**2) + mean_log_eps2
    
    # storage
    x_filt = np.zeros(T)
    P_filt = np.zeros(T)
    # initial state belief
    x_filt[0] = 0
    P_filt[0] = sigma_sv**2 / (1 - alpha_sv**2)
    
    L = 1
    lambda_ = alpha**2 * (L + kappa) - L
    for t in range(1, T):
        
        sigma_points = np.zeros(2*L+1)
            
        P = max(P_filt[t-1], 1e-10)
        sqrt_matrix = np.sqrt((L + lambda_) * P)
        
        sigma_points[0] = x_filt[t-1]
        sigma_points[1] = x_filt[t-1] + sqrt_matrix
        sigma_points[2] = x_filt[t-1] - sqrt_matrix
        
        Wm = np.zeros(2*L+1)
        Wc = np.zeros(2*L+1)
        Wm[0] = lambda_ / (L + lambda_)
        Wc[0] = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)
        for i in range(1, 2*L+1):
            Wm[i] = 1 / (2 * (L + lambda_))
            Wc[i] = 1 / (2 * (L + lambda_))    
        # -------------------------
        # 2. Time prediction
        # -------------------------
        x_pred = alpha_sv * sigma_points
        x_pred_mean = np.sum(Wm * x_pred)
        
        y_pred = x_pred + c
        y_pred_mean = np.sum(Wm * y_pred)
        
        Pxx = sigma_sv**2
        Pxy = 0.0
        Pyy = var_log_eps2
        for i in range(len(sigma_points)):
            dx = x_pred[i] - x_pred_mean
            dy = y_pred[i] - y_pred_mean
            Pxx += Wc[i] * dx * dx
            Pxy += Wc[i] * dx * dy
            Pyy += Wc[i] * dy * dy
        
        K = Pxy  / Pyy
        x_filt[t] = x_pred_mean + K * (Y_transformed[t] - y_pred_mean)
        P_filt[t] = Pxx - K * Pyy * K
       
    return x_filt, P_filt

if __name__ == "__main__":
    T = 500
    
    x, y = generate_data(T)
    x_hat, P = run_ukf_sv(y )
    plt.style.use('seaborn-v0_8-darkgrid') 
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1 = axes[0]
    # ax1.scatter(range(T), y, label='Observations, $Y_n$', color='#F18F01', s=15, alpha=0.6, edgecolors='none')
    ax1.plot(range(T), y, label='Observations, $Y_n$', color='#F18F01', alpha=0.6)
    ax1.set_ylabel('Observation Value', fontsize=12, fontweight='bold')
    ax1.set_title('Observations', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    # for point in extrem_points:
    #     ax1.scatter([point], [y[point]], color='red', s=100, alpha=0.5, edgecolors='none')
    # 第二个子图：状态估计对比
    ax2 = axes[1]
    ax2.plot(x, label='True State', color='#2E86AB', linewidth=2, alpha=0.8)
    ax2.plot(x_hat, label='UKF Estimated State', color='#A23B72', linewidth=2, alpha=0.8)
    
    # 添加不确定性区间（±2标准差）
    uncertainty = 2 * np.sqrt(P)
    ax2.fill_between(range(T), x_hat - uncertainty, x_hat + uncertainty, 
                     color='#A23B72', alpha=0.2, label='95% Confidence Interval')
    
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Volatility State, $X_n$', fontsize=12, fontweight='bold')
    ax2.set_title('Stochastic Volatility Model: State Estimation', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    # ax2.scatter([55],[] )
    plt.tight_layout()
    plt.savefig('uncent-kalman-fielter-1d.png', dpi=300, bbox_inches='tight')
    plt.show()
    print()