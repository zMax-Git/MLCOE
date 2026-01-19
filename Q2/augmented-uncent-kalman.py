'''
Author: Jiajia HUANG
Date: 2026-01-06 17:11:12
LastEditors: Jiajia HUANG
LastEditTime: 2026-01-14 15:56:13
Description: fill the seacription
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
def generate_data(alpha, sigma, beta, n_obs):
    """
    Stochastic Volatility Model (Example 4 from the paper)
    
    State equation: X_n = alpha * X_{n-1} + sigma * V_n
    Observation equation: Y_n = beta * exp(X_n / 2) * W_n
    
    where V_n, W_n ~ N(0, 1)
    """
    
    X_true = np.zeros(n_obs)
    Y = np.zeros(n_obs)
    
    X_true[0] = np.random.normal(0, sigma/np.sqrt(1-alpha**2))
    for t in range(1, n_obs):
        X_true[t] = alpha * X_true[t-1] + sigma * np.random.normal()
        
    for t in range(n_obs):
        Y[t] = beta * np.exp(X_true[t]/2) * np.random.normal()
    return X_true, Y

# =========================
# Unscented Transform tools
# =========================
def sigma_points(mean, cov, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Generate sigma points and weights for UT
    """
    n = len(mean)
    lam = alpha**2 * (n + kappa) - n
    S = np.linalg.cholesky((n + lam) * cov)

    chi = np.zeros((2 * n + 1, n))
    chi[0] = mean

    for i in range(n):
        chi[i + 1]     = mean + S[:, i]
        chi[i + 1 + n] = mean - S[:, i]

    Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))

    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    return chi, Wm, Wc

# =========================
# Augmented UKF for SV model
# =========================
def augmented_ukf_sv(y, alpha_sv, sigma_sv, beta_sv,
                     m0, P0):
    """
    Augmented UKF for stochastic volatility model
    """
    T = len(y)
    log_y = np.log(y**2 + 1e-10)
    # storage
    m_filt = np.zeros(T)
    P_filt = np.zeros(T)
    # initial state belief
    m = m0
    P = P0
    for t in range(T):
        # -------------------------
        # 1. Augmented state
        # Z = [X, V, W]
        # -------------------------
        mean_z = np.array([m, 0.0, 0.0])
        cov_z  = np.diag([P, 1.0, 1.0])

        chi, Wm, Wc = sigma_points(mean_z, cov_z)
        # -------------------------
        # 2. Time prediction
        # -------------------------
        chi_pred = np.zeros_like(chi)
        for i in range(len(chi)):
            X, V, W = chi[i]
            X_pred = alpha_sv * X + sigma_sv * V
            chi_pred[i] = np.array([X_pred, V, W])
        # predicted mean
        mean_pred = np.sum(Wm[:, None] * chi_pred, axis=0)
        m_pred = mean_pred[0]

        # predicted covariance
        cov_pred = np.zeros((3, 3))
        for i in range(len(chi)):
            diff = chi_pred[i] - mean_pred
            cov_pred += Wc[i] * np.outer(diff, diff)

        P_pred = cov_pred[0, 0]
        # -------------------------
        # 3. Measurement prediction
        # y = beta * exp(X/2) * W
        # -------------------------
        y_sigma = np.zeros(len(chi))
        for i in range(len(chi)):
            Xp, _, Wp = chi_pred[i]
            # y_sigma[i] = beta_sv * np.exp(Xp / 2.0) * Wp ### failed because sigma point is symmetric
            y_sigma[i] =  Xp + np.log(beta_sv**2) + np.log((Wp+1e-6)**2)
            
        y_pred = np.sum(Wm * y_sigma)

        S = 0.0
        Cxy = 0.0
        for i in range(len(chi)):
            dy = y_sigma[i] - y_pred
            dx = chi_pred[i, 0] - m_pred
            S   += Wc[i] * dy * dy #+np.pi**2 / 2
            Cxy += Wc[i] * dx * dy
     
        # -------------------------
        # 4. Kalman update
        # -------------------------
        K = Cxy / S

        m = m_pred + K * (log_y[t] - y_pred)
        P = P_pred - K * S * K

        m_filt[t] = m
        P_filt[t] = P

    return m_filt, P_filt

if __name__ == "__main__":
    np.random.seed(42)
    T = 500
    alpha, sigma, beta = 0.91, 1.0, 0.5
    X_true, Y = generate_data(alpha, sigma, beta, T)
    m0 = 0.001
    P0 = sigma**2 / (1 - alpha**2)
    m_filt, P_filt = augmented_ukf_sv(Y, alpha, sigma, beta, m0, P0)
    print(m_filt)
    print(P_filt)
    plt.plot(X_true, 'g-', label='True X', alpha=0.7, linewidth=1.5)
    plt.plot(m_filt, 'r-', label='UKF estimate', alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('augmented-uncent-kalman.png', dpi=300, bbox_inches='tight')
    plt.show()
