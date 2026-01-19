import numpy as np
import matplotlib.pyplot as plt

# Generate data
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

# def run_uncent_kalman_filter(Y, alpha_sv, beta_sv, sigma_sv):
#     """
#     Run unscented Kalman filter for SV model
#     # log(Y^2) = log(beta^2) + X + log(epsilon^2)
#     observation equation: Y_n = beta_sv * exp(X_n / 2) * W_n
#     state equation: X_n = alpha_sv * X_{n-1} + sigma_sv * V_n
#     https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
#     """
#     n = len(Y)
#     L=1
#     ### x_hat: [X_n, V_n, W_n]
#     x_hat = np.zeros((n, 3))
#     P = np.zeros((n, 3, 3))
#     x_pred = np.zeros((n, 3))
#     P_pred = np.zeros((n, 3, 3))

    
#     alpha = 0.001
#     beta = 2
#     kappa = 0
#     lambda_ = alpha**2 * (1 + kappa) - L
#     n_sigma = 2 * L + 1
    
#     Wm = np.zeros(n_sigma)
#     Wc = np.zeros(n_sigma)
#     Wm[0] = lambda_ / (1 + lambda_)
#     Wc[0] = lambda_ / (1 + lambda_) + (1 - alpha**2 + beta)
#     for i in range(1, n_sigma):
#         Wm[i] = 1 / (2 * (L + lambda_))
#         Wc[i] = 1 / (2 * (L + lambda_))
    
#     x_hat[0] = np.array([0, np.random.normal(), np.random.normal()]) # W_n ~ N(0, 1)
#     P_pred[0] = np.diag([sigma_sv**2 / (1 - alpha_sv**2), 1, 1])
    
#     for t in range(1, n):
#         # x_pred[t][0] = alpha_sv * x_hat[t-1][0] + sigma_sv * x_hat[t-1][1]
#         # x_pred[t][1] = np.random.normal()
#         # x_pred[t][2] = np.random.normal()
   
        
#         sigma_points = np.zeros((n_sigma, 3))
        
#         sigma_points[0] = x_pred[t-1]
#         for i in range(3):
#             sigma_points[1][i] = x_pred[t-1][i] + np.sqrt((L + lambda_) * P_pred[t][i])
#             sigma_points[2][i] = x_pred[t-1][i] - np.sqrt((L + lambda_) * P_pred[t][i])
            
#         sigma_points_pred = np.array([alpha_sv * sp[0] + sigma_sv * sp[1] for sp in sigma_points])
#         print()
        
        

class UKF_StochasticVolatility:
    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5):
        # Model parameters
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        
        # UKF parameters
        self.n_x = 1  # State dimension
        self.n_y = 1  # Observation dimension
        
        # UKF scaling parameters
        self.alpha_ukf = 0.001
        self.beta_ukf = 2
        self.kappa = 0 #3 - self.n_x
        
        # Derived parameters
        self.lambda_ = self.alpha_ukf**2 * (self.n_x + self.kappa) - self.n_x
        self.n_sigma = 2 * self.n_x + 1
        
        # Weights
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        
        self.Wm[0] = self.lambda_ / (self.n_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n_x + self.lambda_) + (1 - self.alpha_ukf**2 + self.beta_ukf)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * (self.n_x + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.n_x + self.lambda_))
    
    def generate_sigma_points(self, x, P):
        """Generate sigma points"""
        n = 1
        sigma_points = np.zeros((self.n_sigma, n))
        
        P = max(P, 1e-10)
        sqrt_matrix = np.sqrt((n + self.lambda_) * P)
        
        sigma_points[0] = x
        sigma_points[1] = x + sqrt_matrix
        sigma_points[2] = x - sqrt_matrix
        
        return sigma_points
    
    
    def h_function(self, x, y_obs):
        """
        Observation function:
        log(|Y_n|) ≈ log(beta) + X_n/2 + log(|W_n|)
        or log(Y^2) = log(beta^2) + X + log(epsilon^2)
        or use: Y_n^2 = beta^2 * exp(X_n) * W_n^2
        """
        # 使用对数绝对值变换
        if abs(y_obs) > 1e-10:
            # 预期的 log(|Y|) given X
            return np.log(self.beta) + x/2
            # return np.log(self.beta) + x/2
        else:
            return 0
    
    def predict(self, x, P):
        """Prediction step"""
        # Generate sigma points
        sigma_points = self.generate_sigma_points(x, P)
        
        # Propagate through state transition
        sigma_points_pred = np.array([self.alpha * sp for sp in sigma_points])
        
        # Compute predicted mean and covariance
        x_pred = np.sum(self.Wm * sigma_points_pred.T, axis=1)[0]
        
        P_pred = self.sigma**2  # Process noise
        for i in range(self.n_sigma):
            diff = sigma_points_pred[i, 0] - x_pred
            P_pred += self.Wc[i] * diff**2
        
        return x_pred, P_pred
    
    def update(self, x_pred, P_pred, y_obs):
        """Update step using the actual observation"""
        # Generate sigma points
        sigma_points = self.generate_sigma_points(x_pred, P_pred)
        
        # 使用平方观测的对数变换
        if abs(y_obs) > 1e-10:
            # 对数变换观测
            z_obs = np.log(y_obs**2)
            
            # 计算每个 sigma point 的预期对数平方观测
            z_sigma = np.zeros(self.n_sigma)
            for i in range(self.n_sigma):
                # E[log(Y^2)|X] ≈ log(beta^2) + X + E[log(W^2)]
                # E[log(W^2)] ≈ -1.27 (for standard normal)
                z_sigma[i] = np.log(self.beta**2) + sigma_points[i, 0] - 1.27
            
            # 预测的观测均值
            z_pred = np.sum(self.Wm * z_sigma)
            
            # 创新协方差
            Pzz = 4.93  # Var[log(W^2)] ≈ 4.93 for standard normal
            for i in range(self.n_sigma):
                diff_z = z_sigma[i] - z_pred
                Pzz += self.Wc[i] * diff_z**2
            
            # 交叉协方差
            Pxz = 0
            for i in range(self.n_sigma):
                diff_x = sigma_points[i, 0] - x_pred
                diff_z = z_sigma[i] - z_pred
                Pxz += self.Wc[i] * diff_x * diff_z
            
            # Kalman gain
            K = Pxz / Pzz if Pzz > 1e-10 else 0
            
            # Update
            innovation = z_obs - z_pred
            x_new = x_pred + K * innovation
            P_new = P_pred - K * Pzz * K
            
        else:
            # 如果观测接近0，使用预测值
            x_new = x_pred
            P_new = P_pred
        
        P_new = max(P_new, 1e-10)
        return x_new, P_new
    
    def filter(self, observations):
        """Run UKF on observations"""
        n = len(observations)
        
        # Initialize
        x = 0.0
        P = self.sigma**2 / (1 - self.alpha**2)
        
        # Storage
        filtered_states = np.zeros(n)
        filtered_variances = np.zeros(n)
        
        for t in range(n):
            # Predict
            x_pred, P_pred = self.predict(x, P)
            
            # Update
            x, P = self.update(x_pred, P_pred, observations[t])
            
            filtered_states[t] = x
            filtered_variances[t] = P
        
        return filtered_states, filtered_variances



if __name__ == "__main__":
    np.random.seed(42)
    T = 500
    alpha, sigma, beta = 0.91, 1.0, 0.5
    X_true, Y = generate_data(alpha, sigma, beta, T)
    # Apply filters
    print("Running UKF...")
    ukf = UKF_StochasticVolatility(alpha=alpha, sigma=sigma, beta=beta)
    X_ukf, P_ukf = ukf.filter(Y)
    # X_ukf, P_ukf = run_uncent_kalman_filter(Y, alpha, beta, sigma)



    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    # Observations
    axes[0].plot(Y, 'b-', alpha=0.6, linewidth=0.8, label='Observations Y')
    axes[0].plot(Y**2, 'c-', alpha=0.3, linewidth=0.5, label='Y²')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Stochastic Volatility Model - UKF vs Particle Filter')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # True vs filtered states
    axes[1].plot(X_true, 'g-', label='True X', alpha=0.7, linewidth=1.5)
    axes[1].plot(X_ukf, 'r-', label='UKF estimate', alpha=0.7)
    # axes[1].plot(X_pf, 'b--', label='PF estimate', alpha=0.7)
    axes[1].set_ylabel('Log-volatility X')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Volatility
    axes[2].plot(np.exp(X_true/2), 'g-', label='True volatility', alpha=0.7)
    axes[2].plot(np.exp(X_ukf/2), 'r-', label='UKF volatility', alpha=0.7)
    # axes[2].plot(np.exp(X_pf/2), 'b--', label='PF volatility', alpha=0.7)
    axes[2].set_ylabel('exp(X/2)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Errors
    axes[3].plot(X_true - X_ukf, 'r-', alpha=0.7, label='UKF error')
    # axes[3].plot(X_true - X_pf, 'b-', alpha=0.5, label='PF error')
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Error')
    axes[3].set_xlabel('Time')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('uncent-kalman-fielter.png', dpi=300, bbox_inches='tight')
    plt.show()

