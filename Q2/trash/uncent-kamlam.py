import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
np.random.seed(42)

class StochasticVolatilityModel:
    """
    Stochastic Volatility Model
    State: X_n = alpha * X_{n-1} + sigma * V_n
    Obs  : Y_n = beta * exp(X_n / 2) * W_n
    """
    
    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        
        # Initial state distribution
        self.x0_mean = 0
        self.x0_var = sigma**2 / (1 - alpha**2)
    
    def simulate(self, T):
        # np.random.seed(42) # 固定随机种子以便复现
        x = np.zeros(T)
        y = np.zeros(T)
        
        x[0] = np.random.normal(self.x0_mean, np.sqrt(self.x0_var))
        y[0] = self.beta * np.exp(x[0] / 2) * np.random.normal(0, 1)
        
        for t in range(1, T):
            x[t] = self.alpha * x[t-1] + self.sigma * np.random.normal(0, 1)
            y[t] = self.beta * np.exp(x[t] / 2) * np.random.normal(0, 1)
        return x, y

# def run_ukf_sv_transformed(y_obs, alpha, beta, sigma):
#     """
#     使用变换观测方程的 UKF: z_k = log(y_k^2 + offset)
#     """
#     # --- 1. 预处理观测数据 (Transform Observations) ---
#     # 使用偏移量法处理 log(0) 问题
#     offset = 1e-6
#     z_obs = np.log(y_obs**2 + offset)
    
#     # --- 2. 定义噪声参数 ---
#     # 过程噪声方差 Q
#     Q = sigma**2
    
#     # 观测噪声 R
#     # log(w^2) 的方差是 pi^2 / 2 ≈ 4.93
#     R = (np.pi**2) / 2
    
#     # 观测噪声的均值偏移
#     # E[log(w^2)] ≈ -1.2704 (负欧拉常数 - ln(2))
#     noise_mean_offset = -1.2704
    
#     # --- 3. UKF 参数设置 ---
#     L = 1              # 状态维度现在只是 1 (只估计 x)，不再需要扩维
#     alpha_ukf = 1e-3
#     kappa = 0
#     beta_ukf = 2.0
    
#     lam = alpha_ukf**2 * (L + kappa) - L
#     gamma = np.sqrt(L + lam)
    
#     # 权重
#     Wm = np.zeros(2 * L + 1)
#     Wc = np.zeros(2 * L + 1)
#     Wm[0] = lam / (L + lam)
#     Wc[0] = lam / (L + lam) + (1 - alpha_ukf**2 + beta_ukf)
#     for i in range(1, 2 * L + 1):
#         Wm[i] = 1.0 / (2 * (L + lam))
#         Wc[i] = 1.0 / (2 * (L + lam))
        
#     # --- 4. 滤波初始化 ---
#     n_steps = len(y_obs)
#     x_est = np.zeros(n_steps)
#     P_est = np.zeros(n_steps)
    
#     # 初始状态
#     x_curr = 0.0
#     P_curr = sigma**2 / (1 - alpha**2)
    
#     x_est[0] = x_curr
#     P_est[0] = P_curr
    
#     print("开始 UKF 滤波 (Transformed Observation)...")
    
#     for k in range(1, n_steps):
#         # --- A. 生成 Sigma 点 (基于上一步后验) ---
#         # 1D 情况下 Cholesky 就是 sqrt
#         sqrt_P = np.sqrt(P_curr)
        
#         X_sig = np.zeros(2 * L + 1)
#         X_sig[0] = x_curr
#         X_sig[1] = x_curr + gamma * sqrt_P
#         X_sig[2] = x_curr - gamma * sqrt_P
        
#         # --- B. 时间更新 (Prediction) ---
#         # 状态方程: x_k = alpha * x_{k-1}
#         X_sig_pred = alpha * X_sig
        
#         # 预测均值
#         x_pred_mean = np.sum(Wm * X_sig_pred)
        
#         # 预测协方差 (加上过程噪声 Q)
#         P_pred = 0
#         for i in range(2 * L + 1):
#             P_pred += Wc[i] * (X_sig_pred[i] - x_pred_mean)**2
#         P_pred += Q
        
#         # --- C. 重新生成预测后的 Sigma 点 ---
#         # 这一步对于非线性强的模型很重要，用预测的 P_pred 生成新的分布
#         sqrt_P_pred = np.sqrt(P_pred)
#         X_sig_new = np.zeros(2 * L + 1)
#         X_sig_new[0] = x_pred_mean
#         X_sig_new[1] = x_pred_mean + gamma * sqrt_P_pred
#         X_sig_new[2] = x_pred_mean - gamma * sqrt_P_pred
        
#         # --- D. 测量更新 (Correction) ---
#         # 变换后的观测方程: z = log(beta^2) + x + log(w^2)
#         # 我们预测 z 时，需要加上 log(w^2) 的期望值
        
#         # h(x) = x + log(beta^2) + E[log(w^2)]
#         Z_sig_pred = X_sig_new + np.log(beta**2) + noise_mean_offset
        
#         # 预测观测均值
#         z_pred_mean = np.sum(Wm * Z_sig_pred)
        
#         # 观测协方差 S (加上观测噪声 R)
#         S = 0
#         for i in range(2 * L + 1):
#             S += Wc[i] * (Z_sig_pred[i] - z_pred_mean)**2
#         S += R
        
#         # 互协方差 P_xy
#         P_xy = 0
#         for i in range(2 * L + 1):
#             diff_x = X_sig_new[i] - x_pred_mean
#             diff_z = Z_sig_pred[i] - z_pred_mean
#             P_xy += Wc[i] * diff_x * diff_z
            
#         # --- E. 卡尔曼增益与最终更新 ---
#         K = P_xy / S
        
#         # Innovation: 实际变换后的观测 z_obs - 预测观测
#         innovation = z_obs[k] - z_pred_mean
        
#         x_curr = x_pred_mean + K * innovation
#         P_curr = P_pred - K * S * K
        
#         x_est[k] = x_curr
#         P_est[k] = P_curr
        
#     return x_est, P_est


class StochasticVolatilityAugmentedUKF:
    """
    基于增广状态 (Augmented State) 的无迹卡尔曼滤波器 (UKF)，
    专门用于随机波动率 (Stochastic Volatility) 模型。
    
    状态方程: x_k = alpha * x_{k-1} + sigma * v_k
    观测方程: y_k = beta * exp(x_k / 2) * w_k
    其中 v_k, w_k ~ N(0, 1)
    """

    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5, 
                 x0_mean=0.0, x0_cov=1.0, 
                 alpha_ukf=1e-3, kappa=0, beta_ukf=2.0):
        """
        初始化滤波器参数
        """
        # 模型参数
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        
        # 滤波器状态初始化
        self.x = x0_mean
        self.P = x0_cov
        
        # UKF 参数 (Merwe's scaled sigma points)
        self.L = 3  # 增广状态维度 [x, v, w]
        self.alpha_ukf = alpha_ukf
        self.kappa = kappa
        self.beta_ukf = beta_ukf
        
        # 预计算权重 (Weights)
        self._init_weights()

    def _init_weights(self):
        """计算 UKF 的均值权重 (Wm) 和协方差权重 (Wc)"""
        lam = self.alpha_ukf**2 * (self.L + self.kappa) - self.L
        self.gamma = np.sqrt(self.L + lam)
        
        self.Wm = np.zeros(2 * self.L + 1)
        self.Wc = np.zeros(2 * self.L + 1)
        
        self.Wm[0] = lam / (self.L + lam)
        self.Wc[0] = lam / (self.L + lam) + (1 - self.alpha_ukf**2 + self.beta_ukf)
        
        for i in range(1, 2 * self.L + 1):
            weight = 1.0 / (2 * (self.L + lam))
            self.Wm[i] = weight
            self.Wc[i] = weight

    def step(self, y_obs):
        """
        执行单步滤波：预测 + 更新
        :param y_obs: 当前时刻的观测值
        :return: (x_est, P_est) 当前时刻的状态估计和协方差
        """
        # --- 1. 构建增广状态和协方差 ---
        # 均值: [x_curr, 0, 0] (噪声均值为0)
        x_aug_mean = np.array([self.x, 0, 0])
        
        # 协方差: 块对角矩阵
        P_aug = np.zeros((self.L, self.L))
        P_aug[0, 0] = self.P
        P_aug[1, 1] = 1.0  # Process noise variance (v_k)
        P_aug[2, 2] = 1.0  # Measurement noise variance (w_k)
        
        # --- 2. 生成 Sigma 点 ---
        try:
            sqrt_P = cholesky(P_aug, lower=True)
        except np.linalg.LinAlgError:
            # 数值稳定性处理：如果非正定，添加微小扰动
            P_aug += np.eye(self.L) * 1e-6
            sqrt_P = cholesky(P_aug, lower=True)

        X_sig = np.zeros((self.L, 2 * self.L + 1))
        X_sig[:, 0] = x_aug_mean
        for i in range(self.L):
            X_sig[:, i + 1]          = x_aug_mean + self.gamma * sqrt_P[:, i]
            X_sig[:, i + 1 + self.L] = x_aug_mean - self.gamma * sqrt_P[:, i]
            
        # 拆分 Sigma 点: X_x (状态), X_v (过程噪声), X_w (测量噪声)
        X_x = X_sig[0, :]
        X_v = X_sig[1, :]
        X_w = X_sig[2, :]
        
        # --- 3. 时间更新 (Prediction) ---
        # 状态方程传播: x_k = alpha * x_{k-1} + sigma * v_k
        X_x_pred = self.alpha * X_x + self.sigma * X_v
        
        # 预测均值
        x_pred_mean = np.sum(self.Wm * X_x_pred)
        
        # 预测协方差
        P_pred = 0
        for i in range(2 * self.L + 1):
            diff = X_x_pred[i] - x_pred_mean
            P_pred += self.Wc[i] * (diff**2)
            
        # --- 4. 测量更新 (Correction) ---
        # 观测方程传播: y_k = beta * exp(x_k / 2) * w_k
        # 注意：使用预测后的状态 X_x_pred 和 Sigma 点中的测量噪声 X_w
        Y_sig_pred = self.beta * np.exp(X_x_pred / 2.0) * X_w
        
        # 预测观测均值
        y_pred_mean = np.sum(self.Wm * Y_sig_pred)
        
        # 观测协方差 S (Innovation Covariance)
        S = 0
        for i in range(2 * self.L + 1):
            diff = Y_sig_pred[i] - y_pred_mean
            S += self.Wc[i] * (diff**2)
            
        # 互协方差 P_xy
        P_xy = 0
        for i in range(2 * self.L + 1):
            diff_x = X_x_pred[i] - x_pred_mean
            diff_y = Y_sig_pred[i] - y_pred_mean
            P_xy += self.Wc[i] * diff_x * diff_y
            
        # --- 5. 卡尔曼增益与最终更新 ---
        # 添加极小值防止除零
        K = P_xy / (S + 1e-9)
        
        innovation = y_obs - y_pred_mean
        
        # 更新内部状态
        self.x = x_pred_mean + K * innovation
        self.P = P_pred - K * S * K
        
        return self.x, self.P

    def batch_filter(self, y_data):
        """
        批量处理整个时间序列
        """
        n = len(y_data)
        estimates = np.zeros(n)
        covariances = np.zeros(n)
        
        for k in range(n):
            estimates[k], covariances[k] = self.step(y_data[k])
            
        return estimates, covariances

# ==========================================
# 使用示例 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 1. 设置参数
    alpha = 0.91
    sigma = 1.0
    beta = 0.5
    n_steps = 500
    sv = StochasticVolatilityModel()
    # 2. 生成真实数据 (Ground Truth)sv.simulate(T)
    true_x, observed_y = sv.simulate(n_steps)

    x0_var = sigma**2 / (1 - alpha**2)
    ukf = StochasticVolatilityAugmentedUKF(
        alpha=alpha, sigma=sigma, beta=beta,
        x0_mean=0.0, x0_cov=x0_var
    )
    x_est, P_est = ukf.batch_filter(observed_y)
    
    # 4. 绘图验证
    plt.figure(figsize=(10, 6))
    
    # 上图：状态估计
    plt.subplot(2, 1, 1)
    plt.plot(true_x, 'k-', label='True Volatility', linewidth=1.5)
    plt.plot(x_est, 'r--', label='UKF Estimate', linewidth=1.5)
    std_dev = np.sqrt(P_est)
    plt.fill_between(range(n_steps), x_est - 2*std_dev, x_est + 2*std_dev, 
                     color='r', alpha=0.2, label='95% Conf')
    plt.title('Class-based Augmented UKF Result')
    plt.legend()
    plt.grid(True)
    
    # 下图：观测数据
    plt.subplot(2, 1, 2)
    plt.plot(observed_y, 'b.', label='Observed Returns', markersize=4)
    plt.title('Observations')
    plt.grid(True)
    plt.savefig('uncent-kamlam.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
