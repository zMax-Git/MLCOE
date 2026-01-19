'''
Example to demonstrate the difference between Joseph and Standard covariance update methods
Key scenario: Small measurement noise + long time horizon to accumulate numerical errors
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
        self.Q = self.B @ self.B.T
        self.R = self.D @ self.D.T
        self.nx = A.shape[0]
        self.ny = C.shape[0]

    def simulate(self, T):
        x_true = np.zeros((T, self.nx))
        y_obs = np.zeros((T, self.ny))
        x_true[0] = np.random.multivariate_normal(np.zeros(self.nx), self.Sigma_init)
        
        nv = self.B.shape[1]
        nw = self.D.shape[1]
        
        for n in range(1, T):
            vn = np.random.randn(nv)
            x_true[n] = self.A @ x_true[n-1] + self.B @ vn
            
        for n in range(T):
            wn = np.random.randn(nw)
            y_obs[n] = self.C @ x_true[n] + self.D @ wn
            
        return x_true, y_obs

def run_kalman_filter(ssm, y_obs, P_updation='standard'):
    T = y_obs.shape[0]
    nx = ssm.nx
    
    x_filt = np.zeros((T, nx))
    P_filt = np.zeros((T, nx, nx))
    
    # Track numerical properties
    symmetry_errors = np.zeros(T)  # ||P - P^T||_F
    is_positive_definite = np.zeros(T, dtype=bool)
    min_eigenvalues = np.zeros(T)
    condition_numbers = np.zeros(T)
    
    x_pred = np.zeros(nx) 
    P_pred = ssm.Sigma_init
    
    for k in range(T):
        y_pred = ssm.C @ x_pred
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
        
        # Check numerical properties
        symmetry_errors[k] = np.linalg.norm(P_now - P_now.T, 'fro')
        eigenvals = np.linalg.eigvals(P_now)
        min_eigenvalues[k] = np.min(eigenvals)
        is_positive_definite[k] = np.all(eigenvals > 0)
        condition_numbers[k] = np.linalg.cond(P_now)
        
        x_filt[k] = x_now
        P_filt[k] = P_now
        
        x_pred = ssm.A @ x_now
        P_pred = ssm.A @ P_now @ ssm.A.T + ssm.Q 
            
    return x_filt, P_filt, {
        'symmetry_errors': symmetry_errors,
        'is_positive_definite': is_positive_definite,
        'min_eigenvalues': min_eigenvalues,
        'condition_numbers': condition_numbers
    }

if __name__ == "__main__":
    # Key parameters to create numerical instability:
    # 1. Small measurement noise (d) -> large Kalman gain -> numerical errors amplified
    # 2. Large initial uncertainty -> large initial covariance
    # 3. Long time horizon -> errors accumulate
    
    dt = 1
    sigma_var = 1000.0  # Large initial uncertainty
    b = 0.5 ## process noise
    d = 0.01 ## measurement noise
    T_steps = 500  # Long time horizon to accumulate errors

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

    D = np.eye(2) * d
    Sigma_init = np.eye(4) * sigma_var

    lgssm = LinearGaussianSSM(A, B, C, D, Sigma_init)
    x_true, y_obs = lgssm.simulate(T_steps)
    
    # Run both methods
    x_est_std, P_est_std, stats_std = run_kalman_filter(lgssm, y_obs, P_updation='standard')
    x_est_jos, P_est_jos, stats_jos = run_kalman_filter(lgssm, y_obs, P_updation='joseph')
    
    # Create comparison plots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    time = np.arange(T_steps)
    
    # Plot 1: Symmetry errors
    axes[0].semilogy(time, stats_std['symmetry_errors'], 'r-', label='Standard', linewidth=1.5)
    axes[0].semilogy(time, stats_jos['symmetry_errors'], 'b-', label='Joseph', linewidth=1.5)
    axes[0].set_ylabel('Symmetry Error\n$||P - P^T||_F$')
    axes[0].set_title('(a) Symmetry Error Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.95, '(a)', transform=axes[0].transAxes, fontsize=12, 
                 fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Minimum eigenvalues (should be > 0 for positive definiteness)
    axes[1].plot(time, stats_std['min_eigenvalues'], 'r-', label='Standard', linewidth=1.5)
    axes[1].plot(time, stats_jos['min_eigenvalues'], 'b-', label='Joseph', linewidth=1.5)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero line')
    axes[1].set_ylabel('Minimum Eigenvalue')
    axes[1].set_title('(b) Minimum Eigenvalue (Positive Definiteness Check)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.02, 0.95, '(b)', transform=axes[1].transAxes, fontsize=12, 
                 fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Condition numbers
    axes[2].semilogy(time, stats_std['condition_numbers'], 'r-', label='Standard', linewidth=1.5)
    axes[2].semilogy(time, stats_jos['condition_numbers'], 'b-', label='Joseph', linewidth=1.5)
    axes[2].set_ylabel('Condition Number')
    axes[2].set_title('(c) Condition Number of Covariance Matrix')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.02, 0.95, '(c)', transform=axes[2].transAxes, fontsize=12, 
                 fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Position estimates comparison
    axes[3].plot(time, x_true[:, 0], 'k-', label='True $x_t$', linewidth=1.5)
    axes[3].plot(time, x_est_std[:, 0], 'r--', label='Standard KF', linewidth=1.5, alpha=0.7)
    axes[3].plot(time, x_est_jos[:, 0], 'b--', label='Joseph KF', linewidth=1.5, alpha=0.7)
    axes[3].set_xlabel('Time Step')
    axes[3].set_ylabel('Position X')
    axes[3].set_title('(d) State Estimates Comparison')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].text(0.02, 0.95, '(d)', transform=axes[3].transAxes, fontsize=12, 
                 fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Joseph vs Standard Covariance Update\n'
                 f'Parameters: $\\sigma_0$={sigma_var}, $b$={b}, $d$={d}, $T$={T_steps}', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('Joseph-cov-comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NUMERICAL PROPERTIES COMPARISON")
    print("="*60)
    print(f"\nSymmetry Error (final):")
    print(f"  Standard: {stats_std['symmetry_errors'][-1]:.2e}")
    print(f"  Joseph:   {stats_jos['symmetry_errors'][-1]:.2e}")
    
    print(f"\nPositive Definite Violations:")
    violations_std = np.sum(~stats_std['is_positive_definite'])
    violations_jos = np.sum(~stats_jos['is_positive_definite'])
    print(f"  Standard: {violations_std}/{T_steps} steps")
    print(f"  Joseph:   {violations_jos}/{T_steps} steps")
    
    print(f"\nMinimum Eigenvalue (final):")
    print(f"  Standard: {stats_std['min_eigenvalues'][-1]:.6f}")
    print(f"  Joseph:   {stats_jos['min_eigenvalues'][-1]:.6f}")
    
    print(f"\nCondition Number (final):")
    print(f"  Standard: {stats_std['condition_numbers'][-1]:.2e}")
    print(f"  Joseph:   {stats_jos['condition_numbers'][-1]:.2e}")
    print("="*60)