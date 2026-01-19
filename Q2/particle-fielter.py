'''
Author: Jiajia HUANG
Date: 2026-01-01 14:25:48
LastEditors: Jiajia HUANG
LastEditTime: 2026-01-14 20:30:37
Description: fill the seacription
'''
import numpy as np
import matplotlib.pyplot as plt
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


# Alternative implementation using auxiliary particle filter for comparison

def run_particle_filter(observations, alpha=0.91, sigma=1.0, beta=0.5, n_particles=1000):
    n = len(observations)
    filtered_states = np.zeros(n)

    # Initialize particles
    particles = np.random.normal(0, sigma/np.sqrt(1-alpha**2), n_particles)
    weights = np.ones(n_particles) / n_particles
    
    for t in range(n):
        # Predict
        particles = alpha * particles + sigma * np.random.normal(size=n_particles)
        
        # Update weights based on observation likelihood
        # p(y|x) ∝ exp(-y^2/(2*beta^2*exp(x)))
        log_weights = -observations[t]**2 / (2 * beta**2 * np.exp(particles))
        log_weights -= np.max(log_weights)  # Numerical stability
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        
        # Resample
        indices = np.random.choice(n_particles, n_particles, p=weights, replace=True)
        particles = particles[indices]
        weights = np.ones(n_particles) / n_particles
        
        # Store filtered mean
        filtered_states[t] = np.mean(particles)
    
    return filtered_states

if __name__ == "__main__":
    np.random.seed(42)
    T = 500
    alpha, sigma, beta = 0.91, 1.0, 0.5
    X_true, Y = generate_data(alpha, sigma, beta, T)
    X_pf = run_particle_filter(Y, alpha, sigma, beta)
    plt.plot(X_true, 'g-', label='True X', alpha=0.7, linewidth=1.5)
    plt.plot(X_pf, 'b--', label='PF estimate', alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('particle-fielter.png', dpi=300, bbox_inches='tight')
    plt.show()