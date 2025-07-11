import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VasicekModel:
    """
    Vasicek Interest Rate Model Implementation
    
    Model: dr(t) = α(θ - r(t))dt + σdW(t)
    
    Parameters:
    - α (alpha): Speed of mean reversion
    - θ (theta): Long-term mean level
    - σ (sigma): Volatility parameter
    """
    
    def __init__(self, alpha=0.1, theta=0.05, sigma=0.02):
        self.alpha = alpha  # Speed of mean reversion
        self.theta = theta  # Long-term mean
        self.sigma = sigma  # Volatility
        self.dt = 1/252     # Daily time step (252 trading days)
        
    def simulate_path(self, r0, T, n_steps):
        """
        Simulate interest rate path using Euler-Maruyama method
        
        Parameters:
        - r0: Initial interest rate
        - T: Time horizon (in years)
        - n_steps: Number of simulation steps
        
        Returns:
        - Array of simulated interest rates
        """
        dt = T / n_steps
        rates = np.zeros(n_steps + 1)
        rates[0] = r0
        
        # Generate random shocks
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        
        # Simulate using Euler-Maruyama
        for i in range(n_steps):
            dr = self.alpha * (self.theta - rates[i]) * dt + self.sigma * dW[i]
            rates[i + 1] = rates[i] + dr
            
        return rates
    
    def monte_carlo_simulation(self, r0, T, n_steps, n_simulations=1000):
        """
        Monte Carlo simulation for path generation
        
        Parameters:
        - r0: Initial interest rate
        - T: Time horizon (in years)
        - n_steps: Number of steps per path
        - n_simulations: Number of simulation paths
        
        Returns:
        - Matrix of simulated paths (n_simulations x n_steps+1)
        """
        np.random.seed(42)  # For reproducibility
        
        paths = np.zeros((n_simulations, n_steps + 1))
        
        for i in range(n_simulations):
            paths[i, :] = self.simulate_path(r0, T, n_steps)
            
        return paths
    
    def analytical_mean(self, r0, t):
        """
        Analytical solution for the mean of r(t)
        
        E[r(t)] = θ + (r0 - θ) * exp(-α*t)
        """
        return self.theta + (r0 - self.theta) * np.exp(-self.alpha * t)
    
    def analytical_variance(self, t):
        """
        Analytical solution for the variance of r(t)
        
        Var[r(t)] = σ²/(2α) * (1 - exp(-2*α*t))
        """
        return (self.sigma**2) / (2 * self.alpha) * (1 - np.exp(-2 * self.alpha * t))
    
    def bond_price(self, r0, T, maturity):
        """
        Analytical bond pricing formula for Vasicek model
        
        P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
        
        where:
        - B(t,T) = (1 - exp(-α*(T-t))) / α
        - A(t,T) = exp((θ - σ²/(2*α²)) * (B(t,T) - (T-t)) - σ²/(4*α) * B(t,T)²)
        """
        tau = maturity - T  # Time to maturity
        
        if tau <= 0:
            return 1.0
        
        # Calculate B(t,T)
        if self.alpha == 0:
            B = tau
        else:
            B = (1 - np.exp(-self.alpha * tau)) / self.alpha
        
        # Calculate A(t,T)
        if self.alpha == 0:
            A_exponent = (self.theta - self.sigma**2/2) * tau - (self.sigma**2 * tau**3) / 6
        else:
            A_exponent = ((self.theta - self.sigma**2/(2*self.alpha**2)) * 
                         (B - tau) - (self.sigma**2 * B**2) / (4*self.alpha))
        
        A = np.exp(A_exponent)
        
        # Bond price
        return A * np.exp(-B * r0)
    
    def yield_curve(self, r0, maturities):
        """
        Calculate yield curve for given maturities
        
        Parameters:
        - r0: Current short rate
        - maturities: Array of maturities (in years)
        
        Returns:
        - Array of yields
        """
        yields = []
        
        for T in maturities:
            if T <= 0:
                yields.append(r0)
            else:
                bond_price = self.bond_price(r0, 0, T)
                yield_rate = -np.log(bond_price) / T
                yields.append(yield_rate)
        
        return np.array(yields)
    
    def log_likelihood(self, params, rates, dt):
        """
        Log-likelihood function for parameter estimation
        
        Parameters:
        - params: [alpha, theta, sigma]
        - rates: Observed interest rate time series
        - dt: Time step
        
        Returns:
        - Negative log-likelihood
        """
        alpha, theta, sigma = params
        
        if alpha <= 0 or sigma <= 0:
            return np.inf
        
        n = len(rates) - 1
        log_likelihood = 0
        
        for i in range(n):
            # Conditional mean and variance
            mean = rates[i] * np.exp(-alpha * dt) + theta * (1 - np.exp(-alpha * dt))
            variance = (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * dt))
            
            if variance <= 0:
                return np.inf
            
            # Log-likelihood contribution
            log_likelihood += -0.5 * np.log(2 * np.pi * variance)
            log_likelihood += -0.5 * (rates[i+1] - mean)**2 / variance
        
        return -log_likelihood
    
    def estimate_parameters(self, rates, dt=1/252):
        """
        Parameter estimation using Maximum Likelihood
        
        Parameters:
        - rates: Observed interest rate time series
        - dt: Time step (default: daily)
        
        Returns:
        - Estimated parameters [alpha, theta, sigma]
        """
        # Initial guess
        initial_guess = [0.1, np.mean(rates), np.std(rates)]
        
        # Bounds for parameters
        bounds = [(0.001, 10), (0.001, 0.5), (0.001, 1)]
        
        # Optimize
        result = minimize(
            self.log_likelihood,
            initial_guess,
            args=(rates, dt),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            self.alpha, self.theta, self.sigma = result.x
            return result.x
        else:
            print("Parameter estimation failed!")
            return None
    
    def model_summary(self):
        """
        Print model parameters summary
        """
        print("=" * 50)
        print("VASICEK MODEL PARAMETERS")
        print("=" * 50)
        print(f"Speed of Mean Reversion (α): {self.alpha:.4f}")
        print(f"Long-term Mean (θ):          {self.theta:.4f}")
        print(f"Volatility (σ):              {self.sigma:.4f}")
        print(f"Half-life:                   {np.log(2)/self.alpha:.2f} years")
        print("=" * 50)

class VasicekAnalyzer:
    """
    Analysis and validation tools for Vasicek model
    """
    
    def __init__(self, model):
        self.model = model
    
    def model_validation(self, observed_rates, dt=1/252):
        """
        Validate model against observed data
        
        Parameters:
        - observed_rates: Historical interest rate data
        - dt: Time step
        
        Returns:
        - Validation metrics
        """
        n = len(observed_rates) - 1
        residuals = []
        
        for i in range(n):
            # Predicted rate
            predicted_mean = (observed_rates[i] * np.exp(-self.model.alpha * dt) + 
                            self.model.theta * (1 - np.exp(-self.model.alpha * dt)))
            
            # Residual
            residual = observed_rates[i+1] - predicted_mean
            residuals.append(residual)
        
        residuals = np.array(residuals)
        
        # Calculate metrics
        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(mse)
        
        # Normality test (Jarque-Bera)
        from scipy.stats import jarque_bera
        jb_stat, jb_pvalue = jarque_bera(residuals)
        
        validation_metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'residuals': residuals,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue
        }
        
        return validation_metrics
    
    def generate_model_data(self, r0, T=5, n_steps=1000, n_simulations=1000):
        """
        Generate comprehensive model data for visualization
        
        Returns:
        - Dictionary with all model outputs
        """
        # Monte Carlo simulation
        paths = self.model.monte_carlo_simulation(r0, T, n_steps, n_simulations)
        
        # Time grid
        time_grid = np.linspace(0, T, n_steps + 1)
        
        # Analytical solutions
        analytical_mean = np.array([self.model.analytical_mean(r0, t) for t in time_grid])
        analytical_std = np.array([np.sqrt(self.model.analytical_variance(t)) for t in time_grid])
        
        # Yield curve
        maturities = np.linspace(0.1, 10, 50)
        yields = self.model.yield_curve(r0, maturities)
        
        # Bond prices
        bond_prices = np.array([self.model.bond_price(r0, 0, T) for T in maturities])
        
        return {
            'paths': paths,
            'time_grid': time_grid,
            'analytical_mean': analytical_mean,
            'analytical_std': analytical_std,
            'maturities': maturities,
            'yields': yields,
            'bond_prices': bond_prices,
            'r0': r0,
            'T': T
        }

# Example usage and testing
if __name__ == "__main__":
    # Create Vasicek model instance
    vasicek = VasicekModel(alpha=0.2, theta=0.05, sigma=0.02)
    
    # Display model parameters
    vasicek.model_summary()
    
    # Load your collected interest rate data
    try:
        # Load repo rate data from your data collection
        repo_data = pd.read_csv('data/processed/combined_interest_rates.csv')
        repo_data['date'] = pd.to_datetime(repo_data['date'])
        
        # Use repo rates for parameter estimation
        repo_rates = repo_data['repo_rate'].values / 100  # Convert to decimal
        
        print("\n🔍 Estimating parameters from repo rate data...")
        estimated_params = vasicek.estimate_parameters(repo_rates)
        
        if estimated_params is not None:
            print(f"✅ Parameters estimated successfully!")
            vasicek.model_summary()
            
            # Create analyzer
            analyzer = VasicekAnalyzer(vasicek)
            
            # Generate model data for visualization
            current_rate = repo_rates[-1]  # Use latest rate as r0
            model_data = analyzer.generate_model_data(current_rate)
            
            print(f"\n📊 Model data generated for visualization")
            print(f"Current rate (r0): {current_rate*100:.3f}%")
            print(f"Generated {len(model_data['paths'])} simulation paths")
            
            # Save model data for notebook visualization
            import pickle
            with open('data/processed/vasicek_model_data.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            print("✅ Model data saved to 'data/processed/vasicek_model_data.pkl'")
            
        else:
            print("❌ Parameter estimation failed!")
            
    except FileNotFoundError:
        print("❌ Interest rate data not found. Please run data collection first.")
        
        # Use example data
        print("\n🔧 Using example data for demonstration...")
        current_rate = 0.065  # 6.5%
        
        analyzer = VasicekAnalyzer(vasicek)
        model_data = analyzer.generate_model_data(current_rate)
        
        # Save example model data
        import pickle
        with open('data/processed/vasicek_model_data.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Example model data saved for visualization")
