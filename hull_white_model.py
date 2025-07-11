import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HullWhiteModel:
    """
    Hull-White Interest Rate Model Implementation
    
    Model: dr(t) = (θ(t) - αr(t))dt + σdW(t)
    
    Parameters:
    - α (alpha): Speed of mean reversion
    - θ(t) (theta): Time-dependent long-term mean level
    - σ (sigma): Volatility parameter
    """
    
    def __init__(self, alpha=0.1, sigma=0.02, theta_func=None):
        self.alpha = alpha  # Speed of mean reversion
        self.sigma = sigma  # Volatility
        self.theta_func = theta_func  # Time-dependent theta function
        self.dt = 1/252     # Daily time step (252 trading days)
        
        # Default constant theta if no function provided
        if self.theta_func is None:
            self.theta_func = lambda t: 0.05  # 5% constant
    
    def theta(self, t):
        """Time-dependent theta function"""
        if callable(self.theta_func):
            return self.theta_func(t)
        else:
            return self.theta_func  # Constant value
    
    def simulate_path(self, r0, T, n_steps, theta_values=None):
        """
        Simulate interest rate path using Euler-Maruyama method
        
        Parameters:
        - r0: Initial interest rate
        - T: Time horizon (in years)
        - n_steps: Number of simulation steps
        - theta_values: Array of theta values (optional)
        
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
            t = i * dt
            
            # Get theta value for current time
            if theta_values is not None:
                theta_t = theta_values[i]
            else:
                theta_t = self.theta(t)
            
            # Hull-White SDE: dr = (theta(t) - alpha*r(t))dt + sigma*dW
            dr = (theta_t - self.alpha * rates[i]) * dt + self.sigma * dW[i]
            rates[i + 1] = rates[i] + dr
            
        return rates
    
    def monte_carlo_simulation(self, r0, T, n_steps, n_simulations=1000, theta_values=None):
        """
        Monte Carlo simulation for path generation
        
        Parameters:
        - r0: Initial interest rate
        - T: Time horizon (in years)
        - n_steps: Number of steps per path
        - n_simulations: Number of simulation paths
        - theta_values: Array of theta values for time-dependent calibration
        
        Returns:
        - Matrix of simulated paths (n_simulations x n_steps+1)
        """
        np.random.seed(42)  # For reproducibility
        
        paths = np.zeros((n_simulations, n_steps + 1))
        
        for i in range(n_simulations):
            paths[i, :] = self.simulate_path(r0, T, n_steps, theta_values)
            
        return paths
    
    def analytical_mean(self, r0, t, theta_values=None):
        """
        Analytical solution for the mean of r(t) in Hull-White model
        
        E[r(t)] = r0*exp(-α*t) + ∫[0,t] θ(s)*α*exp(-α*(t-s))ds
        
        For constant θ: E[r(t)] = θ + (r0 - θ) * exp(-α*t)
        """
        if theta_values is None:
            # Constant theta case (same as Vasicek)
            theta_const = self.theta(0)
            return theta_const + (r0 - theta_const) * np.exp(-self.alpha * t)
        else:
            # Time-dependent theta case (numerical integration)
            # This is a simplified approximation
            theta_mean = np.mean(theta_values)
            return theta_mean + (r0 - theta_mean) * np.exp(-self.alpha * t)
    
    def analytical_variance(self, t):
        """
        Analytical solution for the variance of r(t)
        
        Var[r(t)] = σ²/(2α) * (1 - exp(-2*α*t))
        """
        return (self.sigma**2) / (2 * self.alpha) * (1 - np.exp(-2 * self.alpha * t))
    
    def bond_price(self, r0, t, T, theta_values=None):
        """
        Analytical bond pricing formula for Hull-White model
        
        P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
        
        where:
        - B(t,T) = (1 - exp(-α*(T-t))) / α
        - A(t,T) involves integration of θ(s) over [t,T]
        """
        tau = T - t  # Time to maturity
        
        if tau <= 0:
            return 1.0
        
        # Calculate B(t,T)
        if self.alpha == 0:
            B = tau
        else:
            B = (1 - np.exp(-self.alpha * tau)) / self.alpha
        
        # Calculate A(t,T) - simplified for constant theta
        if theta_values is None:
            theta_const = self.theta(t)
            A_exponent = ((theta_const - self.sigma**2/(2*self.alpha**2)) * 
                         (B - tau) - (self.sigma**2 * B**2) / (4*self.alpha))
        else:
            # More complex calculation for time-dependent theta
            # This is a simplified approximation
            theta_mean = np.mean(theta_values)
            A_exponent = ((theta_mean - self.sigma**2/(2*self.alpha**2)) * 
                         (B - tau) - (self.sigma**2 * B**2) / (4*self.alpha))
        
        A = np.exp(A_exponent)
        
        # Bond price
        return A * np.exp(-B * r0)
    
    def yield_curve(self, r0, maturities, theta_values=None):
        """
        Calculate yield curve for given maturities
        
        Parameters:
        - r0: Current short rate
        - maturities: Array of maturities (in years)
        - theta_values: Array of theta values
        
        Returns:
        - Array of yields
        """
        yields = []
        
        for T in maturities:
            if T <= 0:
                yields.append(r0)
            else:
                bond_price = self.bond_price(r0, 0, T, theta_values)
                yield_rate = -np.log(bond_price) / T
                yields.append(yield_rate)
        
        return np.array(yields)
    
    def calibrate_theta_to_yield_curve(self, market_yields, maturities, r0):
        """
        Calibrate time-dependent theta to fit market yield curve
        
        Parameters:
        - market_yields: Observed market yields
        - maturities: Corresponding maturities
        - r0: Current short rate
        
        Returns:
        - Calibrated theta function
        """
        def objective(theta_params):
            """Objective function to minimize yield curve fitting error"""
            
            # Create piecewise constant theta function
            def theta_func(t):
                for i, T in enumerate(maturities[:-1]):
                    if t <= T:
                        return theta_params[i]
                return theta_params[-1]
            
            # Calculate model yields
            model_yields = []
            for T in maturities:
                bond_price = self.bond_price(r0, 0, T)
                model_yield = -np.log(bond_price) / T
                model_yields.append(model_yield)
            
            # Return sum of squared errors
            return np.sum((np.array(model_yields) - market_yields)**2)
        
        # Initial guess for theta parameters
        initial_guess = [0.05] * len(maturities)
        
        # Optimize
        result = minimize(objective, initial_guess, method='L-BFGS-B',
                         bounds=[(0.01, 0.20)] * len(maturities))
        
        if result.success:
            # Create calibrated theta function
            theta_params = result.x
            
            def calibrated_theta(t):
                for i, T in enumerate(maturities[:-1]):
                    if t <= T:
                        return theta_params[i]
                return theta_params[-1]
            
            self.theta_func = calibrated_theta
            return calibrated_theta
        else:
            print("Theta calibration failed!")
            return None
    
    def estimate_parameters(self, rates, dt=1/252):
        """
        Parameter estimation using Maximum Likelihood
        
        Parameters:
        - rates: Observed interest rate time series
        - dt: Time step (default: daily)
        
        Returns:
        - Estimated parameters [alpha, sigma, theta_mean]
        """
        
        def log_likelihood(params):
            alpha, sigma, theta_mean = params
            
            if alpha <= 0 or sigma <= 0:
                return np.inf
            
            n = len(rates) - 1
            log_likelihood = 0
            
            for i in range(n):
                # Hull-White with constant theta reduces to Vasicek
                mean = rates[i] * np.exp(-alpha * dt) + theta_mean * (1 - np.exp(-alpha * dt))
                variance = (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * dt))
                
                if variance <= 0:
                    return np.inf
                
                # Log-likelihood contribution
                log_likelihood += -0.5 * np.log(2 * np.pi * variance)
                log_likelihood += -0.5 * (rates[i+1] - mean)**2 / variance
            
            return -log_likelihood
        
        # Initial guess
        initial_guess = [0.1, np.std(rates), np.mean(rates)]
        
        # Bounds for parameters
        bounds = [(0.001, 10), (0.001, 1), (0.001, 0.5)]
        
        # Optimize
        result = minimize(log_likelihood, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            self.alpha, self.sigma, theta_mean = result.x
            self.theta_func = lambda t: theta_mean  # Constant theta
            return result.x
        else:
            print("Parameter estimation failed!")
            return None
    
    def model_summary(self):
        """
        Print model parameters summary
        """
        print("=" * 50)
        print("HULL-WHITE MODEL PARAMETERS")
        print("=" * 50)
        print(f"Speed of Mean Reversion (α): {self.alpha:.4f}")
        print(f"Volatility (σ):              {self.sigma:.4f}")
        print(f"Theta Function:              {'Time-dependent' if callable(self.theta_func) else 'Constant'}")
        print(f"Half-life:                   {np.log(2)/self.alpha:.2f} years")
        print("=" * 50)

class HullWhiteAnalyzer:
    """
    Analysis and validation tools for Hull-White model
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
            # Predicted rate (assuming constant theta for simplicity)
            theta_t = self.model.theta(i * dt)
            predicted_mean = (observed_rates[i] * np.exp(-self.model.alpha * dt) + 
                            theta_t * (1 - np.exp(-self.model.alpha * dt)))
            
            # Residual
            residual = observed_rates[i+1] - predicted_mean
            residuals.append(residual)
        
        residuals = np.array(residuals)
        
        # Calculate metrics
        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(mse)
        
        # Normality test
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
    
    def generate_model_data(self, r0, T=5, n_steps=1000, n_simulations=1000, theta_values=None):
        """
        Generate comprehensive model data for visualization
        
        Returns:
        - Dictionary with all model outputs
        """
        # Monte Carlo simulation
        paths = self.model.monte_carlo_simulation(r0, T, n_steps, n_simulations, theta_values)
        
        # Time grid
        time_grid = np.linspace(0, T, n_steps + 1)
        
        # Analytical solutions
        analytical_mean = np.array([self.model.analytical_mean(r0, t, theta_values) for t in time_grid])
        analytical_std = np.array([np.sqrt(self.model.analytical_variance(t)) for t in time_grid])
        
        # Yield curve
        maturities = np.linspace(0.1, 10, 50)
        yields = self.model.yield_curve(r0, maturities, theta_values)
        
        # Bond prices
        bond_prices = np.array([self.model.bond_price(r0, 0, T, theta_values) for T in maturities])
        
        return {
            'paths': paths,
            'time_grid': time_grid,
            'analytical_mean': analytical_mean,
            'analytical_std': analytical_std,
            'maturities': maturities,
            'yields': yields,
            'bond_prices': bond_prices,
            'theta_values': theta_values,
            'r0': r0,
            'T': T
        }

# Example usage and testing
if __name__ == "__main__":
    # Create Hull-White model instance
    hull_white = HullWhiteModel(alpha=0.2, sigma=0.02)
    
    # Display model parameters
    hull_white.model_summary()
    
    # Load your collected interest rate data
    try:
        # Load repo rate data from your data collection
        repo_data = pd.read_csv('data/processed/combined_interest_rates.csv')
        repo_data['date'] = pd.to_datetime(repo_data['date'])
        
        # Use repo rates for parameter estimation
        repo_rates = repo_data['repo_rate'].values / 100  # Convert to decimal
        
        print("\n🔍 Estimating parameters from repo rate data...")
        estimated_params = hull_white.estimate_parameters(repo_rates)
        
        if estimated_params is not None:
            print(f"✅ Parameters estimated successfully!")
            hull_white.model_summary()
            
            # Create analyzer
            analyzer = HullWhiteAnalyzer(hull_white)
            
            # Generate model data for visualization
            current_rate = repo_rates[-1]  # Use latest rate as r0
            model_data = analyzer.generate_model_data(current_rate)
            
            print(f"\n📊 Model data generated for visualization")
            print(f"Current rate (r0): {current_rate*100:.3f}%")
            print(f"Generated {len(model_data['paths'])} simulation paths")
            
            # Try yield curve calibration with sample data
            print("\n🎯 Attempting yield curve calibration...")
            sample_maturities = np.array([0.25, 0.5, 1, 2, 5, 10])
            sample_yields = np.array([0.04, 0.045, 0.05, 0.055, 0.06, 0.065])
            
            calibrated_theta = hull_white.calibrate_theta_to_yield_curve(
                sample_yields, sample_maturities, current_rate
            )
            
            if calibrated_theta is not None:
                print("✅ Yield curve calibration successful!")
                
                # Generate calibrated model data
                theta_values = np.array([calibrated_theta(t) for t in model_data['time_grid']])
                calibrated_model_data = analyzer.generate_model_data(
                    current_rate, theta_values=theta_values
                )
                
                # Save both model datasets
                import pickle
                with open('data/processed/hull_white_model_data.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                
                with open('data/processed/hull_white_calibrated_data.pkl', 'wb') as f:
                    pickle.dump(calibrated_model_data, f)
                
                print("✅ Both model datasets saved for visualization")
            else:
                # Save basic model data
                import pickle
                with open('data/processed/hull_white_model_data.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                
                print("✅ Basic model data saved for visualization")
            
        else:
            print("❌ Parameter estimation failed!")
            
    except FileNotFoundError:
        print("❌ Interest rate data not found. Please run data collection first.")
        
        # Use example data
        print("\n🔧 Using example data for demonstration...")
        current_rate = 0.065  # 6.5%
        
        analyzer = HullWhiteAnalyzer(hull_white)
        model_data = analyzer.generate_model_data(current_rate)
        
        # Save example model data
        import pickle
        with open('data/processed/hull_white_model_data.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Example model data saved for visualization")
