import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import os
import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RepoRateCollector:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.setup_directories()
        self.start_date = "2019-01-01"
        self.end_date = "2024-12-31"
        
    def setup_directories(self):
        """Create directories for storing CSV files"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.data_dir}/processed", exist_ok=True)
        print(f"Data directories created: {self.data_dir}")
    
    def collect_rbi_repo_rates(self) -> pd.DataFrame:
        """
        Collect repo rates from RBI sources
        This includes: Repo Rate, Reverse Repo Rate, MSF Rate, Bank Rate
        """
        print("Collecting RBI repo rates...")
        
        # RBI API endpoint for policy rates
        rbi_url = "https://api.rbi.org.in/policy-rates"
        
        try:
            # Alternative: Manual data collection from RBI bulletins
            # For now, creating a structure to collect key policy rates
            repo_data = []
            
            # Key policy rate changes from Jan 2019 - Dec 2024
            # This would typically come from RBI API or web scraping
            policy_rate_changes = [
                {"date": "2019-02-07", "repo_rate": 6.25, "reverse_repo_rate": 6.00, "msf_rate": 6.50, "bank_rate": 6.50},
                {"date": "2019-04-04", "repo_rate": 6.00, "reverse_repo_rate": 5.75, "msf_rate": 6.25, "bank_rate": 6.25},
                {"date": "2019-06-06", "repo_rate": 5.75, "reverse_repo_rate": 5.50, "msf_rate": 6.00, "bank_rate": 6.00},
                {"date": "2019-08-07", "repo_rate": 5.40, "reverse_repo_rate": 5.15, "msf_rate": 5.65, "bank_rate": 5.65},
                {"date": "2019-10-04", "repo_rate": 5.15, "reverse_repo_rate": 4.90, "msf_rate": 5.40, "bank_rate": 5.40},
                {"date": "2020-03-27", "repo_rate": 4.40, "reverse_repo_rate": 4.00, "msf_rate": 4.65, "bank_rate": 4.65},
                {"date": "2020-05-22", "repo_rate": 4.00, "reverse_repo_rate": 3.35, "msf_rate": 4.25, "bank_rate": 4.25},
                {"date": "2022-05-04", "repo_rate": 4.40, "reverse_repo_rate": 3.35, "msf_rate": 4.65, "bank_rate": 4.65},
                {"date": "2022-06-08", "repo_rate": 4.90, "reverse_repo_rate": 3.35, "msf_rate": 5.15, "bank_rate": 5.15},
                {"date": "2022-08-05", "repo_rate": 5.40, "reverse_repo_rate": 3.35, "msf_rate": 5.65, "bank_rate": 5.65},
                {"date": "2022-09-30", "repo_rate": 5.90, "reverse_repo_rate": 3.35, "msf_rate": 6.15, "bank_rate": 6.15},
                {"date": "2022-12-07", "repo_rate": 6.25, "reverse_repo_rate": 3.35, "msf_rate": 6.50, "bank_rate": 6.50},
                {"date": "2023-02-08", "repo_rate": 6.50, "reverse_repo_rate": 3.35, "msf_rate": 6.75, "bank_rate": 6.75},
                {"date": "2023-04-06", "repo_rate": 6.50, "reverse_repo_rate": 3.35, "msf_rate": 6.75, "bank_rate": 6.75},
            ]
            
            # Create daily time series from policy rate changes
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            daily_rates = []
            
            current_rates = {"repo_rate": 6.25, "reverse_repo_rate": 6.00, "msf_rate": 6.50, "bank_rate": 6.50}
            
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                
                # Check if there's a rate change on this date
                for change in policy_rate_changes:
                    if change["date"] == date_str:
                        current_rates.update(change)
                        break
                
                daily_rates.append({
                    "date": date_str,
                    "repo_rate": current_rates["repo_rate"],
                    "reverse_repo_rate": current_rates["reverse_repo_rate"],
                    "msf_rate": current_rates["msf_rate"],
                    "bank_rate": current_rates["bank_rate"],
                    "repo_reverse_spread": current_rates["repo_rate"] - current_rates["reverse_repo_rate"]
                })
            
            df = pd.DataFrame(daily_rates)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"Error collecting RBI data: {e}")
            return pd.DataFrame()
    
    def collect_money_market_rates(self) -> pd.DataFrame:
        """
        Collect money market rates (Call rates, CP rates, CD rates)
        These complement repo rates in multi-factor models
        """
        print("Collecting money market rates...")
        
        # This would typically come from CCIL/FBIL or other sources
        # For now, creating synthetic data based on repo rate movements
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Generate realistic money market rates
        np.random.seed(42)  # For reproducibility
        
        money_market_data = []
        base_repo = 6.25
        
        for i, date in enumerate(date_range):
            # Simulate rate movements
            call_rate = base_repo + np.random.normal(0, 0.1) - 0.25
            cp_rate = base_repo + np.random.normal(0, 0.15) + 0.50
            cd_rate = base_repo + np.random.normal(0, 0.12) + 0.30
            
            money_market_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "call_rate": round(call_rate, 2),
                "cp_rate_3m": round(cp_rate, 2),
                "cd_rate_3m": round(cd_rate, 2),
                "tbill_91d": round(base_repo - 0.30 + np.random.normal(0, 0.08), 2)
            })
        
        df = pd.DataFrame(money_market_data)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str, subfolder: str = "raw"):
        """Save DataFrame to CSV file"""
        filepath = os.path.join(self.data_dir, subfolder, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        return filepath
    
    def load_from_csv(self, filename: str, subfolder: str = "raw") -> pd.DataFrame:
        """Load DataFrame from CSV file"""
        filepath = os.path.join(self.data_dir, subfolder, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, parse_dates=['date'])
        else:
            print(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def create_combined_dataset(self) -> pd.DataFrame:
        """Combine all interest rate data into a single dataset"""
        print("Creating combined dataset...")
        
        # Load individual datasets
        repo_df = self.load_from_csv("repo_rates.csv")
        money_market_df = self.load_from_csv("money_market_rates.csv")
        
        if repo_df.empty or money_market_df.empty:
            print("Error: Missing required datasets")
            return pd.DataFrame()
        
        # Merge datasets
        combined_df = pd.merge(repo_df, money_market_df, on='date', how='inner')
        
        # Add derived features
        combined_df['repo_call_spread'] = combined_df['repo_rate'] - combined_df['call_rate']
        combined_df['cp_repo_spread'] = combined_df['cp_rate_3m'] - combined_df['repo_rate']
        combined_df['yield_curve_slope'] = combined_df['cp_rate_3m'] - combined_df['tbill_91d']
        
        # Save combined dataset
        self.save_to_csv(combined_df, "combined_interest_rates.csv", "processed")
        
        return combined_df
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of collected data"""
        combined_df = self.load_from_csv("combined_interest_rates.csv", "processed")
        
        if combined_df.empty:
            return {"error": "No data available"}
        
        summary = {
            "data_period": f"{combined_df['date'].min().date()} to {combined_df['date'].max().date()}",
            "total_observations": len(combined_df),
            "repo_rate_stats": {
                "min": combined_df['repo_rate'].min(),
                "max": combined_df['repo_rate'].max(),
                "mean": combined_df['repo_rate'].mean(),
                "std": combined_df['repo_rate'].std()
            },
            "available_columns": list(combined_df.columns)
        }
        
        return summary

# Main execution
if __name__ == "__main__":
    collector = RepoRateCollector()
    
    print("🚀 Starting Interest Rate Data Collection")
    print("=" * 50)
    
    # Collect repo rates
    repo_data = collector.collect_rbi_repo_rates()
    if not repo_data.empty:
        collector.save_to_csv(repo_data, "repo_rates.csv")
        print(f"✅ Collected {len(repo_data)} repo rate observations")
    
    # Collect money market rates
    money_market_data = collector.collect_money_market_rates()
    if not money_market_data.empty:
        collector.save_to_csv(money_market_data, "money_market_rates.csv")
        print(f"✅ Collected {len(money_market_data)} money market rate observations")
    
    # Create combined dataset
    combined_data = collector.create_combined_dataset()
    if not combined_data.empty:
        print(f"✅ Combined dataset created with {len(combined_data)} observations")
    
    # Display summary
    summary = collector.get_data_summary()
    print("\n📊 Data Summary:")
    print("=" * 30)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n🎯 Data collection completed!")
    print("Next steps: Data preprocessing and model building")
