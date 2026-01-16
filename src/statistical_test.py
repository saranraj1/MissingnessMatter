import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ks_2samp

class MissingnessAuditor:
    """
    Modular suite for statistical auditing of missing data mechanisms.
    Designed for research-grade verification of MCAR vs. MAR.
    """
    
    @staticmethod
    def test_mar_mechanism(df, observed_var, missing_indicator_var):
        """
        Performs a T-test to determine if the distribution of an observed 
        variable differs based on whether another variable is missing.
        
        Significant p-values (< 0.05) provide evidence of Missing at Random (MAR).
        """
        # Create groups based on missingness
        group_missing = df[df[missing_indicator_var].isnull()][observed_var].dropna()
        group_present = df[df[missing_indicator_var].notnull()][observed_var].dropna()
        
        if len(group_missing) < 2 or len(group_present) < 2:
            return {"status": "error", "message": "Insufficient data for T-test"}
            
        t_stat, p_val = ttest_ind(group_missing, group_present, equal_var=False)
        
        return {
            "test": "Welch's T-test",
            "observed_var": observed_var,
            "missing_var": missing_indicator_var,
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 6),
            "is_significant": p_val < 0.05
        }

    @staticmethod
    def test_distribution_shift(original_series, imputed_series):
        """
        Uses the Kolmogorov-Smirnov test to quantify if imputation 
        significantly distorted the original data distribution.
        """
        ks_stat, p_val = ks_2samp(original_series.dropna(), imputed_series)
        
        return {
            "test": "Kolmogorov-Smirnov",
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_val, 6),
            "distribution_changed": p_val < 0.05
        }

def run_pima_audit(df):
    """
    Standardized audit suite for the Pima Indians dataset.
    """
    auditor = MissingnessAuditor()
    
    # Audit 1: Does Age influence Insulin missingness? (Evidence for MAR)
    age_insulin_audit = auditor.test_mar_mechanism(df, 'Age', 'Insulin')
    
    # Audit 2: Does BMI influence Insulin missingness? (Evidence for MAR)
    bmi_insulin_audit = auditor.test_mar_mechanism(df, 'BMI', 'Insulin')
    
    return [age_insulin_audit, bmi_insulin_audit]

if __name__ == "__main__":
    # Example usage for testing
    print("🔬 Statistical Audit Module Initialized.")
    
    import os
    data_path = 'data/processed/pima_exposed.csv'
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        try:
            df = pd.read_csv(data_path)
            results = run_pima_audit(df)
            print("\n📊 Audit Results:")
            for res in results:
                print(res)
        except Exception as e:
            print(f"Error running audit: {e}")
    else:
        print(f"Warning: {data_path} not found. Run the data loader first.")