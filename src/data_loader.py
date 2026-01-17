import pandas as pd
import numpy as np
import os

class PimaDataLoader:
    """
    Research-grade data loader for the Pima Indians Diabetes Dataset.
    Focuses on exposing 'Hidden Nulls' and ensuring reproducibility.
    """
    def __init__(self, raw_data_path='data/raw/pima-indians-diabetes.csv'):
        self.raw_data_path = raw_data_path
        self.columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome'
        ]
        # Biological impossibilities to be converted to NaN
        self.hidden_null_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    def load_raw_data(self):
        """Loads data from local path or handles missing file error."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(
                f"❌ Raw data not found at {self.raw_data_path}. "
                "Please ensure the CSV is placed in the data/raw/ directory."
            )
        
        # Load without headers as the raw CSV is typically headerless
        df = pd.read_csv(self.raw_data_path, names=self.columns)
        print(f"✅ Successfully loaded {len(df)} records.")
        return df

    def expose_missingness(self, df):
        """
        Converts physiological 0s into NaNs. 
        This is a critical research step to identify the missingness mechanism.
        """
        df_exposed = df.copy()
        
        # Identify how many 'zeros' we are converting
        zero_stats = {col: (df_exposed[col] == 0).sum() for col in self.hidden_null_cols}
        
        # Perform the replacement
        df_exposed[self.hidden_null_cols] = df_exposed[self.hidden_null_cols].replace(0, np.nan)
        
        print("\n📊 Missingness Audit (Hidden Nulls found):")
        for col, count in zero_stats.items():
            percentage = (count / len(df)) * 100
            print(f" - {col}: {count} zeros ({percentage:.2f}%) converted to NaN")
            
        return df_exposed

    def save_processed_data(self, df, filename='pima_exposed.csv'):
        """Saves the 'exposed' dataset for use in Notebook 01."""
        output_path = os.path.join('data/processed', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Exposed dataset saved to: {output_path}")

def run_loader_pipeline():
    """Main execution pipeline for the data loader."""
    loader = PimaDataLoader()
    
    # 1. Ingest
    raw_df = loader.load_raw_data()
    
    # 2. Audit & Transform
    exposed_df = loader.expose_missingness(raw_df)
    
    # 3. Versioning
    loader.save_processed_data(exposed_df)
    
    return exposed_df

if __name__ == "__main__":
    run_loader_pipeline()