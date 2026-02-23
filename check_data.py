import pandas as pd


try:
    df = pd.read_csv('mmu_pd_features.csv')
    print("--- Feature Means by Category ---")
    print(df.groupby('label').mean())
    print("\n--- Sample Counts ---")
    print(df['label'].value_counts())
except FileNotFoundError:
    print("Error: mmu_pd_features.csv not found. Run preprocess.py first.")