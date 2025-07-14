import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def show_data_summary(df):
    print("First few rows of the dataset:")
    print(df.head())
    print("\nMissing values in each column:")
    print(df.isnull().sum())

def visualize_data(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], bins=50, kde=True)
    plt.title('Distribution of Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.show()

    sns.pairplot(df, hue='smoker')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
