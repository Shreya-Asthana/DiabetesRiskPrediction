import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Create plots directory if it doesn't exist
plots_dir = "ml/plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def bar(df, col):
    """plots bar graph"""
    plt.bar(df[col], 1)
    plt.title(col)
    plt.savefig(f"{plots_dir}/{col}_bar.png")  # Save in plots directory
    plt.show()

def hist(df, col):
    """plots histogram"""
    plt.hist(df[col], bins=10)
    plt.title(col)
    plt.savefig(f"{plots_dir}/{col}_hist.png")  # Save in plots directory
    plt.show()

def box_plot(df, col):
    """plots box plot"""
    plt.boxplot(df[col])
    plt.title(col)
    plt.savefig(f"{plots_dir}/{col}_box.png")  # Save in plots directory
    plt.show()

def scatter_plot(df, x_col, y_col):
    """plots scatter plot"""
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.savefig(f"{plots_dir}/{x_col}_vs_{y_col}.png")  # Save in plots directory
    plt.show()

def correlation_heatmap(df):
    """plots correlation heatmap"""
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.savefig(f"{plots_dir}/correlation_heatmap.png")  # Save in plots directory
    plt.show()

from preprocess import train_df

# Error handling for DataFrame and columns
if 'Glucose' in train_df.columns:
    hist(train_df, "Glucose")
if 'Age' in train_df.columns and 'DiabetesPedigreeFunction' in train_df.columns:
    scatter_plot(train_df, "Age", "DiabetesPedigreeFunction")
if 'BMI' in train_df.columns and 'Outcome' in train_df.columns:
    scatter_plot(train_df, "BMI", "Outcome")
if 'BloodPressure' in train_df.columns:
    hist(train_df, "BloodPressure")
if 'BMI' in train_df.columns:
    hist(train_df, "BMI")

def save_stats(df, filename="stats.csv"):
    stats_list = []

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Only for numerical columns
            mean_value = df[column].mean()
            median_value = df[column].median()
            mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else None  # Handle empty mode case

            stats_list.append({"Column": column, "Mean": mean_value, "Median": median_value, "Mode": mode_value})

    # Convert list to DataFrame
    stats_df = pd.DataFrame(stats_list)

    # Save to CSV
    stats_df.to_csv(filename, index=False)

# Example usage
save_stats(train_df)
