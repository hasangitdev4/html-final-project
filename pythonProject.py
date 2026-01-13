import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(r"C:\Users\pc\OneDrive\Desktop\python\SuperMarketSealsDataSet.csv")

output_dir = r"C:\Users\pc\OneDrive\Desktop\python\plots"
os.makedirs(output_dir, exist_ok=True)

df.columns = df.columns.str.strip()

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

categorical_cols = ['Payment', 'Gender', 'Customer type', 'Branch', 'Product line']
for col in categorical_cols:
    df[col] = df[col].astype('category')

branches = df['Branch'].cat.categories
 
for branch in branches:
    print(f"\n===== Branch: {branch} =====")
    branch_df = df[df['Branch'] == branch]
    
    sales_stats = branch_df['Sales'].agg(['mean','median','min','max'])
    print("Sales Stats:")
    print(sales_stats)
    
   
    
    
    income_stats = branch_df['gross income'].agg(['mean','median','min','max'])
    print("Gross Income Stats:")
    print(income_stats)
    

    print("\nSales by Customer Type:")
    print(branch_df.groupby('Customer type', observed=True)['Sales'].agg(['mean','median','min','max']))
    
    print("\nSales by Gender:")
    print(branch_df.groupby('Gender', observed=True)['Sales'].agg(['mean','median','min','max']))
    
    print("\nSales by Payment Method:")
    print(branch_df.groupby('Payment', observed=True)['Sales'].agg(['mean','median','min','max']))


    plt.hist(branch_df['Sales'], bins=30)
    plt.title(f"Distribution of Sales - Branch {branch}")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    #plt.savefig(f"{output_dir}/1_sales_distribution_branch_{branch}.png", bbox_inches='tight')
    plt.show()
    plt.close()

    plt.hist(branch_df['Rating'], bins=20)
    plt.title(f"Distribution of Ratings - Branch {branch}")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    #plt.savefig(f"{output_dir}/2_rating_distribution_branch_{branch}.png", bbox_inches='tight')
    plt.show()
    plt.close()

    branch_df['Payment'].value_counts().plot(kind='bar')
    plt.title(f"Payment Method Frequency - Branch {branch}")
    plt.xlabel("Payment Method")
    plt.ylabel("Count")
    #plt.savefig(f"{output_dir}/3_payment_frequency_branch_{branch}.png", bbox_inches='tight')
    plt.show()
    plt.close()

    sales_time = branch_df.groupby('Date')['Sales'].sum()
    plt.plot(sales_time)
    plt.title(f"Total Sales Over Time - Branch {branch}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    #plt.savefig(f"{output_dir}/4_sales_over_time_branch_{branch}.png", bbox_inches='tight')
    plt.show()
    plt.close()

    plt.scatter(branch_df['Sales'], branch_df['Rating'])
    plt.title(f"Sales vs Rating - Branch {branch}")
    plt.xlabel("Sales")
    plt.ylabel("Rating")
    #plt.savefig(f"{output_dir}/5_sales_vs_rating_branch_{branch}.png", bbox_inches='tight')
    plt.show()
    plt.close()

    numerical_df = branch_df.select_dtypes(include=np.number)
    corr = numerical_df.corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title(f"Correlation Heatmap - Branch {branch}")
    #plt.savefig(f"{output_dir}/6_correlation_heatmap_branch_{branch}.png", bbox_inches='tight')
    plt.show()
    plt.close()

    branch_df.boxplot(column='gross income', by='Product line', rot=45)
    plt.title(f"Gross Income by Product Line - Branch {branch}")
    plt.suptitle("")
    plt.xlabel("Product Line")
    plt.ylabel("Gross Income")
    #plt.savefig(f"{output_dir}/7_gross_income_boxplot_branch_{branch}.png", bbox_inches='tight')
    plt.show()
    plt.close()

