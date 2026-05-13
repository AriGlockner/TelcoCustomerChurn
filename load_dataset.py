import kagglehub
import pandas as pd

# Download the data
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

# Read the csv file and save it to the data directory
df = pd.read_csv(f"{path}/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.to_csv('data/telco_customer_churn.csv', index=False)
