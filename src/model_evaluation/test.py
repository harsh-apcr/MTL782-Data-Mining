from KFold import KFold
import pandas as pd

kfold = KFold(5, shuffle=True, random_state=1)
df = pd.read_csv('../codon-usage-data-set/codon_usage.csv', low_memory=False)

for idx_train, idx_test in kfold.split(df):
    print(idx_train, idx_test)