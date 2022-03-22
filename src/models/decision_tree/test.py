import pandas as pd
from KFold import KFold

kfold = KFold(5, shuffle=True, random_state=1)
df = pd.read_csv('../../codon-usage-data-set/codon_usage.csv', low_memory=False)

for idx_train, idx_test in kfold.split(df):
    print(len(idx_train), len(idx_test))

# Statistical Measures


