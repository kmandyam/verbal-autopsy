import os
import pandas as pd

# a script that takes as input the out-domain.csv file
# and produces splits for training the baseline classifiers on
# we set L to be 3 arbitrarily (3 instances of each classifier are trained here)

out_domain_df = pd.read_csv("raw/out-domain.csv")

pool_df = out_domain_df

# split the train df into three partitions
df1 = pool_df.sample(frac=0.333)
pool_df = pool_df.drop(df1.index)

df2 = pool_df.sample(frac=0.5)
pool_df = pool_df.drop(df2.index)

df3 = pool_df.sample(frac=1)
pool_df = pool_df.drop(df3.index)

# split 1 trains on df1, df2, evaluates on df3
# split 2 trains on df2, df3, evaluates on df1
# split 3 trains on df3, df1, evaluates on df2





