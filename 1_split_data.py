import os
import pandas as pd

# a script that takes as input the out-domain.csv file
# and produces splits for training the baseline classifiers on
# we set L to be 3 arbitrarily (3 instances of each classifier are trained here)

out_domain_df = pd.read_csv("raw/splitA/out-domain.csv")

pool_df = out_domain_df

# split the train df into three partitions
df1 = pool_df.sample(frac=0.2)
pool_df = pool_df.drop(df1.index)

df2 = pool_df.sample(frac=0.25)
pool_df = pool_df.drop(df2.index)

df3 = pool_df.sample(frac=0.333)
pool_df = pool_df.drop(df3.index)

df4 = pool_df.sample(frac=0.5)
pool_df = pool_df.drop(df4.index)

df5 = pool_df.sample(frac=1)
pool_df = pool_df.drop(df5.index)

print("Dataset Split Sizes")
print("DF1: ", len(df1))
print("DF2: ", len(df2))
print("DF3: ", len(df3))
print("DF4: ", len(df4))
print("DF5: ", len(df5))

assert len(pool_df) == 0

# split 1 trains on df1, df2, df3, df4 evaluates on df5
# split 2 trains on df2, df3, df4, df5 evaluates on df1
# split 3 trains on df3, df4, df5, df1 evaluates on df2
# split 4 trains on df4, df5, df1, df2 evaluates on df3
# split 5 trains on df5, df1, df2, df3 evaluates on df4

if not os.path.exists('augmented_dataset/splitA/split_1'):
    os.makedirs('augmented_dataset/splitA/split_1')

if not os.path.exists('augmented_dataset/splitA/split_2'):
    os.makedirs('augmented_dataset/splitA/split_2')

if not os.path.exists('augmented_dataset/splitA/split_3'):
    os.makedirs('augmented_dataset/splitA/split_3')

if not os.path.exists('augmented_dataset/splitA/split_4'):
    os.makedirs('augmented_dataset/splitA/split_4')

if not os.path.exists('augmented_dataset/splitA/split_5'):
    os.makedirs('augmented_dataset/splitA/split_5')


print("Writing datasets to split folders...")

split_1_train = pd.concat([df1, df2, df3, df4])
split_1_test = df5
split_1_train.to_csv('augmented_dataset/splitA/split_1/train.csv', index=False)
split_1_test.to_csv('augmented_dataset/splitA/split_1/test.csv', index=False)

split_2_train = pd.concat([df2, df3,  df4, df5])
split_2_test = df1
split_2_train.to_csv('augmented_dataset/splitA/split_2/train.csv', index=False)
split_2_test.to_csv('augmented_dataset/splitA/split_2/test.csv', index=False)

split_3_train = pd.concat([df3, df4, df5, df1])
split_3_test = df2
split_3_train.to_csv('augmented_dataset/splitA/split_3/train.csv', index=False)
split_3_test.to_csv('augmented_dataset/splitA/split_3/test.csv', index=False)

split_4_train = pd.concat([df4, df5, df1, df2])
split_4_test = df3
split_4_train.to_csv('augmented_dataset/splitA/split_4/train.csv', index=False)
split_4_test.to_csv('augmented_dataset/splitA/split_4/test.csv', index=False)

split_5_train = pd.concat([df5, df1, df2, df3])
split_5_test = df4
split_5_train.to_csv('augmented_dataset/splitA/split_5/train.csv', index=False)
split_5_test.to_csv('augmented_dataset/splitA/split_5/test.csv', index=False)

print("...writing complete")


