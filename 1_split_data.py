import os
import pandas as pd

# a script that takes as input the out-domain.csv file
# and produces splits for training the baseline classifiers on
# we set L to be 3 arbitrarily (3 instances of each classifier are trained here)

out_domain_df = pd.read_csv("raw/splitA/out-domain.csv")

pool_df = out_domain_df

# split the train df into three partitions
df1 = pool_df.sample(frac=0.1)
pool_df = pool_df.drop(df1.index)

df2 = pool_df.sample(frac=0.1111)
pool_df = pool_df.drop(df2.index)

df3 = pool_df.sample(frac=0.125)
pool_df = pool_df.drop(df3.index)

df4 = pool_df.sample(frac=0.1428)
pool_df = pool_df.drop(df4.index)

df5 = pool_df.sample(frac=0.166667)
pool_df = pool_df.drop(df5.index)

df6 = pool_df.sample(frac=0.2)
pool_df = pool_df.drop(df6.index)

df7 = pool_df.sample(frac=0.25)
pool_df = pool_df.drop(df7.index)

df8 = pool_df.sample(frac=0.3333)
pool_df = pool_df.drop(df8.index)

df9 = pool_df.sample(frac=0.5)
pool_df = pool_df.drop(df9.index)

df10 = pool_df.sample(frac=1)
pool_df = pool_df.drop(df10.index)

print("Dataset Split Sizes")
print("DF1: ", len(df1))
print("DF2: ", len(df2))
print("DF3: ", len(df3))
print("DF4: ", len(df4))
print("DF5: ", len(df5))
print("DF6: ", len(df6))
print("DF7: ", len(df7))
print("DF8: ", len(df8))
print("DF9: ", len(df9))
print("DF10: ", len(df10))

assert len(pool_df) == 0

# split 1 trains on df1, df2, df3, df4, df5, df6, df7, df8, df9 evaluates on df10
# split 2 trains on df2, df3, df4, df5, df6, df7, df8, df9, df10 evaluates on df1
# split 3 trains on df3, df4, df5, df6, df7, df8, df9, df10, df1 evaluates on df2
# split 4 trains on df4, df5, df6, df7, df8, df9, df10, df1, df2 evaluates on df3
# split 5 trains on df5, df6, df7, df8, df9, df10, df1, df2, df3 evaluates on df4
# split 6 trains on df6, df7, df8, df9, df10, df1, df2, df3, df4 evaluates on df5
# split 7 trains on df7, df8, df9, df10, df1, df2, df3, df4, df5 evaluates on df6
# split 8 trains on df8, df9, df10, df1, df2, df3, df4, df5, df6 evaluates on df7
# split 9 trains on df9, df10, df1, df2, df3, df4, df5, df6, df7 evaluates on df8
# split 10 trains on df10, df1, df2, df3, df4, df5, df6, df7, df8 evaluates on df9

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

if not os.path.exists('augmented_dataset/splitA/split_6'):
    os.makedirs('augmented_dataset/splitA/split_6')

if not os.path.exists('augmented_dataset/splitA/split_7'):
    os.makedirs('augmented_dataset/splitA/split_7')

if not os.path.exists('augmented_dataset/splitA/split_8'):
    os.makedirs('augmented_dataset/splitA/split_8')

if not os.path.exists('augmented_dataset/splitA/split_9'):
    os.makedirs('augmented_dataset/splitA/split_9')

if not os.path.exists('augmented_dataset/splitA/split_10'):
    os.makedirs('augmented_dataset/splitA/split_10')


print("Writing datasets to split folders...")

split_1_train = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9])
split_1_test = df10
split_1_train.to_csv('augmented_dataset/splitA/split_1/train.csv', index=False)
split_1_test.to_csv('augmented_dataset/splitA/split_1/test.csv', index=False)

split_2_train = pd.concat([df2, df3,  df4, df5, df6, df7, df8, df9, df10])
split_2_test = df1
split_2_train.to_csv('augmented_dataset/splitA/split_2/train.csv', index=False)
split_2_test.to_csv('augmented_dataset/splitA/split_2/test.csv', index=False)

split_3_train = pd.concat([df3, df4, df5, df6, df7, df8, df9, df10, df1])
split_3_test = df2
split_3_train.to_csv('augmented_dataset/splitA/split_3/train.csv', index=False)
split_3_test.to_csv('augmented_dataset/splitA/split_3/test.csv', index=False)

split_4_train = pd.concat([df4, df5, df6, df7, df8, df9, df10, df1, df2])
split_4_test = df3
split_4_train.to_csv('augmented_dataset/splitA/split_4/train.csv', index=False)
split_4_test.to_csv('augmented_dataset/splitA/split_4/test.csv', index=False)

split_5_train = pd.concat([df5, df6, df7, df8, df9, df10, df1, df2, df3])
split_5_test = df4
split_5_train.to_csv('augmented_dataset/splitA/split_5/train.csv', index=False)
split_5_test.to_csv('augmented_dataset/splitA/split_5/test.csv', index=False)

split_6_train = pd.concat([df6, df7, df8, df9, df10, df1, df2, df3, df4])
split_6_test = df5
split_6_train.to_csv('augmented_dataset/splitA/split_6/train.csv', index=False)
split_6_test.to_csv('augmented_dataset/splitA/split_6/test.csv', index=False)

split_7_train = pd.concat([df7, df8, df9, df10, df1, df2, df3, df4, df5])
split_7_test = df6
split_7_train.to_csv('augmented_dataset/splitA/split_7/train.csv', index=False)
split_7_test.to_csv('augmented_dataset/splitA/split_7/test.csv', index=False)

split_8_train = pd.concat([df8, df9, df10, df1, df2, df3, df4, df5, df6])
split_8_test = df7
split_8_train.to_csv('augmented_dataset/splitA/split_8/train.csv', index=False)
split_8_test.to_csv('augmented_dataset/splitA/split_8/test.csv', index=False)

split_9_train = pd.concat([df9, df10, df1, df2, df3, df4, df5, df6, df7])
split_9_test = df8
split_9_train.to_csv('augmented_dataset/splitA/split_9/train.csv', index=False)
split_9_test.to_csv('augmented_dataset/splitA/split_9/test.csv', index=False)

split_10_train = pd.concat([df10, df1, df2, df3, df4, df5, df6, df7, df8])
split_10_test = df9
split_10_train.to_csv('augmented_dataset/splitA/split_10/train.csv', index=False)
split_10_test.to_csv('augmented_dataset/splitA/split_10/test.csv', index=False)

print("...writing complete")


