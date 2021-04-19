import pandas as pd


df = pd.read_csv('stocks.csv', header=[0, 1])
df.drop([0], axis=0, inplace=True)  # drop this row because it only has one column with Date in it
df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')] = pd.to_datetime(df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')], format='%Y-%m-%d')  # convert the first column to a datetime
df.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), inplace=True)  # set the first column as the index
df.index.name = None  # rename the index


# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std

df = df['Adj Close']
df.dropna(thresh=472, inplace=True)

df = z_score(df)
print(df)