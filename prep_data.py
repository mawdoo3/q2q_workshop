import pandas as pd
from sklearn.model_selection import train_test_split


def num(label):
    if label == "yes":
        return 1
    else:
        return 0

df = pd.read_csv('q2q_similarity_workshop_v2.tsv', sep='\t')

df.rename(columns={'semantically similar2':'label'} ,inplace=True)
df['label'] = df['label'].apply(lambda x: num(x))

train, test = train_test_split(df, test_size = 4000, random_state = 0, stratify=df['label'])
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
