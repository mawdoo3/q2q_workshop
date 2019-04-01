import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

print('reading training and testing datasets...')
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print('processing datasets...')
df['question1'] = df['question1'].apply(lambda x: re.sub('؟', '', x))
df['question2'] = df['question2'].apply(lambda x: re.sub('؟', '', x))
df_test['question1'] = df_test['question1'].apply(lambda x: re.sub('؟', '', x))
df_test['question2'] = df_test['question2'].apply(lambda x: re.sub('؟', '', x))

print('vectorizing data...')
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
transformer = vectorizer.fit(pd.concat([df['question1'],df['question2']],axis = 0))
q1_tranformed = transformer.transform(df['question1'])
q2_tranformed = transformer.transform(df['question2'])
q1_tranformed_test = transformer.transform(df_test['question1'])
q2_tranformed_test = transformer.transform(df_test['question2'])
features = np.hstack([q1_tranformed.toarray(), q2_tranformed.toarray()])
features_test = np.hstack([q1_tranformed_test.toarray(), q2_tranformed_test.toarray()])

print('spliting training data to a training and a validation set...')
train_features, _, train_labels, _ = train_test_split(features, df['label'], test_size = 0.2, random_state = 0, stratify=df['label'])

print('training model (this could take sometime)...')
clf = LogisticRegression(solver='lbfgs')
clf.fit(train_features, train_labels)

print('calculating results...')
predictions = clf.predict(features_test)

print('This solution gets an f-1 score of: {0:.4g}'.format(f1_score(df_test['label'], predictions)))
