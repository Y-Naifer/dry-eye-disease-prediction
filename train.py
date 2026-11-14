import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = 'data/Dry_Eye_Dataset.csv'
TARGET = 'dry_eye_disease'
SYMPTOMS = ['discomfort_eye_strain', 'redness_in_eye', 'itchiness_irritation_in_eye']
FEATURES = ['any_symptom']


def add_features(df):
    df = df.copy()
    df['any_symptom'] = (df[SYMPTOMS].sum(axis=1) > 0).astype(int)
    return df


def load_data(path=DATA):
    df = pd.read_csv(path)
    df.columns = (df.columns.str.lower()
                  .str.replace('/', '_', regex=False)
                  .str.replace('-', '_', regex=False)
                  .str.replace(' ', '_', regex=False))
    for c in SYMPTOMS + [TARGET]:
        df[c] = (df[c] == 'Y').astype(int)
    return add_features(df[SYMPTOMS + [TARGET]])


def main():
    df = load_data()

    # same split as in the notebook
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(df_full_train[FEATURES], df_full_train[TARGET])

    auc = roc_auc_score(df_test[TARGET], model.predict_proba(df_test[FEATURES])[:, 1])
    print(f'test ROC AUC: {auc:.3f}')

    with open('model.bin', 'wb') as f_out:
        pickle.dump(model, f_out)
    print('saved model.bin')


if __name__ == '__main__':
    main()
