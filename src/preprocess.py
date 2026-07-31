import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df, verbose=False):
    df = df.copy()

    # lowercase column names
    df.columns = df.columns.str.lower()

    if verbose:
        print('starting shape:', df.shape)

    # drop columns not useful as predictors
    excluded_cols = ['timestamp', 'comments', 'state']
    df = df.drop(columns=excluded_cols, errors='ignore')

    # age: keep only realistic values
    df = df[(df['age'] >= 18) & (df['age'] <= 100)]
    if verbose:
        print('after age filter:', df.shape)

    # country: collapse to a simple US vs. non-US flag
    df['is_us'] = (df['country'] == 'United States').astype(int)
    df = df.drop(columns=['country'])

    # gender: clean free text into 3 categories
    df['gender'] = df['gender'].str.lower().str.strip()
    df['gender'] = df['gender'].apply(
        lambda x: 'female' if ('female' in x or x == 'f')
        else 'male' if ('male' in x or x == 'm')
        else 'other'
    )

    # standardize "unsure" wording, only in columns that actually use it
    unsure_cols = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity',
                   'mental_vs_physical', 'mental_health_consequence', 'phys_health_consequence',
                   'mental_health_interview', 'phys_health_interview', 'leave']
    df[unsure_cols] = df[unsure_cols].replace({"Don't know": "Unsure", "Not sure": "Unsure", "Maybe": "Unsure"})

    # self_employed: drop the few rows with missing values
    df = df.dropna(subset=['self_employed'])
    if verbose:
        print('after self_employed dropna:', df.shape)

    # work_interfere: NaN means "not applicable" (no treatment sought),
    # except for the 4 rows where treatment == 'Yes' but the question was skipped — drop those
    drop_mask = df['work_interfere'].isnull() & (df['treatment'] == 'Yes')
    df = df[~drop_mask]
    df['work_interfere'] = df['work_interfere'].fillna('Not applicable')
    if verbose:
        print('after work_interfere handling:', df.shape)

    # safety check: no column should have missing values left before encoding
    remaining_nulls = df.isnull().sum()
    if remaining_nulls.sum() > 0:
        raise ValueError(f'Unexpected missing values remain before encoding:\n{remaining_nulls[remaining_nulls > 0]}')

    return df


def encode_data(df):
    df = df.copy()

    # binary columns: yes/no -> 1/0
    binary_cols = ['self_employed', 'family_history', 'treatment',
                   'remote_work', 'tech_company', 'obs_consequence']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # ordinal columns: map to numbers based on natural order
    work_interfere_map = {'Not applicable': 0, 'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4}
    df['work_interfere'] = df['work_interfere'].map(work_interfere_map)

    leave_map = {'Very easy': 0, 'Somewhat easy': 1, 'Unsure': 2, 'Somewhat difficult': 3, 'Very difficult': 4}
    df['leave'] = df['leave'].map(leave_map)

    no_employees_map = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5}
    df['no_employees'] = df['no_employees'].map(no_employees_map)

    # nominal columns: one-hot encode
    nominal_cols = ['gender', 'benefits', 'care_options', 'wellness_program',
                    'seek_help', 'anonymity', 'mental_vs_physical', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor',
                    'mental_health_interview', 'phys_health_interview']
    df = pd.get_dummies(df, columns=nominal_cols)

    # safety check: mapping should not have introduced any new NaNs
    mapped_cols = ['work_interfere', 'leave', 'no_employees'] + binary_cols
    remaining_nulls = df[mapped_cols].isnull().sum()
    if remaining_nulls.sum() > 0:
        raise ValueError(f'Mapping introduced missing values (unmapped category):\n{remaining_nulls[remaining_nulls > 0]}')

    # scale age so it's not on a much larger range than the binary/ordinal columns
    scaler = StandardScaler()
    df['age'] = scaler.fit_transform(df[['age']])

    return df


def preprocess(path, verbose=False):
    df = load_data(path)
    df = clean_data(df, verbose=verbose)
    df = encode_data(df)
    return df