import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df, verbose=False):
    df = df.copy()

    df.columns = df.columns.str.lower()

    if verbose:
        print('starting shape:', df.shape)

    excluded_cols = ['timestamp', 'comments', 'state']
    df = df.drop(columns=excluded_cols, errors='ignore')

    df = df[(df['age'] >= 18) & (df['age'] <= 100)]
    if verbose:
        print('after age filter:', df.shape)

    df['is_us'] = (df['country'] == 'United States').astype(int)
    df = df.drop(columns=['country'])

    df['gender'] = df['gender'].str.lower().str.strip()
    df['gender'] = df['gender'].apply(
        lambda x: 'female' if ('female' in x or x == 'f')
        else 'male' if ('male' in x or x == 'm')
        else 'other'
    )

    unsure_cols = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity',
                   'mental_vs_physical', 'mental_health_consequence', 'phys_health_consequence',
                   'mental_health_interview', 'phys_health_interview', 'leave']
    df[unsure_cols] = df[unsure_cols].replace({"Don't know": "Unsure", "Not sure": "Unsure", "Maybe": "Unsure"})

    df = df.dropna(subset=['self_employed'])
    if verbose:
        print('after self_employed dropna:', df.shape)

    drop_mask = df['work_interfere'].isnull() & (df['treatment'] == 'Yes')
    df = df[~drop_mask]
    df['work_interfere'] = df['work_interfere'].fillna('Not applicable')
    if verbose:
        print('after work_interfere handling:', df.shape)

    remaining_nulls = df.isnull().sum()
    if remaining_nulls.sum() > 0:
        raise ValueError(f'Unexpected missing values remain before encoding:\n{remaining_nulls[remaining_nulls > 0]}')

    return df


def encode_data(df):
    """
    Encodes categorical columns (binary, ordinal, one-hot).
    Does NOT scale age -- that's handled separately by scale_age(),
    since scaling must be fit only on training data, after the split.
    """
    df = df.copy()

    binary_cols = ['self_employed', 'family_history', 'treatment',
                   'remote_work', 'tech_company', 'obs_consequence']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    work_interfere_map = {'Not applicable': 0, 'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4}
    df['work_interfere'] = df['work_interfere'].map(work_interfere_map)

    leave_map = {'Very easy': 0, 'Somewhat easy': 1, 'Unsure': 2, 'Somewhat difficult': 3, 'Very difficult': 4}
    df['leave'] = df['leave'].map(leave_map)

    no_employees_map = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5}
    df['no_employees'] = df['no_employees'].map(no_employees_map)

    nominal_cols = ['gender', 'benefits', 'care_options', 'wellness_program',
                    'seek_help', 'anonymity', 'mental_vs_physical', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor',
                    'mental_health_interview', 'phys_health_interview']
    df = pd.get_dummies(df, columns=nominal_cols)

    mapped_cols = ['work_interfere', 'leave', 'no_employees'] + binary_cols
    remaining_nulls = df[mapped_cols].isnull().sum()
    if remaining_nulls.sum() > 0:
        raise ValueError(f'Mapping introduced missing values (unmapped category):\n{remaining_nulls[remaining_nulls > 0]}')

    return df


def scale_age(df, scaler=None):
    """
    Scales the 'age' column.
    If scaler is None, fits a new StandardScaler on this data (use only on training data).
    If scaler is provided, reuses it to transform this data (use on test data / new predictions).
    Returns (df_with_scaled_age, the_scaler_used).
    """
    df = df.copy()

    if scaler is None:
        scaler = StandardScaler()
        df['age'] = scaler.fit_transform(df[['age']])
    else:
        df['age'] = scaler.transform(df[['age']])

    return df, scaler


def preprocess(path, verbose=False):
    """
    Full cleaning + categorical encoding pipeline.
    Does NOT scale age or split train/test -- that happens in train_models.py,
    where the scaler is fit on training data only.
    """
    df = load_data(path)
    df = clean_data(df, verbose=verbose)
    df = encode_data(df)
    return df