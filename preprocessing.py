"""
Step 2 :Preprocessing
Feature engineering + encoding, lalu simpan preprocessor ke pickle.
"""

from pathlib import Path
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

BASE_DIR       = Path(__file__).parent
INGESTED_DIR   = BASE_DIR / "ingested"
PROCESSED_DIR  = BASE_DIR / "processed"
INPUT_FILE     = INGESTED_DIR / "train.csv"
OUTPUT_FILE    = PROCESSED_DIR / "train_processed.csv"
PREPROCESSOR_FILE = BASE_DIR / "models" / "preprocessor.pkl"

CATEGORICAL_FEATURES = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
NUMERICAL_FEATURES   = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
    'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing',
    'RoomService_ratio', 'FoodCourt_ratio', 'ShoppingMall_ratio',
    'Spa_ratio', 'VRDeck_ratio'
]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Deck']      = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
    df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else -1).astype(float)
    df['Side']      = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')

    df['Group']      = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Group_size'] = df.groupby('Group')['Group'].transform('count')
    df['Solo']       = (df['Group_size'] == 1).astype(int)

    df['FirstName']   = df['Name'].apply(lambda x: x.split()[0]  if pd.notna(x) else 'Unknown')
    df['LastName']    = df['Name'].apply(lambda x: x.split()[-1] if pd.notna(x) else 'Unknown')
    df['Family_size'] = df.groupby('LastName')['LastName'].transform('count')

    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['HasSpending']   = (df['TotalSpending'] > 0).astype(int)
    df['NoSpending']    = (df['TotalSpending'] == 0).astype(int)
    for col in spending_cols:
        df[f'{col}_ratio'] = df[col] / (df['TotalSpending'] + 1)

    df['Age_group'] = pd.cut(
        df['Age'], bins=[0, 12, 18, 30, 50, 100],
        labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior']
    ).astype(str)

    df['Age_missing']      = df['Age'].isna().astype(int)
    df['CryoSleep_missing'] = df['CryoSleep'].isna().astype(int)

    return df


def preprocess_data(df: pd.DataFrame):
    """Fit encoders dan return X, y, feature_columns, preprocessor dict."""
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('Unknown')
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fit LabelEncoders + simpan medians untuk inference
    label_encoders = {}
    medians = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            medians[col] = df[col].median()

    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = df[feature_columns]
    y = df['Transported'].astype(int)
    preprocessor = {'label_encoders': label_encoders, 'medians': medians}

    return X, y, feature_columns, preprocessor


def transform(df: pd.DataFrame, preprocessor: dict, feature_columns: list) -> pd.DataFrame:
    """Transform data baru menggunakan preprocessor yang sudah di-fit (untuk inference)."""
    df = df.copy()
    label_encoders = preprocessor['label_encoders']
    medians        = preprocessor['medians']

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('Unknown').astype(str)
        le = label_encoders[col]
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0))

    return df[feature_columns]


def save_preprocessor(preprocessor: dict, feature_columns: list):
    PREPROCESSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROCESSOR_FILE, 'wb') as f:
        pickle.dump({'preprocessor': preprocessor, 'feature_columns': feature_columns}, f)
    print(f"Preprocessor saved → {PREPROCESSOR_FILE}")


def load_preprocessor():
    with open(PREPROCESSOR_FILE, 'rb') as f:
        obj = pickle.load(f)
    return obj['preprocessor'], obj['feature_columns']
