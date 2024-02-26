import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer  # För att hantera saknade värden mer effektivt

def load_data(filepath, delimiter=','):
    """
    Generisk funktion för att ladda data från en fil.

    Args:
        filepath (str): Sökväg till filen som ska läsas.
        delimiter (str): Avgränsare för data i filen.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Egenskaper (X) och etiketter/mål (y).
    """
    # Exempel på att läsa en CSV-fil, anpassa efter behov
    data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=1)
    X, y = data[:, :-1], data[:, -1]
    return X, y

def clean_data(X, y, strategy='mean'):
    """
    Rengör dataset genom att fylla i saknade värden.

    Args:
        X (np.ndarray): Egenskaper.
        y (np.ndarray): Etiketter/mål.
        strategy (str): Strategi för att hantera saknade värden ('mean', 'median', 'most_frequent').

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rengjorda egenskaper och etiketter/mål.
    """
    imputer = SimpleImputer(strategy=strategy)
    X_clean = imputer.fit_transform(X)
    # Antag att y inte har några saknade värden eller hantera y separat om nödvändigt
    return X_clean, y

def normalize_data(X):
    """
    Standardiserar data (genomsnitt = 0 och standardavvikelse = 1).

    Args:
        X (np.ndarray): Egenskaper som ska normaliseras.

    Returns:
        np.ndarray: Normaliserade egenskaper.
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def encode_categorical(X, categorical_features='all'):
    """
    Kodar om kategoriska variabler till numeriska med one-hot encoding.

    Args:
        X (np.ndarray): Egenskaper.
        categorical_features (str or list of int): Specifierar vilka kategoriska funktioner som ska kodas.

    Returns:
        np.ndarray: Egenskaper med kategoriska variabler kodade.
    """
    encoder = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X) if categorical_features == 'all' else encoder.fit_transform(X[:, categorical_features])
    return X_encoded

def preprocess_data(filepath, delimiter=',', test_size=0.2, strategy='mean', categorical_features='all'):
    """
    Förbereder data från fil genom att ladda, rengöra, normalisera och dela upp den.

    Args:
        filepath (str): Sökväg till datan.
        delimiter (str): Avgränsare för data i filen.
        test_size (float): Andel av data som ska användas som testset.
        strategy (str): Strategi för att hantera saknade värden.
        categorical_features (str or list of int): Specifierar kategoriska funktioner för encoding.

    Returns:
        Tuple: Uppdelade och förberedda tränings- och testset.
    """
    X, y = load_data(filepath, delimiter)
    X, y = clean_data(X, y, strategy)
    if categorical_features != 'none':
        X = encode_categorical(X, categorical_features)
    X = normalize_data(X)
    return train_test_split(X, y, test_size=test_size, random_state=42)

