from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import gc

def preprocess_data(data):
    data.iloc[:, 1:30] = StandardScaler().fit_transform(data.iloc[:, 1:30])
    X = normalize(data.iloc[:, 1:30].values, norm="l1")
    y = data.iloc[:, 30].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    gc.collect()
    return X_train, X_test, y_train, y_test

