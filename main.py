import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False


def extract_1d_column_grayscale_features(image_path, n_parts):
    """
    Converts image to Grayscale, flattens to 1D column vector,
    divides into n_parts equal segments, and returns mean intensity per segment.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flattened_col = gray.flatten()
    chunks = np.array_split(flattened_col, n_parts)
    return [float(np.mean(chunk)) if chunk.size > 0 else 0.0 for chunk in chunks]


def load_dataset(data_dir):
    """
    Scans data directory for ClientRaw (real) and ImposterRaw (fake) images.
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    image_paths = []
    class_labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                full_path = os.path.join(root, file)
                lpath = full_path.lower()
                if 'clientraw' in lpath or 'client' in lpath or 'real' in lpath:
                    image_paths.append(full_path)
                    class_labels.append('real')
                elif 'imposterraw' in lpath or 'imposter' in lpath or 'fake' in lpath:
                    image_paths.append(full_path)
                    class_labels.append('fake')

    return image_paths, class_labels


def get_models():
    """
    Returns the dictionary of machine learning classifiers.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        "Support Vector Machine": SVC(random_state=42, probability=True),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "MLP Classifier": MLPClassifier(max_iter=500, random_state=42),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis()
    }

    if BOOSTING_AVAILABLE:
        models["XGBoost"] = XGBClassifier(eval_metric='logloss', random_state=42)
        models["LightGBM"] = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1)
        models["CatBoost"] = CatBoostClassifier(verbose=0, iterations=100, random_state=42)

    return models


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw")

    print(f"Loading dataset from: {raw_dir}")
    image_paths, class_labels = load_dataset(raw_dir)
    print(f"Found {len(image_paths)} total images.")

    if not image_paths:
        print("No images found. Please check dataset path.")
        return

    # Subsample 2000 images for fast cross-validation
    if len(image_paths) > 2000:
        print("Subsampling 2000 images for efficient processing...")
        np.random.seed(42)
        idx = np.random.choice(len(image_paths), 2000, replace=False)
        image_paths = [image_paths[i] for i in idx]
        class_labels = [class_labels[i] for i in idx]

    n_values = [2, 4, 8, 16,20,24,32,40,48,56,64]
    models = get_models()
    all_accuracies = {}

    for n in n_values:
        print(f"\n--- Processing n_parts = {n} ---")
        X, y = [], []
        for path, label in zip(image_paths, class_labels):
            feat = extract_1d_column_grayscale_features(path, n)
            if feat is not None:
                X.append(feat)
                y.append(1 if label == 'real' else 0)

        X_arr, y_arr = np.array(X), np.array(y)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        res = {}
        for name, model in models.items():
            scores = []
            for tr, ts in kf.split(X_arr, y_arr):
                model.fit(X_arr[tr], y_arr[tr])
                preds = model.predict(X_arr[ts])
                scores.append(accuracy_score(y_arr[ts], preds))
            res[name] = float(np.mean(scores))
            print(f"  {name:<30}: {res[name]:.4f}")

        all_accuracies[n] = res

    # Create summary DataFrame
    df_results = pd.DataFrame(all_accuracies).T
    df_results.index.name = 'n_parts (1D Column)'

    # Row Averages (Average accuracy for each n_parts)
    df_results['Average_Accuracy_by_N_Parts'] = df_results.mean(axis=1)

    # Column Averages (Average accuracy for each model across all n_parts)
    df_results.loc['Average_Accuracy_by_Model'] = df_results.mean(axis=0)

    print("\n" + "=" * 60)
    print("FINAL MODEL PERFORMANCE SUMMARY (ACCURACY):")
    print("=" * 60)
    print(df_results.to_string())

    excel_path = os.path.join(base_dir, "new_excel_sheet.xlsx")
    try:
        df_results.to_excel(excel_path)
        print(f"\nResults saved to: {excel_path}")
    except Exception as e:
        print(f"\nNote: Could not save Excel file ({e}). Results displayed above.")


if __name__ == "__main__":
    main()
