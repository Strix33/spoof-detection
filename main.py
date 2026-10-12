import os
import cv2
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

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


def get_top_3_families():
    """
    Returns the top 3 performing model families for spoof detection.
    """
    tree_boosting = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
    }
    if BOOSTING_AVAILABLE:
        tree_boosting["XGBoost"] = XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=42)
        tree_boosting["LightGBM"] = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1)
        tree_boosting["CatBoost"] = CatBoostClassifier(verbose=0, iterations=50, thread_count=-1, random_state=42)

    distance_kernel = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
        "SVM (RBF Kernel)": SVC(kernel='rbf', max_iter=1000, cache_size=1000, random_state=42, probability=True),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(dual='auto', max_iter=2000, random_state=42))
    }

    probabilistic_bayes = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "Bernoulli Naive Bayes": BernoulliNB()
    }

    return {
        "Tree & Boosting Family": tree_boosting,
        "Distance & Kernel Family": distance_kernel,
        "Probabilistic & Bayes Family": probabilistic_bayes
    }


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

    n_values = [2, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64,70,90,110,130,150,170,340,520,620]
    families = get_top_3_families()
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
        all_family_models = {}

        # 1. Evaluate Individual Models & Create Intra-Family Voting Ensembles
        for fam_name, models in families.items():
            print(f"\n  [{fam_name}]")
            for name, model in models.items():
                scores = []
                for tr, ts in kf.split(X_arr, y_arr):
                    model.fit(X_arr[tr], y_arr[tr])
                    preds = model.predict(X_arr[ts])
                    scores.append(accuracy_score(y_arr[ts], preds))
                res[(fam_name, name)] = float(np.mean(scores))
                print(f"    {name:<35}: {res[(fam_name, name)]:.4f}")
                all_family_models[f"{fam_name}_{name}"] = model

            # Build Intra-Family Hard Voting Ensemble
            estimators = [(m_name.replace(' ', '_'), m_obj) for m_name, m_obj in models.items()]
            intra_ensemble = VotingClassifier(estimators=estimators, voting='hard')
            scores = []
            for tr, ts in kf.split(X_arr, y_arr):
                intra_ensemble.fit(X_arr[tr], y_arr[tr])
                preds = intra_ensemble.predict(X_arr[ts])
                scores.append(accuracy_score(y_arr[ts], preds))
            res[(fam_name, 'Intra_Family_Voting_Ensemble')] = float(np.mean(scores))
            print(f"    {'Intra_Family_Voting_Ensemble':<35}: {res[(fam_name, 'Intra_Family_Voting_Ensemble')]:.4f}")

        # 2. Inter-Family Meta Ensemble (Combining Top 3 Families)
        print(f"\n  [Ensemble Learning - Inter Family]")
        top_meta_estimators = [
            ("Tree_ExtraTrees", ExtraTreesClassifier(n_estimators=50, n_jobs=-1, random_state=42)),
            ("Distance_KNN", KNeighborsClassifier(n_jobs=-1)),
            ("Probabilistic_QDA", QuadraticDiscriminantAnalysis(reg_param=0.1))
        ]

        # Inter-Family Voting Ensemble
        inter_voting = VotingClassifier(estimators=top_meta_estimators, voting='hard')
        scores = []
        for tr, ts in kf.split(X_arr, y_arr):
            inter_voting.fit(X_arr[tr], y_arr[tr])
            preds = inter_voting.predict(X_arr[ts])
            scores.append(accuracy_score(y_arr[ts], preds))
        res[('Ensemble Learning', 'Inter_Family_Voting_Ensemble')] = float(np.mean(scores))
        print(f"    {'Inter_Family_Voting_Ensemble':<35}: {res[('Ensemble Learning', 'Inter_Family_Voting_Ensemble')]:.4f}")

        # Inter-Family Stacking Classifier
        inter_stacking = StackingClassifier(estimators=top_meta_estimators, final_estimator=LogisticRegression(random_state=42))
        scores = []
        for tr, ts in kf.split(X_arr, y_arr):
            inter_stacking.fit(X_arr[tr], y_arr[tr])
            preds = inter_stacking.predict(X_arr[ts])
            scores.append(accuracy_score(y_arr[ts], preds))
        res[('Ensemble Learning', 'Inter_Family_Stacking_Ensemble')] = float(np.mean(scores))
        print(f"    {'Inter_Family_Stacking_Ensemble':<35}: {res[('Ensemble Learning', 'Inter_Family_Stacking_Ensemble')]:.4f}")

        all_accuracies[n] = res

    # Create MultiIndex DataFrame (Level 0: Family / Ensemble, Level 1: Model Name)
    df_results = pd.DataFrame(all_accuracies).T
    df_results.index.name = 'n_parts (1D Column)'

    # Column Averages (Average accuracy across all n_parts)
    df_results.loc['Average_Accuracy_by_Model'] = df_results.mean(axis=0)

    print("\n" + "=" * 70)
    print("TOP 3 FAMILIES & ENSEMBLE LEARNING SUMMARY:")
    print("=" * 70)
    print(df_results.to_string())

    excel_path = os.path.join(base_dir, "final_excel.xlsx")
    try:
        df_results.to_excel(excel_path)
        print(f"\nTop 3 Families & Ensemble results successfully saved to: {excel_path}")
    except Exception as e:
        print(f"\nNote: Could not save Excel file ({e}). Results displayed above.")


if __name__ == "__main__":
    main()
