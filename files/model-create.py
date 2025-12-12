import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("files/behavior-performance.txt", sep="\t")

results = {}

video_ids = data['VidID'].unique()

for video_id in video_ids:
    subset = data[data['VidID'] == video_id]

    if len(subset) < 20:
        results[video_id] = None
        continue

    X = subset.drop(columns=["VidID", "userID", "s"])
    y = subset["s"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_0 = X_train[y_train == 0]
    X_train_1 = X_train[y_train == 1]

    if len(X_train_0) < 2 or len(X_train_1) < 2:
        results[video_id] = None
        continue

    gmm0 = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm1 = GaussianMixture(n_components=2, covariance_type='full', random_state=42)

    gmm0.fit(X_train_0)
    gmm1.fit(X_train_1)

    logp0 = gmm0.score_samples(X_test)
    logp1 = gmm1.score_samples(X_test)

    y_pred = np.where(logp1 > logp0, 1, 0)

    acc = accuracy_score(y_test, y_pred)
    results[video_id] = acc

formatted = []

for vid in sorted(results.keys()):
    acc = results[vid]
    if acc is None:
        formatted.append((vid, None))
    else:
        formatted.append((vid, round(acc, 4)))

print("\nVideoID | Accuracy")
print("--------+----------")

for vid, acc in formatted:
    if acc is None:
        print(f"{vid:<7} | N/A")
    else:
        print(f"{vid:<7} | {acc:.4f}")

