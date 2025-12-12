import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

df = pd.read_csv("behavior-performance.txt")

valid_df = df.groupby("studentID").filter(lambda x: len(x) >= 5)

features = [
    "fracSpent", "fracComp", "fracPaused", "numPauses",
    "avgPBR", "numRWs", "numFFs"
]

X = valid_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

bic_scores = []
aic_scores = []
sil_scores = []

ks = range(2, 10)

for k in ks:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
    gmm.fit(X_scaled)

    labels = gmm.predict(X_scaled)

    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

    sil_scores.append(silhouette_score(X_scaled, labels))

print("K values:", list(ks))
print("BIC scores:", bic_scores)
print("AIC scores:", aic_scores)
print("Silhouette scores:", sil_scores)

best_k = 3
final_gmm = GaussianMixture(
    n_components=best_k,
    covariance_type='full',
    random_state=0
).fit(X_scaled)

valid_df["cluster"] = final_gmm.predict(X_scaled)

cluster_means = pd.DataFrame(
    scaler.inverse_transform(final_gmm.means_),
    columns=features
)

print("\nCluster means (real-scale):")
print(cluster_means)

print("\nCluster counts:")
print(valid_df["cluster"].value_counts())
