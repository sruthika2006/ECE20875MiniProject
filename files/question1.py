import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

df = pd.read_csv("files/behavior-performance.txt", sep="\t")

valid_df = df.groupby("userID").filter(lambda x: len(x) >= 5)

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
    X_sample, labels_sample = resample(X_scaled, labels, n_samples=2000, random_state=0)
    sil_scores.append(silhouette_score(X_sample, labels_sample))

print("K values:", list(ks))
print("BIC:", bic_scores)
print("AIC:", aic_scores)
print("Silhouette:", sil_scores)

best_k = ks[np.argmin(bic_scores)]
print("\nBest k based on BIC:", best_k)

final_gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=0).fit(X_scaled)

valid_df["cluster"] = final_gmm.predict(X_scaled)

cluster_means = pd.DataFrame(scaler.inverse_transform(final_gmm.means_), columns=features)

print("\nCluster means (original scale):")
print(cluster_means)

print("\nCluster counts:")
print(valid_df["cluster"].value_counts())
