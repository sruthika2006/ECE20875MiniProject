import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")

data = pd.read_csv("behavior-performance.txt", sep="\t")

rows_per_video = data.groupby('VidID').size()

video_id =90
subset = data[data["VidID"] == video_id]

if len(subset) < 20:
    print(f"Video {video_id} has less than 20 rows.")

else:
  X = subset.drop(columns=["VidID","userID", "s"])
  Y = subset["s"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1500).fit(X_train,Y_train)
Y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print(f"Video {video_id} Accuracy: {acc}")

video_ids = data['VidID'].unique()

results = {}

for video_id in video_ids:
    subset = data[data['VidID'] == video_id]


    if len(subset) < 20:
        print(f"Video {video_id} has less than 20 rows.")
        results[video_id] = None
        continue


    X = subset.drop(columns=["VidID", "userID", "s"])
    Y = subset["s"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    model = LogisticRegression(max_iter=1500)
    model.fit(X_train, Y_train)


    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    results[video_id] = acc

completed_videos = data.groupby('userID')['VidID'].nunique()
eligible_students = completed_videos[completed_videos >= 5].index
subset = data[data['userID'].isin(eligible_students)]

features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
X = subset[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(1, 20):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
subset['cluster'] = clusters

label=kmeans.labels_

from collections import Counter
Counter(label)

score=silhouette_score(X,label)
print(score)