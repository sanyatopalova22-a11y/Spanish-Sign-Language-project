import glob, os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
 
X,y = [],[]
for path in glob.glob("data/*.npy"):
    label = os.path.splitext(os.path.basename(path))[0]
    arr = np.load(path)
    X.append(arr)
    y += [label]*len(arr)
 
X = np.vstack(X)
y = np.array(y)
print("Classes:", sorted(set(y)),"|Samples:", len(y))
 
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,
                                       random_state=42,stratify=y)
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_tr,y_tr)
 
print("Train accuracy:", round(model.score(X_tr,y_tr),3))
joblib.dump(model,"model.joblib")
print("Saved model.joblib")
 