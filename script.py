import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Step 1: Data Preparation
data = pd.read_csv('data.csv')
X = data[['DCPLX', 'CCPLX', 'ICPLX', 'DATA', 'REUSE', 'DOCU', 'OI', 'OCAP', 'DECAP', 'OEXP', 'DEEXP', 'PCON', 'LEXP', 'TEXP', 'TOOL', 'SITE', 'SCED']].values
A = data['A'].values
Size = data['Size'].values
actual_effort = data['Actual Effort'].values

# Step 2: Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Fuzzy Logic Implementation
def fuzzy_membership(x, low, high):
    return max(0, min((x - low) / (high - low), 1))

def fuzzy_inference(data_point):
    low, medium, high = 0.2, 0.5, 0.8  # Define fuzzy ranges
    if data_point < low:
        return fuzzy_membership(data_point, 0, low)
    elif low <= data_point <= medium:
        return 1.0
    elif medium < data_point <= high:
        return fuzzy_membership(data_point, medium, high)
    else:
        return 0.0

effort_fuzzy = np.array([fuzzy_inference(value) for value in X_scaled.sum(axis=1)])

# Step 4: Artificial Neural Network (ANN) Implementation
X_train, X_test, A_train, A_test, Size_train, Size_test, actual_effort_train, actual_effort_test = train_test_split(X_scaled, A, Size, actual_effort, test_size=0.32, random_state=42)

ann = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
ann.fit(X_train, A_train * Size_train)

# Step 5: Hybrid Approach
effort_ann_train = ann.predict(X_train)
effort_ann_test = ann.predict(X_test)

effort_hybrid_train = (effort_fuzzy[:len(X_train)] + effort_ann_train) / 2
effort_hybrid_test = (effort_fuzzy[len(X_train):] + effort_ann_test) / 2

# Step 6: Evaluate Performance
target_mmx_train = A_train * Size_train * np.prod(X_train[:, :17], axis=1)
target_mmx_test = A_test * Size_test * np.prod(X_test[:, :17], axis=1)

mre_hybrid_test = np.mean(np.abs(actual_effort_test - effort_hybrid_test) / actual_effort_test)

# printing efforts for ontologies:

print("Efforts for all ontologies:")
for i in range(len(effort_hybrid_test)):
    print(f"Ontology {i+1}: {effort_hybrid_test[i]}")

print("MRE (Hybrid Approach):", mre_hybrid_test)
