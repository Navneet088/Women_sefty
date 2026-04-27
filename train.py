import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('CrimesOnWomenData.csv')
if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])

# Feature Engineering
crime_cols = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']
weights = {'Rape':1.0, 'K&A':0.9, 'DD':0.95, 'AoW':0.7, 'AoM':0.5, 'DV':0.8, 'WT':0.95}
df['Safety_Score'] = sum(df[col] * weights[col] for col in crime_cols)

# Create Labels
low_t, high_t = df['Safety_Score'].quantile([0.33, 0.67])
def categorize(s):
    if s <= low_t: return 'Low'
    return 'High' if s > high_t else 'Mid'
df['Risk_Category'] = df['Safety_Score'].apply(categorize)

# Prep Data
le = LabelEncoder()
y = le.fit_transform(df['Risk_Category'])
X = df[crime_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Best Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# SAVE EVERYTHING
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(" Model and Scaler saved successfully!")