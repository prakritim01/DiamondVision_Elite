import pandas as pd
import xgboost as xg
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_price_model():
    df = pd.read_csv('data/surat_diamonds.csv')
    
    # Encoding categorical data
    le = LabelEncoder()
    for col in ['cut', 'color', 'clarity']:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('surat_price_inr', axis=1)
    y = df['surat_price_inr']
    
    model = xg.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    
    # Save model and encoders
    with open('models/price_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return "✅ Pricing model trained and saved to models/."

if __name__ == "__main__":
    train_price_model()