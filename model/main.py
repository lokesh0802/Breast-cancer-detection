import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle5 as pickle


def create_model(data): 
    X = data.drop(['diagnosis'], axis=1)
  
    # scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # perform clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    data['cluster'] = kmeans.fit_predict(X_scaled)
    
    return kmeans, scaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data


def main():
    data = get_clean_data()

    kmeans, scaler = create_model(data)

    # Save the clustering model and scaler
    with open('model/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
        
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
  

if __name__ == '__main__':
    main()
