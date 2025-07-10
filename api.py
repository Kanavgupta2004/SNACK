from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI()

# Load data ONCE (faster)
rfm_table = pd.read_csv("user_clusters.csv")
clean_snacks = pd.read_csv("clean_snacks.csv")
transactions = pd.read_csv("transactions.csv")

def recommend_for_user(user_id, n=5):
    if user_id not in rfm_table['user_id'].values:
        return None
    cluster = rfm_table.loc[rfm_table['user_id'] == user_id, 'cluster'].values[0]
    segment = rfm_table.loc[rfm_table['user_id'] == user_id, 'segment'].values[0]
    transactions_clusters = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
    top_snacks = (
        transactions_clusters[transactions_clusters['cluster'] == cluster]
        .groupby('snack_id').size().sort_values(ascending=False).head(n).index.tolist()
    )
    snack_details = clean_snacks[clean_snacks['snack_id'].isin(top_snacks)][['snack_name','brand','price']]
    result = {
        "user_id": user_id,
        "cluster": int(cluster),
        "segment": segment,
        "top_snacks": snack_details.to_dict(orient='records')
    }
    return result

@app.get("/")
def root():
    return {"message": "Snack Recommendation API is running!"}

@app.get("/recommend/{user_id}")
def get_recommendation(user_id: str):
    result = recommend_for_user(user_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No user found with user_id: {user_id}")
    return result
