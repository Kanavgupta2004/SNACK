import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD & CLEAN SNACKS DATA
# =========================
snacks_df = pd.read_csv("ProductExport.csv")
clean_snacks = snacks_df[["productId", "name", "price__value", "productClass", "brand"]].copy()
clean_snacks.columns = ["snack_id", "snack_name", "price", "category", "brand"]
clean_snacks = clean_snacks.dropna(subset=["snack_id", "snack_name", "price"])

# =========================
# 2. GENERATE DUMMY USERS
# =========================
num_users = 200
user_ids = [f"user_{i+1}" for i in range(num_users)]
ages = np.random.randint(18, 60, num_users)
genders = np.random.choice(["Male", "Female"], num_users)
locations = np.random.choice(["Delhi", "Mumbai", "Bangalore", "Chennai"], num_users)
users_df = pd.DataFrame({
    "user_id": user_ids,
    "age": ages,
    "gender": genders,
    "location": locations
})
users_df.to_csv("users.csv", index=False)

# =========================
# 3. GENERATE DUMMY TRANSACTIONS
# =========================
snack_ids = clean_snacks['snack_id'].tolist()
transactions = []
for user in user_ids:
    for _ in range(random.randint(5, 18)):
        snack = random.choice(snack_ids)
        days_ago = random.randint(1, 120)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0,59))
        quantity = random.randint(1, 4)
        transactions.append({
            "user_id": user,
            "snack_id": snack,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "quantity": quantity
        })
transactions_df = pd.DataFrame(transactions)
transactions_df.to_csv("transactions.csv", index=False)

# =========================
# 4. BUILD RFM FEATURES
# =========================
transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
rfm_data = transactions_df.merge(clean_snacks[['snack_id', 'price']], on='snack_id', how='left')

now = transactions_df['timestamp'].max()

# Recency
recency_df = rfm_data.groupby('user_id')['timestamp'].max().reset_index()
recency_df['recency'] = (now - recency_df['timestamp']).dt.days
recency_df = recency_df[['user_id', 'recency']]

# Frequency
frequency_df = rfm_data.groupby('user_id').size().reset_index(name='frequency')

# Monetary
rfm_data['spend'] = rfm_data['price'] * rfm_data['quantity']
monetary_df = rfm_data.groupby('user_id')['spend'].sum().reset_index(name='monetary')

# Merge RFM
rfm_table = recency_df.merge(frequency_df, on='user_id').merge(monetary_df, on='user_id')

# =========================
# 5. K-MEANS CLUSTERING
# =========================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_table[['recency', 'frequency', 'monetary']])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_table['cluster'] = kmeans.fit_predict(rfm_scaled)

# =========================
# 6. LABEL CLUSTERS (Business-Friendly Names)
# =========================
# First, get a summary to identify which cluster is which
cluster_summary = rfm_table.groupby('cluster')[['recency', 'frequency', 'monetary']].mean().round(2)
print("\nCluster means (use this to double-check labels):\n", cluster_summary)

# Assign labels -- you may want to adjust mapping based on above summary!
cluster_labels = {
    0: "Core Regulars",
    1: "New/Light Users",
    2: "VIPs / Power Users",
    3: "Churn Risk"
}
rfm_table['segment'] = rfm_table['cluster'].map(cluster_labels)
rfm_table.to_csv("user_clusters.csv", index=False)   # <-- This line should save the 'segment' column


# =========================
# 7. VISUALIZATIONS
# =========================

# Pie chart: Segment distribution
segment_counts = rfm_table['segment'].value_counts()
segment_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('tab10'))
plt.ylabel("")
plt.title("User Segments (Cluster Labels)")
plt.show()

# 3D RFM Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    rfm_table['recency'], rfm_table['frequency'], rfm_table['monetary'],
    c=rfm_table['cluster'], cmap='tab10', s=50, alpha=0.7
)
ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary (total spend)')
ax.set_title('User Segments: RFM Clusters')
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()

# Cluster sizes barplot
plt.figure(figsize=(6,4))
sns.countplot(x='cluster', data=rfm_table, palette='tab10')
plt.title('Number of Users in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Users')
plt.show()

# Segment summary table for your PPT/report
segment_summary = rfm_table.groupby('segment')[['recency', 'frequency', 'monetary']].mean().round(2)
segment_summary['num_users'] = rfm_table['segment'].value_counts()
print("\nBusiness Segment Summary:\n", segment_summary)

# =========================
# 8. TOP SNACKS BY CLUSTER
# =========================
transactions_clusters = transactions_df.merge(rfm_table[['user_id', 'cluster']], on='user_id')
top_snacks_per_cluster = (
    transactions_clusters.groupby(['cluster', 'snack_id'])
    .size()
    .reset_index(name='purchase_count')
    .sort_values(['cluster', 'purchase_count'], ascending=[True, False])
)

def get_top_snacks(cluster, n=5):
    snacks = top_snacks_per_cluster[top_snacks_per_cluster['cluster'] == cluster]['snack_id'].head(n).tolist()
    return snacks

# =========================
# 9. RECOMMENDATION FUNCTION
# =========================
def recommend_for_user(user_id, n=5):
    # Get user's cluster
    cluster = rfm_table.loc[rfm_table['user_id'] == user_id, 'cluster'].values
    if len(cluster) == 0:
        return f"No user found with user_id: {user_id}"
    cluster = cluster[0]
    # Get top snacks for cluster
    top_ids = get_top_snacks(cluster, n)
    snacks_info = clean_snacks[clean_snacks['snack_id'].isin(top_ids)][['snack_name', 'brand', 'price']]
    return {
        "user_id": user_id,
        "cluster": int(cluster),
        "segment": rfm_table.loc[rfm_table['user_id'] == user_id, 'segment'].values[0],
        "top_snacks": snacks_info.reset_index(drop=True)
    }

# =========================
# 10. DEMO/INTERACTION
# =========================
if __name__ == "__main__":
    print("\nSample Recommendations:")
    for uid in ["user_1", "user_50", "user_123"]:
        result = recommend_for_user(uid)
        print(f"\nRecommendations for {uid} (Cluster {result['cluster']}, Segment {result['segment']}):")
        print(result['top_snacks'])

    # Interactive user prompt
    while True:
        inp = input("\nEnter a user_id to get recommendations (or 'exit' to quit): ")
        if inp == "exit":
            break
        out = recommend_for_user(inp)
        if isinstance(out, dict):
            print(f"User {inp} is in cluster {out['cluster']} ({out['segment']}). Top snacks:")
            print(out['top_snacks'])
        else:
            print(out)
