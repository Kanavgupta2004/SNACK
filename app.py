# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ----- Load data -----
# @st.cache_data
# def load_data():
#     users = pd.read_csv("users.csv")
#     clusters = pd.read_csv("user_clusters.csv") if "user_clusters.csv" in os.listdir() else None
#     snacks = pd.read_csv("clean_snacks.csv")
#     return users, clusters, snacks

# import os
# users_df, rfm_table, clean_snacks = load_data()

# # ----- Helper function -----
# def recommend_for_user(user_id, n=5):
#     if user_id not in rfm_table['user_id'].values:
#         return None
#     cluster = rfm_table.loc[rfm_table['user_id'] == user_id, 'cluster'].values[0]
#     segment = rfm_table.loc[rfm_table['user_id'] == user_id, 'segment'].values[0]
#     # Find top snacks for that cluster
#     transactions = pd.read_csv("transactions.csv")
#     transactions = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
#     top_snacks = (
#         transactions[transactions['cluster'] == cluster]
#         .groupby('snack_id').size().sort_values(ascending=False).head(n).index.tolist()
#     )
#     snack_details = clean_snacks[clean_snacks['snack_id'].isin(top_snacks)][['snack_name','brand','price']]
#     return cluster, segment, snack_details.reset_index(drop=True)

# # ----- Streamlit UI -----
# st.title("Snack Recommendation Engine (Demo)")
# st.write("**Select a user to get personalized snack recommendations based on behavioral clustering (K-Means)!**")

# # Show user selection
# user_id = st.selectbox("Select a user ID:", users_df['user_id'].sort_values().tolist())

# if st.button("Get Recommendations"):
#     result = recommend_for_user(user_id)
#     if result:
#         cluster, segment, snack_details = result
#         st.success(f"User {user_id} is in Cluster {cluster} ({segment}). Top snacks for this segment:")
#         st.dataframe(snack_details)
#     else:
#         st.error(f"No data found for {user_id}")

# # ----- Visualizations -----
# st.markdown("---")
# st.subheader("Cluster/Segment Distribution")

# # Pie chart for segments
# seg_counts = rfm_table['segment'].value_counts()
# fig1, ax1 = plt.subplots()
# ax1.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('tab10'))
# ax1.axis('equal')
# st.pyplot(fig1)

# # Cluster means table
# st.subheader("Cluster Means (Recency, Frequency, Monetary)")
# st.dataframe(rfm_table.groupby('segment')[['recency', 'frequency', 'monetary']].mean().round(2))

# # ---- Segment Insights Section ----
# st.markdown("---")
# st.header("Segment Insights & Recommendations")

# segment_summary = rfm_table.groupby('segment')[['recency', 'frequency', 'monetary']].mean().round(2)
# segment_summary['num_users'] = rfm_table['segment'].value_counts()
# segment_summary = segment_summary.reset_index()

# insight_map = {
#     "VIPs / Power Users": "These are our best customers. They buy often, spend the most, and engage frequently. Prioritize them for loyalty rewards and early access to offers.",
#     "Churn Risk": "These users haven’t purchased in a long time and are at risk of leaving. Consider special win-back promotions.",
#     "Core Regulars": "Steady and engaged customers. Maintain engagement with regular offers.",
#     "New/Light Users": "Recent or infrequent buyers. Welcome them and nurture to boost loyalty."
# }

# for _, row in segment_summary.iterrows():
#     st.subheader(f"{row['segment']} ({int(row['num_users'])} users)")
#     st.write(f"**Recency:** {row['recency']} days | **Frequency:** {row['frequency']} | **Monetary:** ${row['monetary']}")
#     st.info(insight_map.get(row['segment'], ""))
    
#     # Show top snacks for this segment
#     cluster_id = rfm_table[rfm_table['segment'] == row['segment']]['cluster'].values[0]
#     transactions = pd.read_csv("transactions.csv")
#     transactions_clusters = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
#     top_snacks = (
#         transactions_clusters[transactions_clusters['cluster'] == cluster_id]
#         .groupby('snack_id').size().sort_values(ascending=False).head(5)
#     )
#     snack_names = clean_snacks.set_index('snack_id').loc[top_snacks.index]['snack_name']
#     st.write("**Top snacks for this segment:**")
#     for snack in snack_names:
#         st.write(f"- {snack}")


# st.markdown("---")
# st.subheader("Top Snacks by Segment")

# # Select segment for visualization
# unique_segments = rfm_table['segment'].unique().tolist()
# selected_segment = st.selectbox("Select a segment to view top snacks:", unique_segments)

# # Get cluster number for the selected segment
# segment_to_cluster = rfm_table.drop_duplicates(['cluster', 'segment']).set_index('segment')['cluster'].to_dict()
# cluster_id = segment_to_cluster[selected_segment]

# # Get top 7 snacks for this cluster
# transactions = pd.read_csv("transactions.csv")
# transactions_clusters = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
# top_snacks = (
#     transactions_clusters[transactions_clusters['cluster'] == cluster_id]
#     .groupby('snack_id').size().sort_values(ascending=False).head(7)
# )

# # Merge with clean_snacks to get snack info
# snack_info = (
#     pd.DataFrame({'snack_id': top_snacks.index, 'purchase_count': top_snacks.values})
#     .merge(clean_snacks[['snack_id', 'snack_name', 'brand', 'price']], on='snack_id', how='left')
#     .dropna(subset=['snack_name'])
# )

# # Make y-labels like "Snack Name ($Price, Brand)"
# snack_info['snack_label'] = snack_info.apply(
#     lambda x: f"{x['snack_name']} (${x['price']:.2f}, {x['brand']})", axis=1
# )

# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(x=snack_info['purchase_count'], y=snack_info['snack_label'], ax=ax, palette='Blues_d')
# ax.set_xlabel("Purchase Count")
# ax.set_ylabel("Snack (Price, Brand)")
# ax.set_title(f"Top Snacks for Segment: {selected_segment}")

# # Add purchase count at end of each bar
# for i, v in enumerate(snack_info['purchase_count']):
#     ax.text(v + 0.05, i, str(int(v)), color='black', va='center')

# st.pyplot(fig)

# st.caption("Bar length shows how many times each snack was purchased by users in this segment. Brand and price are shown for business clarity.")



# # 3D plot as optional (or 2D for Streamlit simplicity)
# import plotly.express as px
# fig2 = px.scatter_3d(
#     rfm_table, x='recency', y='frequency', z='monetary', color='segment',
#     title="User RFM Segments (3D Scatter)",
#     width=700, height=500
# )
# st.plotly_chart(fig2)


# 

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import os

# # ----- Load data -----
# @st.cache_data
# def load_data():
#     users = pd.read_csv("users.csv")
#     clusters = pd.read_csv("user_clusters.csv") if "user_clusters.csv" in os.listdir() else None
#     snacks = pd.read_csv("clean_snacks.csv")
#     return users, clusters, snacks

# users_df, rfm_table, clean_snacks = load_data()

# # ----- Helper function -----
# def recommend_for_user(user_id, n=5):
#     if user_id not in rfm_table['user_id'].values:
#         return None
#     cluster = rfm_table.loc[rfm_table['user_id'] == user_id, 'cluster'].values[0]
#     segment = rfm_table.loc[rfm_table['user_id'] == user_id, 'segment'].values[0]
#     # Find top snacks for that cluster
#     transactions = pd.read_csv("transactions.csv")
#     transactions = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
#     top_snacks = (
#         transactions[transactions['cluster'] == cluster]
#         .groupby('snack_id').size().sort_values(ascending=False).head(n).index.tolist()
#     )
#     snack_details = clean_snacks[clean_snacks['snack_id'].isin(top_snacks)][['snack_name','brand','price']]
#     return cluster, segment, snack_details.reset_index(drop=True)

# # ----- Streamlit UI -----
# st.title("Snack Recommendation Engine (Demo)")
# st.write("**Select a user to get personalized snack recommendations based on behavioral clustering (K-Means)!**")

# # -----------------------------------
# # 1. Autocomplete Feature (Search by snack name)
# search_query = st.text_input("Search Snacks (autocomplete):")
# if search_query:
#     # Filter clean_snacks for matches
#     matching_snacks = clean_snacks[clean_snacks['snack_name'].str.contains(search_query, case=False, na=False)]
#     st.write("### Suggestions:")
#     st.dataframe(matching_snacks[['snack_name', 'brand', 'price']])

#     # Add the first matching snack to 'recently_viewed' (if it's not already there)
#     if len(matching_snacks) > 0:
#         viewed_snack = matching_snacks.iloc[0]['snack_name']
#         if viewed_snack not in st.session_state.recently_viewed:
#             st.session_state.recently_viewed.append(viewed_snack)

# # -----------------------------------
# # 2. Filters: Brand/Category/Price
# brand = st.selectbox("Select Brand", options=clean_snacks['brand'].dropna().unique())
# category = st.selectbox("Select Category", options=clean_snacks['category'].dropna().unique())
# price_range = st.slider("Select Price Range", 0.0, 50.0, (0.0, 10.0))

# # Apply filters to snacks
# filtered_snacks = clean_snacks[
#     (clean_snacks['brand'] == brand) &
#     (clean_snacks['category'] == category) &
#     (clean_snacks['price'] >= price_range[0]) &
#     (clean_snacks['price'] <= price_range[1])
# ]

# # Check if the filter returned any results
# if filtered_snacks.empty:
#     st.write("No snacks match your filter criteria.")
# else:
#     st.write("### Filtered Snacks:")
#     st.dataframe(filtered_snacks[['snack_name', 'brand', 'category', 'price']])

# # -----------------------------------

# # Initialize 'recently_viewed' if it doesn't exist
# if "recently_viewed" not in st.session_state:
#     st.session_state.recently_viewed = []

# # Allow the user to search for snacks (autocomplete)
# search_query = st.text_input("Search Snacks:")
# if search_query:
#     matching_snacks = clean_snacks[clean_snacks['snack_name'].str.contains(search_query, case=False, na=False)]
#     st.write("### Suggestions:")
#     st.dataframe(matching_snacks[['snack_name', 'brand', 'price']])

#     # Automatically add the first match to 'recently_viewed' (if it's not already there)
#     if len(matching_snacks) > 0:
#         viewed_snack = matching_snacks.iloc[0]['snack_name']  # Take the first match
#         if viewed_snack not in st.session_state.recently_viewed:
#             st.session_state.recently_viewed.append(viewed_snack)

# # Manually add snack to recently viewed list (for button clicks)
# if st.button("View Snack: Doritos"):  # This could be dynamic
#     snack_name = "Doritos"
#     if snack_name not in st.session_state.recently_viewed:
#         st.session_state.recently_viewed.append(snack_name)

# # Display the recently viewed snacks
# st.write("### Recently Viewed Snacks:")
# st.write(st.session_state.recently_viewed)


# # -----------------------------------
# # 3. Personalized Recommendations based on user segment
# user_id = st.selectbox("Select a User ID", options=users_df['user_id'].sort_values().tolist())

# if user_id:
#     result = recommend_for_user(user_id)
#     if result:
#         cluster, segment, snack_details = result
#         st.success(f"User {user_id} is in Cluster {cluster} ({segment}). Top snacks for this segment:")
#         st.dataframe(snack_details)
#     else:
#         st.error(f"No data found for {user_id}")

# # -----------------------------------
# # 4. Visualizations: Cluster/Segment Distribution

# st.markdown("---")
# st.subheader("Cluster/Segment Distribution")

# # Pie chart for segments
# seg_counts = rfm_table['segment'].value_counts()
# fig1, ax1 = plt.subplots()
# ax1.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('tab10'))
# ax1.axis('equal')
# st.pyplot(fig1)

# # Cluster means table
# st.subheader("Cluster Means (Recency, Frequency, Monetary)")
# st.dataframe(rfm_table.groupby('segment')[['recency', 'frequency', 'monetary']].mean().round(2))

# # -----------------------------------
# # 5. Segment Insights Section
# st.markdown("---")
# st.header("Segment Insights & Recommendations")

# segment_summary = rfm_table.groupby('segment')[['recency', 'frequency', 'monetary']].mean().round(2)
# segment_summary['num_users'] = rfm_table['segment'].value_counts()
# segment_summary = segment_summary.reset_index()

# insight_map = {
#     "VIPs / Power Users": "These are our best customers. They buy often, spend the most, and engage frequently. Prioritize them for loyalty rewards and early access to offers.",
#     "Churn Risk": "These users haven’t purchased in a long time and are at risk of leaving. Consider special win-back promotions.",
#     "Core Regulars": "Steady and engaged customers. Maintain engagement with regular offers.",
#     "New/Light Users": "Recent or infrequent buyers. Welcome them and nurture to boost loyalty."
# }

# for _, row in segment_summary.iterrows():
#     st.subheader(f"{row['segment']} ({int(row['num_users'])} users)")
#     st.write(f"**Recency:** {row['recency']} days | **Frequency:** {row['frequency']} | **Monetary:** ${row['monetary']}")
#     st.info(insight_map.get(row['segment'], ""))
    
#     # Show top snacks for this segment
#     cluster_id = rfm_table[rfm_table['segment'] == row['segment']]['cluster'].values[0]
#     transactions = pd.read_csv("transactions.csv")
#     transactions_clusters = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
#     top_snacks = (
#         transactions_clusters[transactions_clusters['cluster'] == cluster_id]
#         .groupby('snack_id').size().sort_values(ascending=False).head(5)
#     )
#     snack_names = clean_snacks.set_index('snack_id').loc[top_snacks.index]['snack_name']
#     st.write("**Top snacks for this segment:**")
#     for snack in snack_names:
#         st.write(f"- {snack}")

# # -----------------------------------
# # 6. Top Snacks by Segment (Bar chart)
# st.markdown("---")
# st.subheader("Top Snacks by Segment")

# # Select segment for visualization
# unique_segments = rfm_table['segment'].unique().tolist()
# selected_segment = st.selectbox("Select a segment to view top snacks:", unique_segments)

# # Get cluster number for the selected segment
# segment_to_cluster = rfm_table.drop_duplicates(['cluster', 'segment']).set_index('segment')['cluster'].to_dict()
# cluster_id = segment_to_cluster[selected_segment]

# # Get top 7 snacks for this cluster
# transactions = pd.read_csv("transactions.csv")
# transactions_clusters = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
# top_snacks = (
#     transactions_clusters[transactions_clusters['cluster'] == cluster_id]
#     .groupby('snack_id').size().sort_values(ascending=False).head(7)
# )

# # Merge with clean_snacks to get snack info
# snack_info = (
#     pd.DataFrame({'snack_id': top_snacks.index, 'purchase_count': top_snacks.values})
#     .merge(clean_snacks[['snack_id', 'snack_name', 'brand', 'price']], on='snack_id', how='left')
#     .dropna(subset=['snack_name'])
# )

# # Make y-labels like "Snack Name ($Price, Brand)"
# snack_info['snack_label'] = snack_info.apply(
#     lambda x: f"{x['snack_name']} (${x['price']:.2f}, {x['brand']})", axis=1
# )

# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(x=snack_info['purchase_count'], y=snack_info['snack_label'], ax=ax, palette='Blues_d')
# ax.set_xlabel("Purchase Count")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# ----- Load data -----
@st.cache_data
def load_data():
    users = pd.read_csv("users.csv")
    clusters = pd.read_csv("user_clusters.csv") if "user_clusters.csv" in os.listdir() else None
    snacks = pd.read_csv("clean_snacks.csv")
    return users, clusters, snacks

users_df, rfm_table, clean_snacks = load_data()

# ----- Helper function -----
def recommend_for_user(user_id, n=5):
    if user_id not in rfm_table['user_id'].values:
        return None
    cluster = rfm_table.loc[rfm_table['user_id'] == user_id, 'cluster'].values[0]
    segment = rfm_table.loc[rfm_table['user_id'] == user_id, 'segment'].values[0]
    # Find top snacks for that cluster
    transactions = pd.read_csv("transactions.csv")
    transactions = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
    top_snacks = (
        transactions[transactions['cluster'] == cluster]
        .groupby('snack_id').size().sort_values(ascending=False).head(n).index.tolist()
    )
    snack_details = clean_snacks[clean_snacks['snack_id'].isin(top_snacks)][['snack_name','brand','price']]
    return cluster, segment, snack_details.reset_index(drop=True)

# ----- Streamlit UI -----
st.title("Snack Recommendation Engine (Demo)")
st.write("**Select a user to get personalized snack recommendations based on behavioral clustering (K-Means)!**")

# -----------------------------------
# 1. Autocomplete Feature (Search by snack name)
search_query = st.text_input("Search Snacks (autocomplete):")
if search_query:
    # Filter clean_snacks for matches
    matching_snacks = clean_snacks[clean_snacks['snack_name'].str.contains(search_query, case=False, na=False)]
    st.write("### Suggestions:")
    st.dataframe(matching_snacks[['snack_name', 'brand', 'price']])

    # Add the first matching snack to 'recently_viewed' (if it's not already there)
    if len(matching_snacks) > 0:
        viewed_snack = matching_snacks.iloc[0]['snack_name']
        if viewed_snack not in st.session_state.recently_viewed:
            st.session_state.recently_viewed.append(viewed_snack)

# -----------------------------------
# 2. Filters: Brand/Category/Price
brand = st.selectbox("Select Brand", options=clean_snacks['brand'].dropna().unique())
category = st.selectbox("Select Category", options=clean_snacks['category'].dropna().unique())
price_range = st.slider("Select Price Range", 0.0, 50.0, (0.0, 10.0))

# Apply filters to snacks
filtered_snacks = clean_snacks[
    (clean_snacks['brand'] == brand) &
    (clean_snacks['category'] == category) &
    (clean_snacks['price'] >= price_range[0]) &
    (clean_snacks['price'] <= price_range[1])
]

# Check if the filter returned any results
if filtered_snacks.empty:
    st.write("No snacks match your filter criteria.")
else:
    st.write("### Filtered Snacks:")
    st.dataframe(filtered_snacks[['snack_name', 'brand', 'category', 'price']])

# -----------------------------------

# Initialize 'recently_viewed' if it doesn't exist
if "recently_viewed" not in st.session_state:
    st.session_state.recently_viewed = []

# Allow the user to search for snacks (autocomplete)
if search_query:
    matching_snacks = clean_snacks[clean_snacks['snack_name'].str.contains(search_query, case=False, na=False)]
    st.write("### Suggestions:")
    st.dataframe(matching_snacks[['snack_name', 'brand', 'price']])

    # Automatically add the first match to 'recently_viewed' (if it's not already there)
    if len(matching_snacks) > 0:
        viewed_snack = matching_snacks.iloc[0]['snack_name']  # Take the first match
        if viewed_snack not in st.session_state.recently_viewed:
            st.session_state.recently_viewed.append(viewed_snack)

# Manually add snack to recently viewed list (for button clicks)
if st.button("View Snack: Doritos"):  # This could be dynamic
    snack_name = "Doritos"
    if snack_name not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.append(snack_name)

# Display the recently viewed snacks
st.write("### Recently Viewed Snacks:")
st.write(st.session_state.recently_viewed)


# -----------------------------------
# 3. Personalized Recommendations based on user segment
user_id = st.selectbox("Select a User ID", options=users_df['user_id'].sort_values().tolist())

if user_id:
    result = recommend_for_user(user_id)
    if result:
        cluster, segment, snack_details = result
        st.success(f"User {user_id} is in Cluster {cluster} ({segment}). Top snacks for this segment:")
        st.dataframe(snack_details)
    else:
        st.error(f"No data found for {user_id}")

# -----------------------------------
# 4. Visualizations: Cluster/Segment Distribution

st.markdown("---")
st.subheader("Cluster/Segment Distribution")

# Pie chart for segments
seg_counts = rfm_table['segment'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('tab10'))
ax1.axis('equal')
st.pyplot(fig1)

# Cluster means table
st.subheader("Cluster Means (Recency, Frequency, Monetary)")
st.dataframe(rfm_table.groupby('segment')[['recency', 'frequency', 'monetary']].mean().round(2))

# -----------------------------------
# 5. Segment Insights Section
st.markdown("---")
st.header("Segment Insights & Recommendations")

segment_summary = rfm_table.groupby('segment')[['recency', 'frequency', 'monetary']].mean().round(2)
segment_summary['num_users'] = rfm_table['segment'].value_counts()
segment_summary = segment_summary.reset_index()

insight_map = {
    "VIPs / Power Users": "These are our best customers. They buy often, spend the most, and engage frequently. Prioritize them for loyalty rewards and early access to offers.",
    "Churn Risk": "These users haven’t purchased in a long time and are at risk of leaving. Consider special win-back promotions.",
    "Core Regulars": "Steady and engaged customers. Maintain engagement with regular offers.",
    "New/Light Users": "Recent or infrequent buyers. Welcome them and nurture to boost loyalty."
}

for _, row in segment_summary.iterrows():
    st.subheader(f"{row['segment']} ({int(row['num_users'])} users)")
    st.write(f"**Recency:** {row['recency']} days | **Frequency:** {row['frequency']} | **Monetary:** ${row['monetary']}")
    st.info(insight_map.get(row['segment'], ""))
    
    # Show top snacks for this segment
    cluster_id = rfm_table[rfm_table['segment'] == row['segment']]['cluster'].values[0]
    transactions = pd.read_csv("transactions.csv")
    transactions_clusters = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
    top_snacks = (
        transactions_clusters[transactions_clusters['cluster'] == cluster_id]
        .groupby('snack_id').size().sort_values(ascending=False).head(5)
    )
    snack_names = clean_snacks.set_index('snack_id').loc[top_snacks.index]['snack_name']
    st.write("**Top snacks for this segment:**")
    for snack in snack_names:
        st.write(f"- {snack}")

# -----------------------------------
# 6. Top Snacks by Segment (Bar chart)
st.markdown("---")
st.subheader("Top Snacks by Segment")

# Select segment for visualization
unique_segments = rfm_table['segment'].unique().tolist()
selected_segment = st.selectbox("Select a segment to view top snacks:", unique_segments)

# Get cluster number for the selected segment
segment_to_cluster = rfm_table.drop_duplicates(['cluster', 'segment']).set_index('segment')['cluster'].to_dict()
cluster_id = segment_to_cluster[selected_segment]

# Get top 7 snacks for this cluster
transactions = pd.read_csv("transactions.csv")
transactions_clusters = transactions.merge(rfm_table[['user_id', 'cluster']], on='user_id')
top_snacks = (
    transactions_clusters[transactions_clusters['cluster'] == cluster_id]
    .groupby('snack_id').size().sort_values(ascending=False).head(7)
)

# Merge with clean_snacks to get snack info
snack_info = (
    pd.DataFrame({'snack_id': top_snacks.index, 'purchase_count': top_snacks.values})
    .merge(clean_snacks[['snack_id', 'snack_name', 'brand', 'price']], on='snack_id', how='left')
    .dropna(subset=['snack_name'])
)

# Make y-labels like "Snack Name ($Price, Brand)"
snack_info['snack_label'] = snack_info.apply(
    lambda x: f"{x['snack_name']} (${x['price']:.2f}, {x['brand']})", axis=1
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=snack_info['purchase_count'], y=snack_info['snack_label'], ax=ax, palette='Blues_d')
ax.set_xlabel("Purchase Count")
ax.set_ylabel("Snack (Price, Brand)")
ax.set_title(f"Top Snacks for Segment: {selected_segment}")

# Add purchase count at end of each bar
for i, v in enumerate(snack_info['purchase_count']):
    ax.text(v + 0.05, i, str(int(v)), color='black', va='center')

st.pyplot(fig)

st.caption("Bar length shows how many times each snack was purchased by users in this segment. Brand and price are shown for business clarity.")

# -----------------------------------
# 7. 3D Scatter Plot (optional)
fig2 = px.scatter_3d(
    rfm_table, x='recency', y='frequency', z='monetary', color='segment',
    title="User RFM Segments (3D Scatter)",
    width=700, height=500
)
st.plotly_chart(fig2)
