
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Title
st.title("Customer Segmentation App")
st.markdown("Upload your marketing_campaign.csv file and explore clustering")

# File uploader
uploaded_file = st.file_uploader("Upload marketing_campaign.csv", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(data.head())

    # Select Features
    st.subheader("Select Features for Clustering")
    features = st.multiselect("Choose columns", data.select_dtypes(include=['int64', 'float64']).columns.tolist())

    if len(features) >= 2:
        X = data[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Choose Clustering Model
        st.subheader("Choose Clustering Model")
        model_choice = st.selectbox("Select model", ["KMeans", "DBSCAN", "Agglomerative"])

        if model_choice == "KMeans":
            k = st.slider("Number of Clusters (K)", 2, 10, 3)
            model = KMeans(n_clusters=k)
        elif model_choice == "DBSCAN":
            eps = st.slider("EPS", 0.1, 5.0, 0.5)
            min_samples = st.slider("Min Samples", 1, 10, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            k = st.slider("Number of Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=k)

        # Fit Model
        labels = model.fit_predict(X_scaled)
        X["Cluster"] = labels

        st.subheader("Clustered Data")
        st.write(X.head())

        # Plot
        st.subheader("Cluster Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=labels, palette="Set2", ax=ax)
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(f"{model_choice} Clustering")
        st.pyplot(fig)
    else:
        st.warning("Please select at least 2 numerical features for clustering.")
