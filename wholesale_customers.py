# Part 2: Cluster Analysis
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(['Channel', 'Region'], axis='columns')
    return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    df = df.describe().loc[['mean','std', 'min', 'max']].transpose()
    df['mean'] = df['mean'].round().astype(int)
    df['std'] = df['std'].round().astype(int)
    return df

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    result = df.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()))
    return result

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
    # data = df.values
    clf = KMeans(n_clusters=k, init='random').fit(df)
    labels = clf.labels_
    
    y = pd.Series(labels)
    return y
    
# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    clf = KMeans(n_clusters=k, init="k-means++").fit(df)
    labels = clf.labels_
    y = pd.Series(labels)
    return y

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    # data = df.values
    clf = AgglomerativeClustering(n_clusters=k).fit(df)
    labels = clf.labels_
    y = pd.Series(labels)
    return y

# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X,y):
    return silhouette_score(X,y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    df_standardized = standardize(df)

# Define k values and number of iterations
    k_values = [3, 5, 10]
    iterations = 10

    # Prepare an empty list to store results
    results = []

    # Perform k-means and agglomerative clustering
    for algorithm in ['Kmeans', 'Agglomerative']:
        for k in k_values:
            for iteration in range(iterations):
                # Clustering
                if algorithm == 'Kmeans':
                    model_original = KMeans(n_clusters=k).fit(df)
                    model_standardized = KMeans(n_clusters=k).fit(df_standardized)
                else:
                    model_original = AgglomerativeClustering(n_clusters=k).fit(df)
                    model_standardized = AgglomerativeClustering(n_clusters=k).fit(df_standardized)

                # Calculate clustering scores
                original_score = clustering_score(df, model_original.labels_)
                standardized_score = clustering_score(df_standardized, model_standardized.labels_)

                # Append results to the list
                results.extend([
                    {'Algorithm': algorithm, 'Data': 'Original', 'k': k, 'Silhouette score': original_score},
                    {'Algorithm': algorithm, 'Data': 'Standardized', 'k': k, 'Silhouette score': standardized_score}
                ])

    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    return results_df
    

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Silhouette score'].max()

# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df)
    labels = kmeans.labels_
    unique_labels = sorted(set(labels))

    # Create scatter plots for each pair of attributes
    attributes = list(df.columns)
    num_attributes = len(attributes)

    with PdfPages("scatter_plots.pdf") as pdf:
        for i in range(num_attributes):
            for j in range(i+1, num_attributes):
                fig, ax = plt.subplots(figsize=(6, 6))

                for label in unique_labels:
                    ax.scatter(
                        df.iloc[labels == label, j],
                        df.iloc[labels == label, i],
                        label=f'Cluster {label}',
                        c=[plt.cm.viridis(label / (k-1))]  # Set color for each cluster
                    )
                ax.set_xlabel(attributes[j])
                ax.set_ylabel(attributes[i])

                plt.subplots_adjust(left=0.15)  # Adjust the space on the left side of the plot
                ax.legend()  # Add a legend
                pdf.savefig(fig)
                plt.close(fig)
