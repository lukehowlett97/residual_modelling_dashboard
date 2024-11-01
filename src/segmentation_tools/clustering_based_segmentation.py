def clustering_segmentation(data, n_clusters):
    from sklearn.cluster import KMeans
    # Use clustering to segment the time series
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    labels = kmeans.labels_
    # Group data points by their cluster labels
    segments = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        segments.append(data[labels == label])
    return segments
