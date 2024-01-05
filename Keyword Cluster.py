import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Download NLTK data (stopwords)
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(filtered_words)


def keyword_cluster(output_file=None, input_file=None, num_clusters=5):
    # Use an absolute path for the default "keywords.txt" file
    if input_file is None:
        input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/your keyword text directory/keywords.txt")

    # Set the default output location
    if output_file is None:
        output_file = os.path.join("your output path/directory", "keyword_clusters.txt")

    # Read keywords from the input file
    with open(input_file, 'r') as file:
        texts = file.readlines()

    # Preprocess text
    preprocessed_texts = [preprocess_text(text) for text in texts]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    # Assign cluster labels to each text
    clusters = kmeans.labels_

    # Display clusters
    for i in range(num_clusters):
        cluster_keywords = [preprocessed_texts[j] for j in range(len(texts)) if clusters[j] == i]
        print(f"Cluster {i + 1} Keywords: {', '.join(cluster_keywords)}\n")

    # Save clusters to the output file
    with open(output_file, 'w') as file:
        for i in range(num_clusters):
            cluster_keywords = [texts[j] for j in range(len(texts)) if clusters[j] == i]
            file.write(f"Cluster {i + 1} Keywords: {', '.join(cluster_keywords)}\n")

    # Plotting (optional)
    plt.scatter(tfidf_matrix[:, 0].toarray(), tfidf_matrix[:, 1].toarray(), c=clusters, cmap='viridis')
    plt.title('Keyword Clusters')
    plt.show()


# Example usage with default input file "keywords.txt" and default output location
keyword_cluster()
