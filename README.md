# Natural-scene-image-retrieval
# Content Based Image Retrieval System

Image retrieval techniques often rely on labels, which can lead to mistakes as they are not based on the actual content of the image. Additionally, traditional methods require manual image annotation, which is laborious and time-consuming.

This repository demonstrates how to build a Content Based Image Retrieval (CBIR) system from scratch:

- Convert the RGB image to the HSI format.
- Solve the feature extraction problem by selecting 300 random points uniformly. Each point represents a window used to extract the mean, standard deviation, and homogeneity obtained from the co-occurrence matrix.
- Apply the K-means algorithm to cluster all the extracted features.
- Train a Bayesian classifier with the clustered data.
- Generate an indexed database. For each image, enumerate the times each feature has been classified as any of the 9 features. The resulting database will look like this:

![Indexed Database](asserts\indexes_Db.png)

Finally, to retrieve similar images given any image, follow these steps:

1. Repeat the process of converting the image to the HSI format and extracting the nine descriptive features.
2. Calculate the Euclidean distance between each the describing vector and the indexed database.
3. Return the images with the closest distances to find the most similar images based on their content.

This approach ensures that the retrieved images are similar in terms of their content, rather than relying on labels or manual annotations. By using the Euclidean distance, we can quantify the similarity between feature vectors and make accurate comparisons.

Here are some diagrams to better understand the process:

![Indexed Diagram](asserts\indexed_diagram.png)

![Final Diagram](asserts\final_diagram.png)

