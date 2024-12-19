
# Find your beer community
## A Comprehensive Analysis of Beer Characteristics

### Abstract

This study employs a range of analytical techniques to uncover trends, preferences, and perceptions within the beer industry. Using natural language processing (NLP) methods, we analyze reviews from BeerAdvocate to identify distinct clusters of beers and explore the key words and characteristics that differentiate them. For each cluster, we highlight the primary flavor and appearance features that define it and compare these characteristics across clusters to identify patterns and differences. Additionally, we investigate how these clusters evolve seasonally, over time, and across different countries.

We also examine the emotional responses associated with each beer cluster, focusing on how specific appearance features and flavor profiles elicit distinct emotions.

To deepen our understanding, we apply principal component analysis (PCA) to extract and interpret the main components that define each cluster. The ultimate goal of this research is to develop a model capable of assigning a beer cluster to a user based on their preferences, allowing the user to discover the cluster that best aligns with their tastes through a web-based questionnaire. This research provides valuable insights for brewers, marketers, and consumers, enhancing the understanding of beer diversity and evolving consumer preferences.

### Some research questions we aim to answer

1. Can beers be clustered based on the characteristics of their reviews?
2. What are the main features defining main beer categories?
3. How do the flavor and appearance features of beers differ across clusters?
4. What are the seasonal, annual, and geographic patterns in beer preferences and reviews?
5. Do they change over clusters?
6. Given a review, what is the mood of the user who wrote it?
7. Is there a connection between beer flavors and the emotional tone of the commentary?
8. How do emotional responses vary across different beer categories and clusters?
9. Can we predict a user's preferred beer cluster based on their tastes and preferences?
10. Which questions are most relevant for assigning a user to a beer cluster?

### Proposed additional datasets

We did not include additional datasets in this analysis. Our focus remained on the BeerAdvocate dataset due to several key considerations: managing computational complexity, ensuring the coherence of the datastory, and maintaining consistency in the language used within the reviews.

### Data story

Our concept is inspired by the website https://www.16personalities.com/free-personality-test or https://www.whatsyourwhisky.com/de-ch/quiz. However, instead of assigning users a personality type or a whisky type, our goal is to match them with a specific beer cluster based on their preferences (a community). This personalized approach will help users discover beer styles that best align with their taste profiles, enhancing their overall beer experience. 

To achieve this, we first need to extract the most important words from user reviews using Natural Language Processing (NLP) techniques. These key terms help us cluster beers by identifying the most distinctive descriptors. Using a similarity graph, the beers are assigned clusters. Once the beer clusters are established, we analyze them to uncover their defining characteristics, recognize the emotions expressed in their reviews, track their evolution over time, and explore their geographical distribution. This process mirrors how the 16Personalities website explains personality types (once a user is assigned a cluster = a personality type).

With these clusters and their attributes clearly identified, the next step is to create a series of targeted questions. These questions will be designed to distinguish between clusters effectively, enabling us to assign users to the beer community that best fits their taste preferences based on their responses.

### Methods

Here is a list of the main methods used in this project. We did not include as methods all the code used to process panda dataframes and show our graphs (basic code from the course).

#### Clustering

Generating a similarity graph and clustering beers based on review characteristics using natural language processing (NLP) methods. The process involves tokenizing reviews, filtering discriminative words, computing cosine similarity, constructing a similarity graph, and applying the Louvain algorithm for clustering, with computationally intensive tasks optimized for parallel processing on an HPC cluster.

Goal: find main clusters of beers in the BeerAdvocate dataset

Folder: scr/scripts/graph_and_clustering

#### Natural Language Processing (NLP)

- <b> Extract main beer charcteristics </b>

The method involves preprocessing, filtering, and analyzing beer reviews. First, reviews are preprocessed by converting text to lowercase, tokenizing, removing punctuation and stopwords, and lemmatizing words. Next, beers with a minimum number of reviews are selected, and reviews are limited to a maximum per beer for computational efficiency. Finally, word frequencies for specific descriptors are computed and compared to identify distinguishing characteristics between beer categories. The main appearance and flavor words are manually selected from the most frequent terms identified in the tokenized reviews. These words are chosen based on their relevance to the beer's characteristics, ensuring they accurately represent the key attributes of each beer category. This selection process helps focus the analysis on the most distinguishing features in the reviews. 

Goal: Find the main appearance and flavor features of each beer cluster

File: scr/scripts/flavors_NLP

- <b> Beer lexicon extractraction </b>

The method involves processing beer reviews to extract meaningful features for analysis. First, the reviews are tokenized by removing stopwords and punctuation, applying lemmatization, and retaining only nouns and adjectives to focus on descriptive content. Next, a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer is used to create a feature matrix, limiting the vocabulary to the 1,000 most important terms that appear in at least 0.5% of the reviews. This representation highlights terms that are both frequent and distinctive, enabling further clustering and analysis of beer descriptions.

Goal: prepare the reviews for principal component analysis. The focus is here on the beer lexicon. 

File: src/scripts/extract_beer_lexicon.ipynb

#### Principal components analysis (PCA)

Before applying this method, beer lexicon extractraction using NLP is computed on the reviews. 

The method applies TF-IDF vectorization to transform beer review texts into numerical features, capturing the importance of words and phrases (up to bigrams) based on their frequency across the dataset. It then reduces the dimensionality of the data using TruncatedSVD (a form of PCA suited for sparse data) to extract the top 5 principal components (PCs), which summarize the underlying structure of the reviews. Finally, the method performs an ANOVA test to examine significant differences across beer clusters for each principal component and inspects the most influential terms for each component to interpret the key features driving each of the PCA axis. 

By manually analysing the positively and negatively contributing words for each PCA axis, we identified key features that distinguish positively and negatively contributing words for each axis. Based on this analysis, we crafted a series of questions, each reflecting a specific aspect of the principal components, which a user can answer on a scale from -1 to +1. The responses to these questions are then used to assign the user to the most appropriate beer cluster.

Goal: craft questions that assign (recommand) a user to a community (cluster). 

File: scr/scripts/cluster_recommander

#### Emotion detection (using NRCLex librairy)

The method is used to extract emotional content from beer reviews using the NRCLex library, which analyzes the text and generates affect frequencies for various emotions. It applies NRCLex to each review, extracting emotional frequencies for categories such as joy, sadness, and anger, and stores these values in new columns. The emotional data is then merged with the original reviews, enabling further analysis of the emotional tone within the dataset.

Goal: Analyze the emotional tones expressed in reviews, either grouped by cluster or associated with individual words.

File: scr/scripts/Emo2Cluster

#### Image generation (DALL·E)

One of our methods involves using DALL·E to generate visual representations for each beer cluster. For each cluster, we provided a list of the top five beer styles associated with it, and DALL·E created an image that reflects these styles, offering a visual interpretation of the cluster's characteristics.

Goal: Create visual representations of the typical beer for each cluster.

File: src/data/beer_images

### Organisation within team

Dani:     1. Implementation of the NLP methods for extracting main beer charcteristics 
          2. Conceptualization and implementation of the PCA method
          3. Creation of graphs for the analyzing the characteristics within each cluster

Guillen:  1. Implementation of the NLP methods for the beer lexicon extractraction 
          2. Implemention of Emotion detection (using NRCLex librairy)
          3. Extraction questions for the PCA axis interpretation

Arnault:  1. Creation of the datastory and visualization and analysis of the results within the project
          2. Creation of images using Dall.e 
          3. Cleaning of the result.ipynb and of the GitHub for rendering the project

Anas:     1. Conceptualization and implementation of the clustering method (a lot of work)
          2. Help in the creation of the PCA methods and coming up with ideas for the different methods in the project
          3. Creation of the website

Valentin: 1. Creation of the datastory
          2. Creation of the website



