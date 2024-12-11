
# Find your community
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

### Methods

#### Clustering

Generating a similarity graph and clustering beers based on review characteristics using natural language processing (NLP) methods. The process involves tokenizing reviews, filtering discriminative words, computing cosine similarity, constructing a similarity graph, and applying the Louvain algorithm for clustering, with computationally intensive tasks optimized for parallel processing on an HPC cluster.

Goal: find main clusters of beers in the BeerAdvocate dataset

#### Natural Language Processing (NLP)

- <b> Extract main beer charcteristics </b>

The method involves preprocessing, filtering, and analyzing beer reviews. First, reviews are preprocessed by converting text to lowercase, tokenizing, removing punctuation and stopwords, and lemmatizing words. Next, beers with a minimum number of reviews are selected, and reviews are limited to a maximum per beer for computational efficiency. Finally, word frequencies for specific descriptors are computed and compared to identify distinguishing characteristics between beer categories. The main appearance and flavor words are manually selected from the most frequent terms identified in the tokenized reviews. These words are chosen based on their relevance to the beer's characteristics, ensuring they accurately represent the key attributes of each beer category. This selection process helps focus the analysis on the most distinguishing features in the reviews. 

Goal: Find the main appearance and flavor features of each beer cluster

- <b> Beer lexicon extractraction </b>

The method involves processing beer reviews to extract meaningful features for analysis. First, the reviews are tokenized by removing stopwords and punctuation, applying lemmatization, and retaining only nouns and adjectives to focus on descriptive content. Next, a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer is used to create a feature matrix, limiting the vocabulary to the 1,000 most important terms that appear in at least 0.5% of the reviews. This representation highlights terms that are both frequent and distinctive, enabling further clustering and analysis of beer descriptions.

Goal: prepare the reviews for principal component analysis. The focus is here on the beer lexicon. 

#### Principal components analysis (PCA)

Before applying this method, beer lexicon extractraction using NLP is computed on the reviews. 

The method applies TF-IDF vectorization to transform beer review texts into numerical features, capturing the importance of words and phrases (up to bigrams) based on their frequency across the dataset. It then reduces the dimensionality of the data using TruncatedSVD (a form of PCA suited for sparse data) to extract the top 5 principal components (PCs), which summarize the underlying structure of the reviews. Finally, the method performs an ANOVA test to examine significant differences across beer clusters for each principal component and inspects the most influential terms for each component to interpret the key features driving each of the PCA axis. 

By manually analysing the positively and negatively contributing words for each PCA axis, we identified key features that distinguish positively and negatively contributing words for each axis. Based on this analysis, we crafted a series of questions, each reflecting a specific aspect of the principal components, which a user can answer on a scale from -1 to +1. The responses to these questions are then used to assign the user to the most appropriate beer cluster.

Goal: craft questions that assign a user to a community (cluster)

#### Emotion detection (using NRCLex librairy)

The method is used to extract emotional content from beer reviews using the NRCLex library, which analyzes the text and generates affect frequencies for various emotions. It applies NRCLex to each review, extracting emotional frequencies for categories such as joy, sadness, and anger, and stores these values in new columns. The emotional data is then merged with the original reviews, enabling further analysis of the emotional tone within the dataset.

Goal: Analyze the emotional tones expressed in reviews, either grouped by cluster or associated with individual words.

#### Statistical analysis
Using statistical analysis for beer descriptions allows you
to quantitatively explore patterns,
relationships, and trends in the data. 

- <b> Descriptive Statistics </b>

Purpose: Summarize key attributes of beer reviews,
such as average rating, word count, or frequency of certain descriptors
(e.g., "hoppy," "bitter").

How: Compute measures like means, medians, and variances to understand
general trends in how beers are described and rated.

- <b> Correlation Analysis </b>

Purpose: Investigate relationships between variables,
such as the frequency of positive comments and a beer's growth
in popularity.

How: Measure correlations between descriptors (e.g., "fruity")
and numerical outcomes (e.g., rating, sales, or growth percentage).

- <b> Trend Analysis </b>

Purpose: Identify changes over time in beer descriptions and ratings.

How: Use time series analysis to detect trends in descriptor usage
(e.g., increasing mentions of "juicy" in IPAs) or variations in average
ratings by year.

#### Clustering and Classification

- <b> Beer similarity Analysis </b>

Purpose: Group beers based on shared characteristics or user preferences.

How: Identify common words or features in beer descriptions and use a metric to measure similarity between beers.

- <b> Cluster Analysis </b>

Purpose: Identify patterns or groups in beer descriptions or ratings.

How: Construct a graph from beer similarities and apply clustering algorithms like Louvain to detect communities of similar beers.

- <b> Classification of Beer Styles </b>

Purpose: Better description of the automatically contructed clusters.

How: Analyse the clusters to find the most common words and beers in each cluster.

#### Recommander system
- <b> Content-Based Recommendation </b>

Purpose: Suggest beers similar to those a user has enjoyed,
based on their descriptions or attributes.

How: Match beer features (e.g., "hoppy," "citrusy") to user preferences
derived from past reviews or ratings.

Example: If a user likes "fruity" and "light-bodied" beers,
recommend others with similar descriptors.

- <b> Collaborative Filtering </b>

Purpose: Recommend beers based on the preferences of similar users.

How: Analyze patterns in user ratings or reviews to identify clusters
of users with shared tastes, then suggest beers that like-minded users
have enjoyed.

Example: If a user enjoys a beer that another user with similar preferences
liked, recommend beers from the latter’s list.

- <b> Matrix Factorization </b>

Purpose: Identify latent factors (hidden patterns) in user-beer interactions.

How: Use techniques like Singular Value Decomposition (SVD) to break down
large rating matrices into lower dimensions, uncovering relationships between
users and beers.

Example: Factorize user-beer ratings to predict a user’s likelihood
of enjoying a new beer based on similar latent factors.

### Timeline and organisation
- Week 1: 
  - There is 2 differents tables that contains some of the same datas, those
  needs to be carefully merged so we don't lose and/or duplicate informations
  - Implement NLP
  - Analyse the rise of IPA
- Week 2:
  - Implement recommander system
- Week 3: 
  - Merging results
  - filtering informations
  - conclusion on the results
- Week 4:
  - Clean git
  - work on deliverable
  - work on website

