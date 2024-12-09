
# From Hops to Glass
## A Comprehensive Analysis of Beer Characteristics

### Abstract
This study employs a range of analytical techniques to uncover trends, preferences, and perceptions within the beer industry. Using natural language processing (NLP) methods, we analyze reviews from BeerAdvocate to identify clusters of beers and explore the key words and characteristics that distinguish them from one another. For each cluster, we highlight the primary flavor and appearance features that define it. We then compare these characteristics across clusters to identify patterns and differences. Additionally, we examine how these clusters evolve seasonally, over time, and across different countries.

We also examine the emotional responses associated with each beer cluster, focusing on how specific appearance features and flavor profiles elicit distinct emotions.

To further understand these patterns, we apply principal component analysis (PCA) to extract and interpret the main components that define each cluster. The ultimate goal of this research is to develop a model capable of assigning a beer cluster to a user based on their preferences, allowing the user to discover the cluster that best aligns with their tastes through a web-based questionnaire. This research offers valuable insights for brewers, marketers, and consumers by enhancing the understanding of beer diversity and evolving consumer preferences.

### Some research questions we aim to answer
1. What are the main features that define the primary beer categories?
2. How do the flavor and appearance features of beers differ across those categories?
3. What are the seasonal, annual, and geographic patterns for those categories?
4. Given a review, what is the mood of the user who wrote it?
5. Is there a connection between beer flavors/appearance features and the emotional tone of the commentary?
6. How do emotional responses vary across different beer categories and clusters?
7. How can the results from PCA be used to better understand the relationships between different beer clusters and consumer preferences?


### Proposed additional datasets
 There are no direct additional datasets.

### Methods

#### Natural Language Processing (NLP)
Using NLP for analyzing beer descriptions involves extracting meaningful
insights from textual data, such as customer reviews or tasting notes,
to better understand perceptions, preferences, and trends.

- <b> Topic Modeling for Descriptions </b>

Goal : Identify common themes or topics in beer descriptions.

How: By applying topic modeling techniques like LDA,
you can uncover recurring aspects (e.g., "hoppy", "citrusy", "malty")
across reviews to group beers by shared characteristics

- <b> Sentiment Analysis </b>

Purpose: Assess overall sentiments in beer reviews to determine customer satisfaction.

How: Analyze whether the descriptions are predominantly positive, negative,
or neutral, and correlate this sentiment to beer ratings or popularity.

- <b> Emotion Detection </b>

Purpose: Link reviews to specific emotions (e.g., happiness, nostalgia)
and associate them with beer styles or brands.

How: Map keywords or expressions in reviews to emotional dimensions
for richer consumer insights.

- <b> Personality Insights (Big Five Framework) </b>

Purpose: Understand how personality traits influence
beer preferences and descriptions.

How: Use linguistic cues in text to predict personality traits and analyze their
association with certain beers or styles.

- <b> Semantic Similarity Analysis </b>

Purpose: Compare descriptions of the same beer by different users to
identify overlaps and divergences.

How: Use text similarity measures to assess how consistently a beer is
described and what might drive differences (e.g., user experience
or tasting environment).

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

