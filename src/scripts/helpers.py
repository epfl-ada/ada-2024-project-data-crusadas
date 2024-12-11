# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import gzip
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# For NLP processing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import pos_tag
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# For topic modeling
from gensim import corpora, models

# For visualization of topics
from wordcloud import WordCloud

# For statistical analysis
from scipy.stats import (
    chi2_contingency, chi2, ttest_ind, mannwhitneyu,
    entropy, ks_2samp
)

# For embeddings and similarity calculations
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Data loading
from src.data.beerdata_loader import BeerDataLoader

# To show progress
from tqdm import tqdm

#For multiprocessing
from multiprocessing import Pool

#For shapefiles
import geopandas as gpd

import os
import pickle

from gensim.models import LdaModel
from gensim.corpora import Dictionary

tqdm.pandas()
from collections import Counter
import plotly.express as px

import string
import swifter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

flavor_descriptors = [
    'hoppy', 'bitter', 'citrus', 'pine', 'floral', 'malty', 'sweet', 'roasted', 'caramel',
    'chocolate', 'coffee', 'fruit', 'spicy', 'herbal', 'earthy', 'toffee', 'tropical', 'yeast',
    'banana', 'clove', 'smoke', 'oak', 'vanilla', 'nutty', 'wheat', 'grain', 'sour', 'tart', 'milk', 'coriander'
]
appearance_descriptors = [
    'head', 'dark', 'brown', 'light', 'white', 'golden', 'straw', 'orange', 'yellow', 'pale', 'hazy', 'opaque', 'amber'
]

cluster_mapping = {
    0: 'Light and Refreshing Lagers',
    1: 'Dark and Strong Ales',
    2: 'Hop-Forward IPAs',
    3: 'Rich and Roasty Stouts',
    4: 'Belgian and Wheat Styles',
    5: 'Malty and Balanced Ales',
    6: 'Citrusy and Balanced IPAs',
    7: 'Sour and Experimental Beers',
    8: 'Specialty and Niche Styles'
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    # Lowercase
    text = text.lower()
    # Tokenize with preserve_line=True to avoid sent_tokenize
    tokens = word_tokenize(text, preserve_line=True)
    # Remove punctuation and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    return tokens

def filter_beer_reviews(ipa_reviews_df, non_ipa_reviews_df):
    # Set parameters
    review_threshold = 100  # Minimum number of reviews a beer must have
    max_reviews_per_beer = 1000  # Maximum number of reviews to process per beer
    sample_size = 100000  # Total reviews to process per category

    ipa_size = (ipa_reviews_df.shape[0])
    if ipa_size < 100000 and ipa_size > 40000:
        review_threshold = 10
    elif ipa_size < 40000:
        review_threshold = 1

    # For IPA reviews
    ipa_review_counts = ipa_reviews_df['beer_id'].value_counts()
    ipa_beers_selected = ipa_review_counts[ipa_review_counts >= review_threshold].index.tolist()

    ipa_reviews_filtered = ipa_reviews_df[ipa_reviews_df['beer_id'].isin(ipa_beers_selected)]
    ipa_reviews_filtered = ipa_reviews_filtered.groupby('beer_id').head(max_reviews_per_beer).reset_index(drop=True)

    # For Non-IPA reviews
    non_ipa_review_counts = non_ipa_reviews_df['beer_id'].value_counts()
    non_ipa_beers_selected = non_ipa_review_counts[non_ipa_review_counts >= review_threshold].index.tolist()

    non_ipa_reviews_filtered = non_ipa_reviews_df[non_ipa_reviews_df['beer_id'].isin(non_ipa_beers_selected)]
    non_ipa_reviews_filtered = non_ipa_reviews_filtered.groupby('beer_id').head(max_reviews_per_beer).reset_index(drop=True)

    # Sample to reduce data size
    sample_size = min(sample_size, ipa_reviews_filtered.shape[0])
    ipa_reviews_filtered = ipa_reviews_filtered.sample(n=sample_size, random_state=42)
    non_ipa_reviews_filtered = non_ipa_reviews_filtered.sample(n=sample_size, random_state=42)

    print("Total number of selected IPA reviews:", len(ipa_reviews_filtered))
    print("Total number of selected Non-IPA reviews:", len(non_ipa_reviews_filtered))
    return ipa_reviews_filtered, non_ipa_reviews_filtered

def tokenize(ipa_reviews_filtered, non_ipa_reviews_filtered):
    # Apply preprocessing
    #print("Preprocessing IPA reviews...")
    ipa_reviews_filtered['tokens'] = ipa_reviews_filtered['text'].progress_apply(preprocess_text)

    #print("Preprocessing Non-IPA reviews...")
    non_ipa_reviews_filtered['tokens'] = non_ipa_reviews_filtered['text'].progress_apply(preprocess_text)

    # Remove entries with empty tokens
    ipa_reviews_filtered = ipa_reviews_filtered[ipa_reviews_filtered['tokens'].str.len() > 0].reset_index(drop=True)
    non_ipa_reviews_filtered = non_ipa_reviews_filtered[non_ipa_reviews_filtered['tokens'].str.len() > 0].reset_index(drop=True)

    #print(f"Total IPA reviews after preprocessing: {len(ipa_reviews_filtered)}")
    #print(f"Total Non-IPA reviews after preprocessing: {len(non_ipa_reviews_filtered)}")
    return ipa_reviews_filtered, non_ipa_reviews_filtered

def extract_top_words(ipa_reviews_filtered, non_ipa_reviews_filtered):
    
    ipa_tokens_list = ipa_reviews_filtered['tokens'].tolist()
    non_ipa_tokens_list = non_ipa_reviews_filtered['tokens'].tolist()
    return ipa_tokens_list, non_ipa_tokens_list


def compute_criterium(ipa_tokens_list, non_ipa_tokens_list, descriptor):
     
    def count_appearance(tokens_list, flavor_list):
        flavor_counts = dict.fromkeys(flavor_list, 0)
        for tokens in tokens_list:
            for token in tokens:
                if token in flavor_list:
                    flavor_counts[token] += 1
        return flavor_counts

    # Count flavors in IPA and Non-IPA reviews
    ipa_appearance_counts = count_appearance(ipa_tokens_list, descriptor)
    non_ipa_appearance_counts = count_appearance(non_ipa_tokens_list, descriptor)

    # Convert to DataFrame
    ipa_appearance_df = pd.DataFrame(list(ipa_appearance_counts.items()), columns=['Appear', 'Count_IPA'])
    non_ipa_appearance_df = pd.DataFrame(list(non_ipa_appearance_counts.items()), columns=['Appear', 'Count_Non_IPA'])
    comparison_df = ipa_appearance_df.merge(non_ipa_appearance_df, on='Appear')
    comparison_df['Difference'] = comparison_df['Count_IPA'] - comparison_df['Count_Non_IPA']
    return comparison_df

def print_cloud(appearance_comparison_df, flavor_comparison_df, cluster_nb):
    
    # Generate word cloud for appearance
    ipa_word_freq = dict(zip(appearance_comparison_df['Appear'], appearance_comparison_df['Difference']))
    ipa_wordcloud = WordCloud(
        width=800, height=400, background_color='white', 
        color_func=lambda *args, **kwargs: 'grey'  # Solid grey color
    )
    ipa_wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=lambda *args, **kwargs: 'blue').generate_from_frequencies(ipa_word_freq)

    # Generate word cloud for Non-IPA
    non_ipa_word_freq = dict(zip(flavor_comparison_df['Appear'], flavor_comparison_df['Difference']))
    non_ipa_wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=lambda *args, **kwargs: 'orange').generate_from_frequencies(non_ipa_word_freq)

    # Plot word clouds
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(ipa_wordcloud, interpolation='bilinear')
    plt.title(f'Appearance Descriptors {cluster_nb}', size=20)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(non_ipa_wordcloud, interpolation='bilinear')
    plt.title(f'Flavor Descriptors {cluster_nb}', size=20)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def create_sunburst(cluster_nbr, assigned_cluster, cluster_name, color):
    # Assuming 'style' column exists
    beer = assigned_cluster[assigned_cluster['cluster'] == cluster_nbr]
    style_counts = beer['style'].value_counts()
    #Take only top 5 popular styles
    popular_styles = style_counts[0:5]
    popular_styles = popular_styles.reset_index()
    popular_styles.columns = ['Style', 'Count']

    # Add "Others" to the DataFrame
    others_count = style_counts.iloc[5:].sum()  # Sum counts of styles outside the top 5
    #if others_count > 0:  # Add only if there are remaining styles
    #    others_row = pd.DataFrame({'Style': ['Others'], 'Count': [others_count]})
    #    popular_styles = pd.concat([popular_styles, others_row], ignore_index=True)
    
    # Optionally, if there are sub-categories, include them
    # For simplicity, we'll assume 'style' is the only category
    
    fig = px.sunburst(popular_styles, path=['Style'], values='Count',
                      title=f'Style Distribution in {cluster_name}',
                      color='Count', 
                      color_continuous_scale='Reds' if color=='darkred' else 'Blues')
    
    fig.add_annotation(
        x=0.9,  # Center of the sunburst
        y=-0.2,  # Place below the chart
        text=f'Other styles: {others_count}',
        showarrow=False,
        font=dict(size=14, color='darkred')
    )
    
    fig.show()


def clean_location(location):
    #The goal of this method is to change the name of some countries
    #to be able to use them on the shapefile
    if isinstance(location, str):  # Check if location is a string
        country = location.split(',')[0].strip()
        # Replace "United States" with "United States of America"
        if country == "United States":
            return "United States of America"
        if country == "England":
            return "United Kingdom"
        return country
    return None

def show_distribution(cluster_name, name, title, xlabel, min_x_axis, max_x_axis):
    # Set the style for seaborn
    sns.set(style="whitegrid")

    # Define a color palette with distinct colors for 9 clusters
    palette = sns.color_palette("bright", n_colors=9)

    # Create the plot
    plt.figure(figsize=(12, 6))

    legend_labels = []

    for i in range(9):  # Loop through clusters 0 to 8
        # Filter data for the current cluster
        cluster = cluster_name[cluster_name['cluster'] == i]
        
        # Plot ABV distributions with unique colors
        sns.kdeplot(
            cluster[name],
            fill=False,
            label=f'Cluster {i}',
            color=palette[i]  # Assign a unique color
        )

        legend_labels.append(cluster_mapping[i]) 


    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.xlim(min_x_axis, max_x_axis)

    plt.legend(labels=legend_labels)

    # Display the plot
    plt.show()

def show_evolution(cluster, name):
    pivot_df = cluster.pivot(index=name, columns='cluster', values='rating').fillna(0)
    pivot_df_normalized = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    # Use a predefined bright color palette from seaborn
    bright_palette = sns.color_palette("bright", n_colors=9)

    # Plotting the stacked bar chart with normalized data (percentage)
    ax = pivot_df_normalized.plot(kind='bar', stacked=True, figsize=(14, 7), color=bright_palette)

    # Update the legend with cluster names instead of cluster numbers
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [cluster_mapping[int(float(label))] for label in labels]  # Convert labels to integers first
    ax.legend(handles, new_labels, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adding labels and title
    plt.title(f'Stacked Bar Chart of Ratings by Cluster and {name} (Clusters 0-8) - Percentage')
    plt.xlabel('name')
    plt.ylabel('Percentage of Total Rating')
    plt.xticks(rotation=90)
    plt.tight_layout()  # Adjust layout for better fit

    # Show the plot
    plt.show()

def filter_location(min_ratings, final_merged_df):
    location_ratings = final_merged_df.groupby('cleaned_location').count()

    # Filter locations with more than 1000 ratings
    locations_over_min = location_ratings[location_ratings['rating'] > min_ratings].index

    # Filter the original DataFrame to include only these locations
    return final_merged_df[final_merged_df['cleaned_location'].isin(locations_over_min)]

def show_distribution_histogram(cluster_name, name, title, xlabel, min_x_axis, max_x_axis):
    sns.set(style="whitegrid")

    #Define the colours
    specific_colors = {
        0: 'yellow',
        1: 'grey',
        2: 'pink', 
        3: 'brown',  
        4: 'purple',
        5: 'red', 
        6: 'green',
        7: 'orange',  
        8: 'blue',  
    }

    # Calculate the weights for normalization
    # Weight is the reciprocal of the number of data points in each cluster
    cluster_counts = cluster_name.groupby('cluster').size()
    cluster_name['weights'] = cluster_name['cluster'].map(lambda x: 1 / cluster_counts[x])

    # Create the plot
    plt.figure(figsize=(12, 6))

    sns.histplot(
        data=cluster_name,
        x=name,
        hue="cluster",
        multiple="dodge",
        bins=16,
        palette=specific_colors,
        weights="weights",  # Apply weights to normalize the histogram
        stat="density",  # Normalize so area under curve equals 1
        discrete=False,  # Ensure continuous binning (adjust if you need discrete bins)
    )

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.xlim(min_x_axis, max_x_axis)

    # Update the legend with cluster names (use cluster_mapping for names)
    legend_labels = [cluster_mapping[i] for i in range(9)]
    plt.legend(labels=legend_labels, title="Cluster")

    # Display the plot
    plt.show()

def geographic_distribution_cluster(cluster_number, cluster_name, cluster_df):
    lager_cluster = cluster_df[cluster_df['cluster'] == cluster_number]
    shapefile_path = r'.\src\data\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp'
    world = gpd.read_file(shapefile_path)

    # Merge your ratings data with the world GeoDataFrame
    merged = world.merge(lager_cluster, how='left', left_on='SOVEREIGNT', right_on='cleaned_location')

    # Fill NaN values in 'nbr_ratings' with 0 for visualization
    merged['rating_percentage'] = merged['rating_percentage'].fillna(0)

    # Create a figure with adjusted subplot sizes
    fig = plt.figure(figsize=(18, 10))

    # World map (larger)
    ax_world = fig.add_axes([0.1, 0.35, 0.75, 0.6])   # Left, Bottom, Width, Height
    merged.boundary.plot(ax=ax_world, color='black')  # World boundaries
    merged.plot(
        column='rating_percentage', cmap='YlGnBu', linewidth=0.8, 
        ax=ax_world, edgecolor='0.8', legend=True
    )
    ax_world.set_title(f'World Map Popularity: {cluster_name}', fontsize=16)
    ax_world.set_axis_off()
    ax_world.set_ylim([-40, 100]) 
    ax_world.set_xlim([-140, 170])

    # Europe zoom (smaller, inset)
    ax_europe = fig.add_axes([0.27, 0.05, 0.25, 0.4])  # Adjust to fit inside world map
    merged.boundary.plot(ax=ax_europe, color='black')  # Europe boundaries
    merged.plot(
        column='rating_percentage', cmap='YlGnBu', linewidth=0.8, 
        ax=ax_europe, edgecolor='0.8'
    )
    ax_europe.set_title(f'Europe Zoom', fontsize=12)
    ax_europe.set_axis_off()

    # Set zoom for Europe
    ax_europe.set_xlim([-12, 45])  # Longitude limits for Europe
    ax_europe.set_ylim([36, 70])   # Latitude limits for Europe

    plt.show()

# to see how I obtained this beer lexicon, check out Extract_feats_from_reviews_Guillen.ipynb --- EXPLAIN
beer_lexicon = {
    'beer', 'ale', 'lager', 'stout', 'porter', 'pilsner', 'ipa', 'brewer', 
 'brewery', 'brewing', 'draft', 'cask', 'bottle', 'glass', 'pour', 'tap','head', 'foam', 
    'frothy', 'bubbly', 'carbonated', 'flat', 'clarity','hazy', 'opaque', 'cloudy', 'clear', 'golden', 'amber', 'brown', 'dark', 
 'black', 'mahogany', 'ruby', 'white', 'pale', 'yellow', 'beige', 'red', 
 'floral', 'herbal', 'earthy', 'spicy', 'citrusy', 'fruity', 'sweet', 
 'sour', 'bitter', 'bitterness', 'smooth', 'rich', 'dry', 'crisp', 'creamy', 
 'sticky', 'tart', 'tangy', 'sweetness', 'malty', 'hoppy', 'aromatic', 
 'aroma', 'aftertaste', 'subtle', 'intense',
'barley', 'malt', 'malty', 'grain', 'yeast', 'hops', 'hop', 'water', 
 'spices', 'vanilla', 'chocolate', 'caramel', 'toffee', 'coffee', 'cocoa', 
 'citrus', 'orange', 'lemon', 'grapefruit', 'pine', 'oak', 'nutty', 'raisin', 
 'plum', 'cherry', 'maple', 'banana', 'berry', 'ginger','session', 'sessionable', 'pint', 'ounce', 'ounce', 'snifter', 'chalice', 
 'pub', 'bar', 'draught', 'taste', 'drink', 'drinker', 'sip', 'gulp', 
 'quaffable','balanced', 'strong', 'bold', 'weak', 'delicate', 'refreshing', 'complex', 
 'light', 'heavy', 'moderate', 'intense', 'robust', 'unique', 'classic', 
 'distinctive', 'remarkable', 'awesome', 'excellent', 'superb', 'good', 
 'decent', 'ok', 'average', 'bad', 'boring', 'cheap', 'quality','balanced', 'strong', 'bold', 'weak', 'delicate', 'refreshing', 'complex', 
 'light', 'heavy', 'moderate', 'intense', 'robust', 'unique', 'classic', 
 'distinctive', 'remarkable', 'awesome', 'excellent', 'superb', 'good', 
 'decent', 'ok', 'average', 'bad', 'boring', 'cheap', 'aged', 'vintage', 'cellar', 'oak', 'barrel', 'reserve', 'craft', 'microbrewery', 'homebrew', 'regional', 'seasonal', 'festive', 
 'holiday', 'celebration', 'festival', 'special'}


# for a review just keep the words that are in the beer lexicon
def clean_text_lexicon(text):
    text = text.lower()
    cleaned_text = [word for word in text.split() if word in beer_lexicon]
    return " ".join(cleaned_text)

def beer_processing_pca(partition, ba_beers_df, ba_reviews_df):

    beer_cluster_map = {int(k): v for k, v in partition.items()}
    # Add cluster information to beers
    ba_beers_df["cluster"] = ba_beers_df["beer_id"].map(beer_cluster_map)

    # Filter reviews to only those beers in the cluster map
    ba_reviews_df = ba_reviews_df[ba_reviews_df["beer_id"].isin(beer_cluster_map.keys())]

    # Drop rows without text and ensure text is a string
    ba_reviews_df = ba_reviews_df.dropna(subset=["text"])
    ba_reviews_df["text"] = ba_reviews_df["text"].astype(str)

    beer_texts = (
    ba_reviews_df.groupby("beer_id")["text"]
    .agg(' '.join)
    .reset_index()
    )

    # Merge with cluster information
    beer_texts = beer_texts.merge(ba_beers_df[["beer_id", "cluster"]], on="beer_id", how="left")
    beer_texts = beer_texts.dropna(subset=["cluster"])

    # Convert cluster to category for efficiency
    beer_texts["cluster"] = beer_texts["cluster"].astype("category")

    # Keep only the first 9 clusters (0 through 8)
    clusters_to_keep = list(range(9))
    beer_texts = beer_texts[beer_texts["cluster"].isin(clusters_to_keep)]

    # IMPORTANT: Remove unused categories after filtering
    beer_texts["cluster"] = beer_texts["cluster"].cat.remove_unused_categories()

    beer_texts["clean_text"] = beer_texts["text"].swifter.apply(clean_text_lexicon)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        #stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5
    )

    X = vectorizer.fit_transform(beer_texts["clean_text"])
    # PCA via TruncatedSVD (for sparse data)
    n_components = 5
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_pca = svd.fit_transform(X)

    # Attach PCA components back to DataFrame
    for i in range(n_components):
        beer_texts[f"PC{i+1}"] = X_pca[:, i]
        
    terms = np.array(vectorizer.get_feature_names_out())
    
    return svd, terms, beer_texts

def create_superposed_spider_chart(normalized_means, num_vars, labels):
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Colors for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, len(normalized_means)))

    # Plot each cluster
    for cluster_id, means in enumerate(normalized_means):
        data = means.tolist() + [means[0]]  # Close the circle
        ax.plot(angles, data, label=cluster_mapping[cluster_id], linewidth=2)
        #ax.fill(angles, data, alpha=0.1)

    # Aesthetic adjustments
    ax.set_yticks([-2, -1, -0.5, 0, 0.5, 1, 2])
    ax.set_yticklabels(['-2', '-1', '-0.5', '0', '0.5', '1', '2'], color="gray", size=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)

    ax.set_title("Superposed Spider Map (Normalized)", size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))
    plt.show()

def standardize(cluster_means):
    # Normalize the data using Min-Max scaling (to range between 0 and 1)
    scaler = StandardScaler()
    normalized_means = scaler.fit_transform(np.array(list(cluster_means.values())))

    # Scale the normalized values to the range [-1, 1]
    normalized_means = 2 * (normalized_means - np.min(normalized_means, axis=0)) / (np.max(normalized_means, axis=0) - np.min(normalized_means, axis=0)) - 1

    return normalized_means

def display_path_2D(cluster_means, user_answers):
    user_answers_path = np.diag(user_answers)
    for i in range(1, user_answers_path.shape[0]):
        for j in range(i):
            user_answers_path[i, j] = user_answers_path[j, j]

    #Normalize both
    normalized_cluster_means = normalize(cluster_means, norm='l2', axis=1)
    normalized_user_answers = normalize(user_answers.reshape(1,-1), norm='l2', axis=1)
    normalized_user_answers_path = normalize(user_answers_path, norm='l2', axis=1)

    # Apply PCA on the normalized cluster means
    pca = PCA(n_components=2)
    pca_cluster_means = pca.fit_transform(normalized_cluster_means)
    pca_user_answers_path = pca.transform(normalized_user_answers_path)

    # Transform user answers using the same PCA
    pca_user_positions = pca.transform(normalized_user_answers)

    # Plotting
    plt.figure(figsize=(10, 8))

    # Plot PCA result for clusters with explicit labeling
    for idx, (x, y) in enumerate(pca_cluster_means):
        plt.scatter(x, y, label=cluster_mapping[idx], s=100, cmap='tab10')


    # Plot user journey
    plt.plot(
        pca_user_answers_path[:, 0],
        pca_user_answers_path[:, 1],
        marker='o',
        color='blue',
        label='User Journey'
    )

    # Annotate user journey points
    for i, (x, y) in enumerate(pca_user_answers_path):
        plt.text(x, y, f'Q{i+1}', fontsize=10, color='blue')


    # Plot the user answers on the same plot
    plt.scatter(
        pca_user_positions[0, 0],
        pca_user_positions[0, 1],
        color='red',
        s=150,
        label="User Position",
        marker='*',
        zorder=5
    )

    # Labels and legend
    plt.title("PCA Cluster Visualization with User Journey")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

def show_spider_emotions(merged_df):

    emotions_list = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy', 'anticipation']

    # filter the dataframe to include only clusters from 0 to 8 and group by 'cluster_id' and calculate the mean for each emotion column
    emotion_means = merged_df[merged_df['cluster_id'].isin(range(9))].groupby('cluster_id')[emotions_list].mean()
    # Cluster mapping
    cluster_mapping = {
        0: 'Light and Refreshing Lagers',
        1: 'Dark and Strong Ales',
        2: 'Hop-Forward IPAs',
        3: 'Rich and Roasty Stouts',
        4: 'Belgian and Wheat Styles',
        5: 'Malty and Balanced Ales',
        6: 'Citrusy and Balanced IPAs',
        7: 'Sour and Experimental Beers',
        8: 'Specialty and Niche Styles'
    }

    # Radar chart setup
    categories = ["joy", "trust", "anticipation", "disgust", "surprise", "anger", "sadness", "fear"]
    num_vars = len(categories)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Repeat the first angle to close the circle

    # Initialize radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each cluster
    for i, row in emotion_means.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]  # Repeat the first value to close the circle
        ax.plot(angles, values, label=cluster_mapping[i])
        #ax.fill(angles, values, alpha=0.1)  # Optional: Add transparency to fill

    # Add labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticks([0.05, 0.10, 0.15])
    ax.set_yticklabels(["0.05", "0.10", "0.15"], color="grey", fontsize=8)
    ax.set_ylim(0, 0.16)
    plt.title("Superposed Radar Chart of Emotions by Cluster", size=16, y=1.1)

    # Add legend with cluster mapping
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    # Show plot
    plt.tight_layout()
    plt.show()

def create_emotions_dataframe(merged_df, descriptors):
    # Create a list to store emotion results for each descriptor
    results = []
    
    # Iterate over the list of descriptors
    for descriptor in descriptors:
        # Vectorize the descriptor search to find rows that match the descriptor
        vectorized_contains = np.vectorize(lambda x: descriptor.lower() in str(x).lower())
        descriptor_rows = merged_df[vectorized_contains(merged_df['text'])]

        # List of emotions to compute the mean for
        emotions_list = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy', 'anticipation']

        # Compute the mean of emotions for the filtered rows
        emotions_descriptor = descriptor_rows[emotions_list].mean()

        # Append the descriptor and its corresponding emotion means to the results list
        results.append([descriptor] + emotions_descriptor.tolist())

    # Create a DataFrame from the results
    emotions_df = pd.DataFrame(results, columns=['Descriptor'] + emotions_list)
    
    return emotions_df

def show_spider_emotions_descriptors(emotion_means, cluster_number):
    # Radar chart setup
    categories = ["joy", "trust", "anticipation", "disgust", "surprise", "anger", "sadness", "fear"]
    num_vars = len(categories)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Repeat the first angle to close the circle

    # Initialize radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each cluster
    for i, row in emotion_means.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]  # Repeat the first value to close the circle
        ax.plot(angles, values, label=row['Descriptor'])
        #ax.fill(angles, values, alpha=0.1)  # Optional: Add transparency to fill

    # Add labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticks([0.05, 0.10, 0.15])
    ax.set_yticklabels(["0.05", "0.10", "0.15"], color="grey", fontsize=8)
    ax.set_ylim(0, 0.16)
    plt.title(f"Emotions from top words of cluster {cluster_number}", size=16, y=1.1)

    # Add legend with cluster mapping
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    # Show plot
    plt.tight_layout()
    plt.show()