# beerdata_loader.py

import pandas as pd
import csv
import os


class BeerDataLoader:
    def __init__(self, data_dir, force_process=False):
        """
        Initializes the data loader with the specified data directory.

        Parameters:
        - data_dir: Directory where the data files are located.
        - force_process: If True, forces reprocessing even if processed files exist.
        """
        self.data_dir = data_dir
        self.force_process = force_process

        # Define file paths
        self.reviews_txt = os.path.join(self.data_dir, "reviews.txt")
        self.ratings_txt = os.path.join(self.data_dir, "ratings.txt")
        self.beers_csv = os.path.join(self.data_dir, "beers.csv")
        self.breweries_csv = os.path.join(self.data_dir, "breweries.csv")
        self.users_csv = os.path.join(self.data_dir, "users.csv")
        self.reviews_processed_csv = os.path.join(
            self.data_dir, "reviews_processed.csv"
        )
        self.ratings_processed_csv = os.path.join(
            self.data_dir, "ratings_processed.csv"
        )

        # DataFrames
        self.reviews_df = None
        self.ratings_df = None
        self.beers_df = None
        self.breweries_df = None
        self.users_df = None

    def process_reviews(self):
        """
        Processes the reviews.txt file and saves it as reviews_processed.csv.
        """
        if not self.force_process and os.path.exists(self.reviews_processed_csv):
            print(
                f"Processed file '{self.reviews_processed_csv}' already exists. Skipping processing."
            )
            return

        columns = [
            "beer_name",
            "beer_id",
            "brewery_name",
            "brewery_id",
            "style",
            "abv",
            "date",
            "user_name",
            "user_id",
            "appearance",
            "aroma",
            "palate",
            "taste",
            "overall",
            "rating",
            "text",
        ]

        with open(self.reviews_txt, "r", encoding="utf-8") as infile, open(
            self.reviews_processed_csv, "w", newline="", encoding="utf-8"
        ) as outfile:

            writer = csv.DictWriter(outfile, fieldnames=columns)
            writer.writeheader()
            review = {}

            for line in infile:
                line = line.strip()
                if line:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        if key in columns:
                            review[key] = value
                    else:
                        if line.endswith(":"):
                            key = line[:-1]
                            if key in columns:
                                review[key] = ""
                else:
                    if review:
                        writer.writerow({k: review.get(k, "") for k in columns})
                        review.clear()
            if review:
                writer.writerow({k: review.get(k, "") for k in columns})

        print(
            f"Processing completed. Processed data saved to '{self.reviews_processed_csv}'."
        )

    def process_ratings(self):
        """
        Processes the ratings.txt file and saves it as ratings_processed.csv.
        """
        if not self.force_process and os.path.exists(self.ratings_processed_csv):
            print(
                f"Processed file '{self.ratings_processed_csv}' already exists. Skipping processing."
            )
            return

        columns = [
            "beer_name",
            "beer_id",
            "brewery_name",
            "brewery_id",
            "style",
            "abv",
            "date",
            "user_name",
            "user_id",
            "appearance",
            "aroma",
            "palate",
            "taste",
            "overall",
            "rating",
            "text",
            "review",
        ]

        with open(self.ratings_txt, "r", encoding="utf-8") as infile, open(
            self.ratings_processed_csv, "w", newline="", encoding="utf-8"
        ) as outfile:

            writer = csv.DictWriter(outfile, fieldnames=columns)
            writer.writeheader()
            review = {}

            for line in infile:
                line = line.strip()
                if line:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        if key in columns:
                            review[key] = value
                    else:
                        if line.endswith(":"):
                            key = line[:-1]
                            if key in columns:
                                review[key] = ""
                else:
                    if review:
                        if "text" not in review:
                            review["text"] = ""
                        if "review" not in review:
                            review["review"] = "False"
                        writer.writerow({k: review.get(k, "") for k in columns})
                        review.clear()
            if review:
                if "text" not in review:
                    review["text"] = ""
                if "review" not in review:
                    review["review"] = "False"
                writer.writerow({k: review.get(k, "") for k in columns})

        print(
            f"Processing completed. Processed data saved to '{self.ratings_processed_csv}'."
        )

    def load_reviews(self):
        """
        Loads the processed reviews data into a DataFrame.
        """
        if self.reviews_df is not None:
            return self.reviews_df

        # Process the reviews if necessary
        self.process_reviews()

        self.reviews_df = pd.read_csv(self.reviews_processed_csv)
        return self.reviews_df

    def load_ratings(self):
        """
        Loads the processed ratings data into a DataFrame.
        """
        if self.ratings_df is not None:
            return self.ratings_df

        # Process the ratings if necessary
        self.process_ratings()

        dtype_dict = {
            "beer_name": "category",
            "beer_id": "int32",
            "brewery_name": "category",
            "brewery_id": "int32",
            "style": "category",
            "abv": "float32",
            "date": "int64",
            "user_name": "category",
            "user_id": "category",
            "appearance": "float32",
            "aroma": "float32",
            "palate": "float32",
            "taste": "float32",
            "overall": "float32",
            "rating": "float32",
            "text": "string",
            "review": "bool",
        }

        self.ratings_df = pd.read_csv(self.ratings_processed_csv, dtype=dtype_dict)
        return self.ratings_df

    def load_beers(self):
        """
        Loads the beers data into a DataFrame.
        """
        if self.beers_df is not None:
            return self.beers_df

        self.beers_df = pd.read_csv(self.beers_csv)
        return self.beers_df

    def load_breweries(self):
        """
        Loads the breweries data into a DataFrame.
        """
        if self.breweries_df is not None:
            return self.breweries_df

        self.breweries_df = pd.read_csv(self.breweries_csv)
        return self.breweries_df

    def load_users(self):
        """
        Loads the users data into a DataFrame.
        """
        if self.users_df is not None:
            return self.users_df

        self.users_df = pd.read_csv(self.users_csv)
        return self.users_df

    def load_all_data(self):
        """
        Loads all datasets and returns them as a tuple of DataFrames.

        Returns:
        - reviews_df, ratings_df, beers_df, breweries_df, users_df
        """
        reviews_df = self.load_reviews()
        ratings_df = self.load_ratings()
        beers_df = self.load_beers()
        breweries_df = self.load_breweries()
        users_df = self.load_users()

        return reviews_df, ratings_df, beers_df, breweries_df, users_df
