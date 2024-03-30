import collections
import re
import string

import copy
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
from nltk import stem
from wordcloud import wordcloud

nltk.data.path.append("/usr/local/share/nltk_data")
from nltk.corpus import stopwords
import datetime
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class AmazonBookReviewProject:
    books_data = pd.DataFrame()
    books_rating = pd.DataFrame()
    books_final = pd.DataFrame()
    stopWords = []

    def run_project_work(self):
        print('load data')
        self.loadData()
        print('Merging data')
        self.books_df = self.mergeData(self.books_data, self.books_rating)
        print('Cleaning data')
        self.books_df = self.cleanData(self.books_df)
        print('Analyse rating')
        # self.analyseRating()    # Generate histogram
        print('Get stop words')
        self.stopWords = self.getStopWords()
        # self.analyzeSentiments()
        print('Predictions')
        self.runTextPredictions()

    def loadData(self):

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # Specify the path to your CSV file
        csv_file_path = r'/Users/hema/PycharmProjects/TSOM/Data/Books_rating.csv'

        # Use pandas to read the CSV file into a DataFrame
        # You may need to adjust parameters like chunksize depending on your available memory
        chunk_size = 100000  # Experiment with different chunk sizes based on your system's memory
        chunks = []

        reader = pd.read_csv(csv_file_path, chunksize=chunk_size)

        # Process each chunk as needed
        for i, chunk in enumerate(reader):
            # Perform your operations on the chunk here
            # Example: Print the first few rows of each chunk
            print(f"Processing Chunk {i + 1}")
            # print(chunk.head())
            chunks.append(chunk)

        self.books_rating = pd.concat(chunks, ignore_index=True)

        """ Load data into books_data. """

        chunks = []

        # Specify the path to your CSV file
        csv_file_path = '/Users/hema/PycharmProjects/TSOM/Data/books_data.csv'

        reader = pd.read_csv(csv_file_path, chunksize=chunk_size)

        # Process each chunk as needed
        for i, chunk in enumerate(reader):
            # Perform your operations on the chunk here
            # Example: Print the first few rows of each chunk
            print(f"Processing Chunk {i + 1}")
            # print(chunk.head())
            chunks.append(chunk)

        self.books_data = pd.concat(chunks, ignore_index=True)

        # self.books_data = pd.read_csv(r"C:\Users\Lenovo\Downloads\Data\books_data.csv")

    def mergeData(self, books_data, books_rating):
        # Merging both dataframes into one dataframe
        self.books_df = self.books_rating.merge(self.books_data, how='outer', on='Title')
        self.books_df = self.books_df[
            ['Id', 'Title', 'profileName', 'review/score', 'review/summary',
             'review/text', 'description', 'authors', 'publisher', 'publishedDate', 'categories', 'ratingsCount']]
        return self.books_df

    def cleanData(self, books_df):
        # print(books_df.columns.tolist())
        # ['Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 'review/score', 'review/time', 'review/summary',
        # 'review/text', 'description', 'authors', 'image', 'previewLink', 'publisher', 'publishedDate', 'infoLink', 'categories', 'ratingsCount']

        # Rename columns as per naming conventions
        books_df.columns = ['id', 'title', 'profileName', 'review_score', 'review_summary',
                            'review_text', 'description', 'authors', 'publisher', 'publishedDate', 'categories',
                            'ratingsCount']

        """ TREAT MISSING VALUES"""
        print("\nNUMBER OF ROWS BEFORE HANDLING MISSING VALUES:\n", len(books_df))

        # Print missing values summary for column in 'books_df'
        print("\nBOOKS_DF MISSING VALUES SUMMARY : BEFORE HANDLING MISSING VALUES\n", books_df.isnull().sum())

        # Treat missing values in the 'profileName', 'review_text', 'description', 'publisher', 'categories', 'ratingsCount' variables. In this case, dropna has been chosen as the strategy.
        books_df.dropna(
            subset=['profileName', 'review_summary', 'review_text', 'description', 'authors', 'publisher', 'categories',
                    'ratingsCount'], inplace=True)
        print("\nNUMBER OF ROWS AFTER HANDLING MISSING VALUES\n", len(books_df))

        # print(books_df.head(5))
        print("\nBOOKS_DF MISSING VALUES SUMMARY AFTER HANDLING MISSING VALUES \n", books_df.isnull().sum())

        # Remove all occurrences of [ " ' enclosing ' " ] in authors and categories
        books_df.loc[:, 'authors'] = books_df['authors'].astype(str).str.replace(r"[\[\]\'\"]+", "", regex=True)
        books_df.loc[:, 'categories'] = books_df['categories'].astype(str).str.replace(r"[\[\]\'\"]+", "", regex=True)

        def standardize_date(date_str):
            # Check if the date string is NaN
            if pd.isna(date_str):
                return np.nan
            # Check if the date string is just a year
            elif len(date_str) == 4:
                return f"01/01/{date_str}"
            # Check if the date string is in the format "YYYY-MM"
            elif len(date_str) == 7 and date_str.count('-') == 1:
                return f"{date_str.replace('-', '/')}/01"
            # If the date string is already in the format "MM/DD/YYYY", return it as is
            elif len(date_str) == 10 and date_str.count('/') == 2:
                return date_str
            # If none of the above conditions are met, return the original string
            else:
                return date_str

        # Apply the standardize_date function to the 'publishedDate' column
        books_df.loc[:, 'publishedDate'] = books_df['publishedDate'].apply(standardize_date)

        # Now convert the standardized date strings into datetime objects
        books_df['publishedDate'] = pd.to_datetime(books_df['publishedDate'], format='%m/%d/%Y', errors='coerce')

        # Format the datetime objects to a string in the format MM/DD/YYYY
        books_df.loc[:, 'publishedDate'] = books_df['publishedDate'].dt.strftime('%m/%d/%Y')

        """ HANDLE DUPLICATE ROWS"""
        # Check for duplicates in entire dataframe
        print("\nBOOKS_DF DUPLICATES PRESENT OR NOT\n", books_df.duplicated().any())
        print("\nLength of books_df before removing duplicates: \n", len(books_df))

        # To remove duplicates considering all columns
        books_df.drop_duplicates(keep='first', inplace=True)
        print("\nLength of books_df after removing duplicates: \n", len(books_df))

        return books_df

    def analyseRating(self):
        # print("\nUNIQUE REVIEW SCORES\n", self.books_df['review/score'].unique())
        # print(self.books_df.head(5))
        """Displaying the frequency of rating using a histogram"""
        fig = px.histogram(self.books_df, x="review_score")
        fig.update_traces(marker_color='#d57f0e',
                          marker_line_color='rgb(80,15,11)',
                          marker_line_width=1.5)
        fig.update_layout(title_text='Review Score')
        fig.show()

    def getStopWords(self):
        nltk.download('stopwords')
        sWords = set(stopwords.words('english'))
        sWords.update({'would', 'could', 'should', 'i', 'we', 'she', 'he', 'it'})
        print("###### STOP WORDS ######")
        # print(sWords, "\n###############\n")
        return sWords

    def analyzeSentiments(self):
        print('analyze sentiments')
        self.books_df['sentiment'] = ['positive' if rating > 3 else
                                      'negative' if rating < 3
                                      else 'neutral' for rating in self.books_df['review_score']]
        print('generate_special_word_clouds all sentiments')
        self.generate_special_word_clouds('all')
        print('generate_special_word_clouds positive sentiments')
        self.generate_special_word_clouds('positive')
        print('generate_special_word_clouds negative sentiments')
        self.generate_special_word_clouds('negative')
        print(self.books_df.head(5))

    def generate_special_word_clouds(self, sentimentType='all'):
        # print(self.books_df.head(10))
        # print(self.books_df['sentiment'].unique())
        corpus_txt = ""
        if (sentimentType.lower() == 'all'):
            corpus_txt = " ".join(x for x in self.books_df['review_text'])
        else:
            reviews_all_sub = self.books_df[self.books_df['sentiment'] == sentimentType]
            corpus_txt = " ".join(x for x in reviews_all_sub['review_text'])
        frequencyDict = self.calculate_word_frequencies(corpus_txt, False, True)
        fileName = ""
        match sentimentType.lower():
            case 'all':
                fileName = r"/Users/hema/PycharmProjects/TSOM/Data/AllReviews.png"
            case 'positive':
                fileName = r"/Users/hema/PycharmProjects/TSOM/Data/PosReviews.png"
            case 'negative':
                fileName = r"/Users/hema/PycharmProjects/TSOM/Data/NegReviews.png"
            case _:
                fileName = r"/Users/hema/PycharmProjects/TSOM/Data/otherReviews.png"
        print('generateWordCloud ',sentimentType.lower())
        self.generateWordCloud(frequencyDict, fileName)

    def generateWordCloud(self, frequencies, path):
        # Define WordCloud parameters
        wordcloud_params = {
            'max_words': 100,  # Maximum number of words to display
            'max_font_size': 50,  # Maximum font size for words
            'stopwords': set(stopwords.words('english')),  # Set of stopwords to exclude
            'width': 800,  # Width of the word cloud image
            'height': 400,  # Height of the word cloud image
            'background_color': 'white',  # Background color of the word cloud image
            'collocations': False,  # Disable collocations
            'mask': None,  # Optional custom mask for the word cloud
            'font_path': None,  # Path to a custom font file
            'scale': 1  # Scale of the word cloud relative to mask size
        }

        # use this code to generate the word cloud
        cloud = wordcloud.WordCloud(**wordcloud_params)
        cloud.generate_from_frequencies(frequencies)
        cloud.to_file(path)
        print("FILE GENERATED:", path)

        plt.interactive(True)
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: ' + path, fontweight="bold")
        plt.show()

        del cloud
        del frequencies

    def calculate_word_frequencies(self, text, stemmed=False, withoutStopWords=False):
        tokens = []
        if stemmed:
            tokens = self.tokenize_withstemming(text)
        else:
            tokens = self.tokenize(text)
        final_tokens = []
        if withoutStopWords:
            final_tokens = self.remove_stopwords(tokens)
        else:
            final_tokens = tokens
        tkn_count_dict = self.get_token_counts(final_tokens)
        return tkn_count_dict

    def tokenize_withstemming(self, doc):
        tokens = self.tokenize(doc)
        # Stemming
        porterstem = stem.PorterStemmer()
        stemTokens = [porterstem.stem(x) for x in tokens]
        return stemTokens

    def tokenize(self, doc):
        text = doc.lower().strip()
        text = re.sub(f'[{string.punctuation}]', " ", text)
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def remove_stopwords(self, token_list):
        tokensFiltered = [token for token in token_list if token not in self.stopWords]
        return tokensFiltered

    def get_token_counts(self, token_list):
        token_counter = collections.Counter([txt.lower() for txt in token_list])
        return dict(sorted(token_counter.items(), key=lambda item: item[1], reverse=True))

    def remove_punc_stopwords(self, text):
        text_tokens = self.tokenize(text)
        text_tokens = self.remove_stopwords((text_tokens))
        return "".join(text_tokens)

    def runTextPredictions(self):

        """ Remove punctuations and stopwords from the text data in ReviewText and Title"""
        self.books_df['title'] = [str(x) for x in self.books_df['title']]
        self.books_df['review_text'] = [str(x) for x in self.books_df['review_text']]

        books_reg = copy.deepcopy(self.books_df)

        # The following applies remove_punc_stopwords function to each value in the given column. The result is a column with lower case values
        # that have no punctuations, no stop words,
        books_reg['title'] = books_reg['title'].apply(self.remove_punc_stopwords)
        books_reg['review_text'] = books_reg['review_text'].apply(self.remove_punc_stopwords)

        # Preprocessing and feature extraction for the final books dataframe
        X = books_reg['review_text']  # Text-based features
        y = books_reg['review_score']  # Target variable

        # Split the dataset into training and testing sets with 15% as test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=16)

        # TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Train multinomial logistic regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = model.predict(X_test_tfidf)

        # Model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Assuming you have already split your data into X_test and y_test
        # and you have predictions in y_pred
        print('Evaluating model')
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        target_names = ['1', '2', '3', '4', '5']
        print('Classification report:')
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Assuming you have already split your data into X_test and y_test
        # and you have predictions in y_pred
        print('Confusion Matrix Plot')
        cm = confusion_matrix(y_test, y_pred)

        # Create a DataFrame from the confusion matrix
        classes = range(1,6)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        print('Heat map ')
        sns.heatmap(cm_df, annot=True, cmap='crest', fmt='g')
        plt.title('Confusion Matrix - Multinomial Regression', y=1.1)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()

arp = AmazonBookReviewProject()
# Record the start time
start_time = datetime.datetime.now()
arp.run_project_work()
# Record the end time
end_time = datetime.datetime.now()
# Calculate the total runtime
runtime = end_time - start_time
print("Total runtime:", runtime)








