# COVID19 Online News Classification using NLP

## Gdelt Data Analysis - How has reporting on topics surrounding covid changed in 2022 as compared to 2021?

<div align="center">
  
| File | Description |
|---|---|
| [Main Notebook](https://github.com/SitwalaM/nlp-topic-modelling/blob/main/Topic_Modelling_Final_TeamB.ipynb) | Main Notebook submitted for Twist Challenge  |
| [Data Extraction Notebook](https://github.com/NancyArmah/COVID19-Online_news-Classification-using-NLP/blob/main/gdelt.ipynb) | Kaggle Notebook used for Data Extraction |
  
</div>

As global cases of COVID19 began to rise from Dec 2019, we witnessed an increase in news reporting on COVID19 related topics (such as panic, shortage, testing, quarantine etc) globally on a daily basis. Every news agency be it TV or Online had something to report, mostly negative. If 2020 was dominated by the news of how COVID-19 spread across the globe, then 2021 has so far been focused on ending the pandemic through vaccine distribution. As vacciness began to roll out and the rate of deaths reducing significantly, the rate of reporting began to decrease and media focus was shifted to new topics.

This project is to compare how reporting on topics surrounding covid has changed between 2021 and 2022. The initial plan was to compare the first quarter of 2021 and 2022 but given the size of the data, it had to be limited to the months of January only. After which i will train a news category classifier with the data using NLP on the data from January 2022.

# Data

The data was extracted from the gdelt public data available on Google big query and because of the limitations and difficulty in getting it directly from Google Big Query, I extracted it through kaggle which can access Big Query directly. The final step was to upload it into google drive for use in Colab.

This data reflects news reported online. The notebook used has been uploaded in GitHub.

As I earlier stated in my introduction, the data had to be limited to just one month of each year due to the download size.

# EDA

The use of line graph, bar graph and pie chart was used on the raw data to visualize the change in reporting on covid related topics for 2021 and 2022. The output of the visualizations did prove the fact that reporting on covid related topics indeed reduced in January 2022 which could be as a result of the roll out of vaccines in 2020 hence less reported cases and deaths.

<div align="center">
  
<img src="https://github.com/NancyArmah/COVID19-Online_news-Classification-using-NLP/blob/main/Linegraph.png">
  
<img src="https://github.com/NancyArmah/COVID19-Online_news-Classification-using-NLP/blob/main/Bargraph.png">

  
</div>

# NLP

Given the structure of our data, training a news category classifier with the data will be fitting because the data is already labelled. The focus will only be on one data set (i.e df2 = January 2022) and i randomly extracted a part of the data (20,000 rows) in order to reduce the run time and prevent the session from crashing.

Text classification is one of the important task in supervised machine learning (ML). It is a process of assigning tags/categories to documents helping us to automatically & quickly structure and analyze text.

## Data Preprocessing

Before we move to model building, we need to preprocess our dataset by removing punctuations & special characters, cleaning texts, removing stop words, applying lemmatization and vectorization. 

With the help of the Regex function, all noise from the data was cleaned in one step. 

Next, was to remove all stopwords. Stop words are generally filtered out before processing in natural language. words. By removing these words, we remove the low-level information from our text in order to give more focus to the important information. We use the NLTK library and import the English stop words list. Before doing this, i created a custom stopwords list for words not included in the nltk stopwords dicitonary.

### Word Cloud

This was done to visualize the most important words in the data.Tags are usually single words, and the importance of each tag is shown with font size or color.

<div align="center">
  
<img src="https://github.com/NancyArmah/COVID19-Online_news-Classification-using-NLP/blob/main/wordcloud.png" width="400">
  
</div>

## Tokenization

Next was to tokenize to split the texts into smaller units, so the meaning of the text could easily be interpreted by analyzing the words present in the text. Before processing a natural language, we need to identify the words that constitute a string of characters. 

## Lemmatization

The preprocessing step continued with lemmatization whcih brings a shorter word or base word. The advantage of this is, we get to reduce the total number of unique words in the dictionary. As a result, the number of columns in the document-word matrix created by TfidfVectorizer will be denser with lesser columns.

## Vectorization

Finally, converting the text into vectors (meaningful representation of numbers) with a vectorizer, as our model can only deal with numbers, not alphabets. First, as with any machine learning task, we split our data into 2 groups of rows, the train and test sets using the train_test_split() function before vectorizing it.

Then we we use a TfidfVectorizer. Basically when this vectorizer is fitted on our data, it takes note of all terms/vocabulary (i.e. all the words and phrases involved) present in the text that we give it. An ngram_range of (1,2) was used so that we are no longer treating a single word, but double words frequently occurring together in the document.

The tfidf_vectorizer was fitted on only the training data to note all the terms involved, we then transform the training data into vectors which account for the frequency of this terms. This is done by the fit_transform() function. We then use the same model to transform the test data into vectors.

We do NOT fit it on the test data as the test data is meant to simulate user input of text for cateogry prediction.

# Modelling

Now that we have converted the text data to numerical data, we can run ML models on X_train_vector_tfidf & y_train. We’ll test this model on X_test_vectors_tfidf to get y_predict and further evaluate the performance of the model. 

Five different models were trained on the data with the best model used for the dashboard.

<div align="center">
  
| Model | Accuracy Score |
|---|---|
| SVC| 0.459 |
| Logistic Regression | 0.567 |
| DecisionTree Classifier| 0.5795 |
| XGBoost Classifier| 0.573 |
| KNN | 0.467 |

</div>

# Observations & Future Improvements
  
This notebook demonstrats the basics of building a text classification model. 

For future improvements;

I observed from the data that all predictions are biased towards covid19 which is understandable because it is the largest topic in the dataset. The data is imbalanced, to improve this either an oversampling or undersampling technique can be used on the data to balance it and improve the results.

Also, using Gridsearch to fine tune the best selected model.

