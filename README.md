## How have reporting on topics surrounding covid changed in 2022 as compared to 2021?

<div align="center">
  
| File | Description |
|---|---|
| [Main Notebook](https://github.com/SitwalaM/nlp-topic-modelling/blob/main/Topic_Modelling_Final_TeamB.ipynb) | Main Notebook submitted for Twist Challenge  |
| [Data Extraction Notebook](https://github.com/SitwalaM/nlp-topic-modelling/blob/main/scripts/nlp_dag.py) | Kaggle Notebook used for Data Extraction |
|[Dashboard](https://public.tableau.com/views/Tanamadosi1/Dashboard?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link)| Dashboard using Flask|
  
</div>

As global cases of COVID19 began to rise from Dec 2019, we witnessed an increase in news reporting on COVID19 related topics (such as panic, shortage, testing, quarantine etc) globally on a daily basis. Every news agency be it TV or Online had something to report, mostly negative. If 2020 was dominated by the news of how COVID-19 spread across the globe, then 2021 has so far been focused on ending the pandemic through vaccine distribution. As vacciness began to roll out and the rate of deaths reducing significantly, the rate of reporting began to decrease and media focus was shifted to new topics.

This project is to compare how reporting on topics surrounding covid has changed between 2021 and 2022. The initial plan was to compare the first quarter of 2021 and 2022 but given the size of the data, it had to be limited to the months of January only. After which i will train a news category classifier with the data using NLP on the data from January 2022 and deploy it using flask.

# Data

The data was extracted from the gdelt public data available on Google big query and because of the limitations and difficulty in getting it directly from Google Big Query, I extracted it through kaggle which can access Big Query directly. The final step was to upload it into google drive for use in Colab.

This data reflects news reported online. The notebook used has been uploaded in GitHub.

As I earlier stated in my introduction, the data had to be limited to just one month of each year due to the download size.

# EDA

The use of line graph, bar graph and pie chart was used on the raw data to visualize the change in reporting on covid related topics for 2021 and 2022. The output of the visualizations did prove the fact that reporting on covid related topics indeed reduced in January 2022 which could be as a result of the roll out of vaccines in 2020 hence less reported cases and deaths.

<div align="center">
  
<img src=....width="400">
  
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
  
<img src=....width="400">
  
</div>

## Lemmarization

The preprocessing step continued with 





It’s time to train a machine learning model on the vectorized dataset and test it. Now that we have converted the text data to numerical data, we can run ML models on X_train_vector_tfidf & y_train. We’ll test this model on X_test_vectors_tfidf to get y_predict and further evaluate the performance of the model

Logistic Regression
Naive Bayes: It’s a probabilistic classifier that makes use of Bayes’ Theorem, a rule that uses probability to make predictions based on prior knowledge of conditions that might be related
In this article, I demonstrated the basics of building a text classification model comparing Bag-of-Words (with Tf-Idf) and Word Embedding with Word2Vec. You can further enhance the performance of your model using this code by

using other classification algorithms like Support Vector Machines (SVM), XgBoost, Ensemble models, Neural networks etc.
using Gridsearch to tune the hyperparameters of your model
using advanced word-embedding methods like GloVe and BERT


