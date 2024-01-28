
## Overview of the App

This app is a news feed Q&A, build using open source databricks/dolly-v2-3b LLM and Vector DB ChromDB

## Run it locally 
From Linux machine root directory ec2-linux
```sh
python3 -m venv venv/  
source ./venv/bin/activate
llm_model_news.py
```
From Windows machine root directory
```sh
windows local
.\venv\Scripts\activate

```
Download the dataset from Kaggle and place in root directory
https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset

## build a LLM model
From Linux machine root directory ec2-linux
```sh
llm_model_news.py
```

## test a LLM model
From Linux machine root directory ec2-linux
```sh
llm_model_news_test.py
```