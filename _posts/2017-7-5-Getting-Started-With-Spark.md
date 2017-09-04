---
layout: post
title: Getting Started with Spark
---

![Spark]({{ site.baseurl }}/images/spark.png)
I started learning data science back in my last year at college by taking Big Data Management course. The course just opened for the first time back then, and it brought me tons of curiousity of what can data do in real life. It’s taught by [Mr. Bayu](https://www.cs.ui.ac.id/index.php/bayu-distiawan-t.html?lang=en) and [Mr. Alfan](https://www.linkedin.com/in/alfan-farizki-wicaksono-26484885/?ppe=1), who inspired me to learn more about this field.

In this course, we got a chance to do a last project using big data in a group of three. I’ll tell you about my project–it’s pretty simple, therefore it has many flaws. The idea is, I want to classify user’s interest based on their tweets. Since the Twitter API brings us all the easiness to fetch user data, it makes us easier to analyze user’s tweets. The goal here is that we can perform a data analysis using big data approach, so it maybe not the best approach to find user’s interests.

Platform that we use in this project is Apache Spark with Python (PySpark). It’s pretty easy to perform big data task using Spark and it also [runs 100 times](https://www.xplenty.com/blog/2014/11/apache-spark-vs-hadoop-mapreduce/) faster than MapReduce because it runs in-memory on cluster. Not only that, Spark has [a lot of libraries](https://spark.apache.org/mllib/) to support machine learning process in big data, so it’s my go-to supervised learning with big data.

### The workflow
<br>
![Spark Flow]({{ site.baseurl }}/images/spark-flow.png)
<p style="font-style: italic; text-align: center">The workflow</p>

That’s the workflow for this project. First, we define what are the user interest. We browse through websites and assume that there are 8 categories: *Science, Food & Drink, News, Shopping, Law & Government, Travel, Leisure & Hobby, Education & Employment, Home & Garden, and Sports*. After we define those categories, then we find Twitter users who speak english related to those categories. For example, in News category we fetch tweets from [@cnnbrk](http://twitter.com/cnnbrk), [@mashable](http://twitter.com/mashable), [@weatherchannel](http://twitter.com/weatherchannel), etc. We took only hundreds of their latest tweet with Twitter API and give each tweet a label of their category. Indeed, it may brings incosistency since not all their tweet can be labeled to the category we defined, but this is out of scope in this project.

### Text pre-processing

The next step which consumes a lot of time, cleaning the tweets. We can’t treat UGC from microblogging (user generated content) like we treat a good english paragraph that we find in article. Twitter has developed new behavior in the way people communicate with each other. There are retweets, mention, hashtag, emoji, and url contain in most of the tweet. We use NLTK (library for nlp) and [tweetpreprocessor](https://pypi.python.org/pypi/tweet-preprocessor/0.4.0) to clean the tweets. Here are some tasks we perform: removing stopwords, stemming, removing RT, mentions, url, emoji, and removing tweets which has less than 5 words. This cleaning method is also not optimized because there are other problems such as normalization of slang word, word shrinking, etc. We don’t deep dive into that. We run all this cleaning process on Python, not in HDFS.
<br><br>
```
Real Madrid leads Club América 1-0 at halftime in the Club World Cup. RM: unbeaten in 35 straight games, 2nd-longest in Spanish Club history
```
```
real madrid lead club amrica halftim club world cup unbeaten straight game ndlongest spanish club histori
```
<p style="font-style: italic; text-align: center">Before-After cleaning</p>

### Classification in Spark
Start from now, we will run the code in Spark, so we need to move the text files to HDFS. The text files that we used already divided by their label, so in folder category1 will ony be filled with tweets associated to category 1.

```python
category1 = sc.textFile("category1")
category1_RDD = category1.map(lambda p: Row(tweet=p, label=float(1)))
category1_DF = sqlContext.createDataFrame(category1_RDD)
category1_DF.write.parquet("all_category/key=1")

mergedDF = sqlContext.read.option("mergeSchema", "true").parquet("all_category")

tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
wordsData = tokenizer.transform(mergedDF)
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
labeled = rescaledData.map(lambda row: LabeledPoint(row.label, row.features))

(trainingData, testData) = labeled.randomSplit([0.7, 0.3])
model = NaiveBayes.train(trainingData, 1.0)

predictionAndLabel = testData.map(lambda p: LabeledPoint(p.label, model.predict(p.features)))
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / testData.count()
```

First, we need to represent our text data into RDD. After that, we create new RDD by transforming the previous RDD. Think of DataFrame, we basically create a table which contain the text and label. And yes, we can create DataFrame also from RDD. This DF can be query using SQL queries syntax too. We transform every single category and then merged them into one single DF. Now we have a single DF that contains all the text and label (which represented by float number). We can use Spark’s machine learning library, and it’s quite simple.

Next, we do the feature engineering. The features that we use only the TF-IDF. TF will find how many times a term occurs in document. We assume that the higher the TF, the more it is important to the document. But, we also need IDF (Inverse Document Frequency). Why? Because the more a term t occurs throughout all documents, the more poorly t discriminates between documents. So, we need IDF which computed at the corpus level, not at the document level. And, if we want to weight terms highly if they are frequent in relevant documents but also infrequent in the collection as a whole (so we can discriminate document more accurate), we can calculate the TF score x IDF score (that’s why we called TF-IDF). The final representation of our document will be a vector which includes the TF-IDF score.

<br>
![TF-IDF]({{ site.baseurl }}/images/tf-idf.png)
<p style="font-style: italic; text-align: center">TF-IDF</p>

Once we have that, we can train our data with classifier that we want. We run the train/test split in this case, by divide our data to 70% of training data and 30% test data. We use Naive Bayes classifier, because sometimes it’s more accurate in text classification, and also it’s pretty fast.

![Naive Bayes Result]({{ site.baseurl }}/images/naive3.png)
<p style="font-style: italic; text-align: center">Prediction result with unseen data</p>