# NewSA
//This project is incomplete

## REAL-TIME TWEETS SENTIMENT ANALYSIS WEB APPLICATION THROUGH DL4J, TWITTER4J, SPRING BOOT AND REACTJS
This project is about building a web application for real-time tweets sentiment analysis.
The flow of this project:
1. Training the LSTM model using twitter malay dataset obtain from kaggle with DL4J 
   https://www.kaggle.com/code/kerneler/starter-malaysia-twitter-sentiment-65d5c14e-c/data
2. Connect to twitter api using the twitter 4j but of course have to make twitter developer account first
   to obtain the necessary credentials to access the api
3. Build and inference class to connect the twitter4j and dl4j. Therefore, new tweets can be predicted as 
   either positive or negative.
4. Apply Spring Boot to send the result to the web page http://localhost:8080/
5. Build the frontend page using reactjs. The origin web for reactjs is http://localhost:3000/

### DL4J
- The model built is LSTM, Long Short Term Memory model. LSTM is the popular choice when it comes to 
analysis text data as it can process the sequence of the text itself. Text differed with numerical data
as each word is interconnected to convey its meaning/context in a sentence.
- Before building the lstm model, word2vec model is first trained and evaluate to obtain word vectors of the 
train and test dataset. 
- The reason why we need word2vec model to generate the word vectors is so that the computer could learn the 
connection between each word in a sentence and this connection is fed into lstm model to determine its polarity.
- Well, the other reason is because the computer cannot read text and thus we need to generate a vector for each 
word to enable the computer to read and analyse it.

##### Word2Vec evaluation
- vector size: 100 ~chosen
Similarity between twitter and tweet 0.6123586297035217
[facebook, akaun, tweet, myspace, fb, gtbe, followedgtthen, malya, kampgampcambffampiltsm, pengikut]

- vector size: 200
Similarity between twitter and tweet 0.5003746151924133
[facebook, malya, akaun, pengikut, tweet, kampgampcambffampiltsm, briward, gtbe, menautkan, followedgtthen]

- vector size: 300
Similarity between twitter and tweet 0.4338834285736084
[facebook, malya, tweet, akaun, hiroyo, tablo, gtbe, followedgtthen, pengikut, briward]

#### LSTM evaluation
- Evaluating Train: 
- Accuracy:        0.8300
- Precision:       0.8211
- Recall:          0.8437
- F1 Score:        0.8323

- Confusion Matrix
-      0      1
- 195893  44107 | 0 = Negative
-  37516 202484 | 1 = Positive


- Evaluating Test: 
- Accuracy:        0.8097
- Precision:       0.8224
- Recall:          0.7901
- F1 Score:        0.8059

-Confusion Matrix
-     0     1
- 49762 10238 | 0 = Negative
- 12597 47403 | 1 = Positive

#### Analysis
: Word2Vec model - The larger the layer size the smaller the distance computed between two similar word

: LSTM model - The result is a bit underfitting at an average of about 80% for all metrics. 
- We measure mostly using the F1-score as F1 score is the weighted average of Precision and Recall. 
  Thus, it takes both the false positives and false negatives into consideration so that the predicted 
  result is more reliable as there will be less wrong prediction.
- The other measure to be considered is the accuracy. Accuracy takes into consideration of all correct 
  prediction against all predicted result. However, F1 score is still better than accuracy as it works better
  with imbalance dataset which is always is the case for real-time data.

##### Training model process
1. Using only about 1,000 tweets --> underfitting at about 30%
   *Solution : Adding more data about 600,000 tweets --> performance improve to about 70% with 10 epoch
2. Adding LSTM layer --> not much change happening
3. Adding more epoch, 100 --> performance increase about 80%, adding more than this performance deteriorate.
Thus, it is safe to say that to achieve more and high performance of the model, more training data is required.
