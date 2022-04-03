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
The model built is LSTM, Long Short Term Memory model. LSTM is the popular choice when it comes to 
analysis text data as it can process the sequence of the text itself. Text differed with numerical data
as each word is interconnected to convey its meaning/context in a sentence.
Before building the lstm model, word2vec model is first trained and evaluate to obtain word vectors of the 
train and test dataset. 
The reason why we need word2vec model to generate the word vectors is so that the computer could learn the 
connection between each word in a sentence and this connection is fed into lstm model to determine its polarity.
Well, the other reason is because the computer cannot read text and thus we need to generate a vector for each 
word to enable the computer to read and analyse it.

