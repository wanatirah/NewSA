package com.DataTraining.TwitterSentiment;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class SentimentIterator implements DataSetIterator {

    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;
    private int maxLength;
    private final String dataDirectory;
    private final List<Pair<String, List<String>>> sentimentData = new ArrayList<>();
    private int cursor = 0;
    private int totalTweets = 0;
    private final TokenizerFactory tokenizerFactory;
    private int tweetsPosition = 0;
    private final List<String> labels;
    private int currSentiment = 0;

    /**
     * @param dataDirectory  the directory of the twitter data set
     * @param wordVectors    WordVectors object
     * @param batchSize      Size of each minibatch for training
     * @param truncateLength If headline length exceed this size, it will be truncated to this size.
     * @param train          If true: return the training data. If false: return the testing data.
     *                       <p>
     *                       - initialize various class variables
     *                       - calls populateData function to load tweets data in sentiment vector
     *                       - also populates labels (i.e. sentiment related information) in labels class variable
     */
    private SentimentIterator(String dataDirectory,
                         WordVectors wordVectors,
                         int batchSize,
                         int truncateLength,
                         boolean train,
                         TokenizerFactory tokenizerFactory) {

        this.dataDirectory = dataDirectory;
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;
        this.tokenizerFactory = tokenizerFactory;
        this.populateData(train);
        this.labels = new ArrayList<>();
        for (Pair<String, List<String>> sentimentDatum : this.sentimentData) {
            this.labels.add(sentimentDatum.getKey().split(",")[1]);
        }
    }

    public static Builder Builder() {
        return new Builder();
    }


    @Override
    public DataSet next(int num) {
        if (cursor >= this.totalTweets) throw new NoSuchElementException();
        return nextDataSet(num);
    }

    private DataSet nextDataSet(int num) {
        // Loads tweets into tweet list from sentimentData List along with sentiment of each tweets
        List<String> tweets = new ArrayList<>(num);
        int[] sentiment = new int[num];

        for (int i = 0; i < num && cursor < this.totalTweets; i++) {
            if (currSentiment < sentimentData.size()) {
                tweets.add(this.sentimentData.get(currSentiment).getValue().get(tweetsPosition));
                sentiment[i] = Integer.parseInt(this.sentimentData.get(currSentiment).getKey().split(",")[0]);
                currSentiment++;
                cursor++;
            } else {
                currSentiment = 0;
                tweetsPosition++;
                i--;
            }
        }

        //Second: tokenize tweets and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(tweets.size());
        maxLength = 0;
        for (String s : tweets) {
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for (String t : tokens) {
                if (wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength, tokensFiltered.size());
        }

        //If longest tweets exceeds 'truncateLength': only take the first 'truncateLength' words
        //System.out.println("maxLength : " + maxLength);
        if (maxLength > truncateLength) maxLength = truncateLength;

        //Create data for training
        //Here: we have tweets.size() examples of varying lengths
        INDArray features = Nd4j.create(tweets.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(tweets.size(), this.sentimentData.size(), maxLength);    //Three labels: Negative, Neutral and Positive

        //Because we are dealing with tweets of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(tweets.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(tweets.size(), maxLength);

        int[] temp = new int[2];
        for (int i = 0; i < tweets.size(); i++) {
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in tweets, and put them in the training data
            for (int j = 0; j < tokens.size() && j < maxLength; j++) {
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{point(i),
                        all(),
                        point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);
            }
            int idx = sentiment[i];
            int lastIdx = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);
            labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);
        }

        return new DataSet(features, labels, featuresMask, labelsMask);
    }

    /**
     * Used post training to load a review from a file to a features INDArray that can be passed to the network output method
     *
     * @param file      File to load the review from
     * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not truncate
     * @return Features array
     * @throws IOException If file cannot be read
     */
    public INDArray loadFeaturesFromFile(File file, int maxLength) throws IOException {
        String tweets = FileUtils.readFileToString(file, (Charset)null);
        return loadFeaturesFromString(tweets, maxLength);
    }

    /**
     * Used post training to convert a String to a features INDArray that can be passed to the network output method
     *
     * @param reviewContents Contents of the review to vectorize
     * @param maxLength      Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not truncate
     * @return Features array for the given input String
     */
    public INDArray loadFeaturesFromString(String reviewContents, int maxLength) {
        List<String> tokens = tokenizerFactory.create(reviewContents).getTokens();
        List<String> tokensFiltered = new ArrayList<>();
        for (String t : tokens) {
            if (wordVectors.hasWord(t)) tokensFiltered.add(t);
        }
        int outputLength = Math.max(maxLength, tokensFiltered.size());

        INDArray features = Nd4j.create(1, vectorSize, outputLength);

        for (int j = 0; j < tokens.size() && j < maxLength; j++) {
            String token = tokens.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            features.put(new INDArrayIndex[]{point(0),
                    all(),
                    point(j)}, vector);
        }
        return features;
    }

    /*
    This function loads tweets from files stored in resources into sentimentData List.
     */
    private void populateData(boolean train) {
        File sentiments = new File(this.dataDirectory + File.separator + "Sentiment.txt");

        try (BufferedReader brCategories = new BufferedReader(new FileReader(sentiments))) {
            String temp;
            while ((temp = brCategories.readLine()) != null) {
                String curFileName = train ?
                        this.dataDirectory + File.separator + "Train" + File.separator + temp.split(",")[0] + ".txt" :
                        this.dataDirectory + File.separator + "Test" + File.separator + temp.split(",")[0] + ".txt";
                File currFile = new File(curFileName);
                BufferedReader currBR = new BufferedReader((new FileReader(currFile)));
                String tempCurrLine;
                List<String> tempList = new ArrayList<>();
                while ((tempCurrLine = currBR.readLine()) != null) {
                    tempList.add(tempCurrLine);
                    this.totalTweets++;
                }
                currBR.close();
                Pair<String, List<String>> tempPair = Pair.of(temp, tempList);
                this.sentimentData.add(tempPair);
            }
        } catch (Exception e) {
            System.out.println("Exception in reading file :" + e.getMessage());
        }
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return this.sentimentData.size();
    }

    @Override
    public void reset() {
        cursor = 0;
        tweetsPosition = 0;
        currSentiment = 0;
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return this.labels;
    }

    @Override
    public boolean hasNext() {
        return cursor < this.totalTweets;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public int getMaxLength() {
        return this.maxLength;
    }

    public static class Builder {
        private String dataDirectory;
        private WordVectors wordVectors;
        private int batchSize;
        private int truncateLength;
        TokenizerFactory tokenizerFactory;
        private boolean train;

        Builder() {
        }

        public SentimentIterator.Builder dataDirectory(String dataDirectory) {
            this.dataDirectory = dataDirectory;
            return this;
        }

        public SentimentIterator.Builder wordVectors(WordVectors wordVectors) {
            this.wordVectors = wordVectors;
            return this;
        }

        public SentimentIterator.Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public SentimentIterator.Builder truncateLength(int truncateLength) {
            this.truncateLength = truncateLength;
            return this;
        }

        public SentimentIterator.Builder train(boolean train) {
            this.train = train;
            return this;
        }

        public SentimentIterator.Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public SentimentIterator build() {
            return new SentimentIterator(dataDirectory,
                    wordVectors,
                    batchSize,
                    truncateLength,
                    train,
                    tokenizerFactory);
        }

        public String toString() {
            return "com.DataTraining.TwitterSentiment.SentimentIterator.Builder(dataDirectory=" +
                    this.dataDirectory + ", wordVectors=" + this.wordVectors +
                    ", batchSize=" + this.batchSize + ", truncateLength="
                    + this.truncateLength + ", train=" + this.train + ")";
        }
    }
}
