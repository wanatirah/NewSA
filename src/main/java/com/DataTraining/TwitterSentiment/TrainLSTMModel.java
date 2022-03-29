package com.DataTraining.TwitterSentiment;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class TrainLSTMModel {
    public static String DATA_PATH = "";
    public static WordVectors wordVectors;

    public static void main(String[] args) throws Exception {
        String dataLocalPath = DownloaderUtility.TWITTERDATA.Download();
        DATA_PATH = new File(dataLocalPath,"LabelledTweets").getAbsolutePath();

        int batchSize = 500;     //Number of examples in each minibatch
        int nEpochs = 10;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 250;  //Truncate reviews with length (# words) greater than this
        int seed = 12345;
        double lr = 1e-3;

        //DataSetIterators for training and testing respectively
        wordVectors = WordVectorSerializer.readWord2VecModel(new File(dataLocalPath,"TwitterWordVector.txt"));

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        SentimentIterator iTrain = new SentimentIterator.Builder()
                .dataDirectory(DATA_PATH)
                .wordVectors(wordVectors)
                .batchSize(batchSize)
                .truncateLength(truncateReviewsToLength)
                .tokenizerFactory(tokenizerFactory)
                .train(true)
                .build();

        SentimentIterator iTest = new SentimentIterator.Builder()
                .dataDirectory(DATA_PATH)
                .wordVectors(wordVectors)
                .batchSize(batchSize)
                .tokenizerFactory(tokenizerFactory)
                .truncateLength(truncateReviewsToLength)
                .train(false)
                .build();

        int inputNeurons = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        int outputs = iTrain.getLabels().size();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        //Set up network configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .l2(0.1)
                .list()
                .layer(new LSTM.Builder()
                        .nIn(inputNeurons)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new RnnOutputLayer.Builder()
                        .nIn(100)
                        .nOut(outputs)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();

        System.out.println("Starting training...");
        net.setListeners(
                new ScoreIterationListener(1)
//                , new EvaluativeListener(iTest, 1, InvocationType.EPOCH_END)
        );
        net.fit(iTrain, nEpochs);

        Evaluation evalTrain = net.evaluate(iTrain);
        System.out.println("\n ================= Evaluating Train: \n" + evalTrain.stats());

        Evaluation evalTest = net.evaluate(iTest);
        System.out.println("\n ================= Evaluating Test: \n" + evalTest.stats());

//        net.save(new File(dataLocalPath,"TwitterModel.net"), true);
//        System.out.println("----- Example complete -----");
    }
}
