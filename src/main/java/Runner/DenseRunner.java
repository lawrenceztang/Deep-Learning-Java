package Runner;

import Display.DisplayImage;
import Network.DenseNetwork;
import Util.ArrOperations;
import Reader.ImageReader;
import Util.NetworkSaver;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.DecimalFormat;
import java.util.Scanner;

//class that integrates reader and network and runs network for user - less customization, higher level architecture
//also logs progress

public class DenseRunner {
    public int batchSize = 1;
    public float[] learningRate;
    public int epochs = 10;
    public int iterations = 1000;
    public int[] layers;
    public float momentum = .9f;
    public float dropoutProbability = 1;

    public int testingIterations = 1000;

    public String trainingDataPath;
    public String testingDataPath;
    public boolean isImage = true;
    public int imageWidth = 28;
    public int imageHeight = 28;

    DenseNetwork network;
    float[][] trainingData;
    float[][] trainingDataOutputs;
    ImageReader reader;

    Scanner scan;

    public DenseRunner() {
        scan = new Scanner(System.in);
    }

    public void printStats() {
        System.out.println("Neural Network Parameters:");
        System.out.println("Layers: " + layers);
        System.out.print("Nodes in each layer: ");
        for (int i = 0; i < network.nodesPerLayer.length; i++) {
            System.out.print(network.nodesPerLayer[i] + ", ");
        }
        System.out.println();
        System.out.println("Learning rate: " + network.learningRate);
        System.out.println("Momentum: " + network.momentum);
        System.out.println();
    }

    public void train() throws Exception {
        System.out.println(System.getenv());

        reader = new ImageReader(trainingDataPath);
        if (isImage) {
            trainingData = reader.get1dColorMatricesFromImages(batchSize * iterations, imageWidth);
            reader.setPreprocessParameters(trainingData);
            trainingData = reader.preprocessTrainingSet(trainingData);
            trainingDataOutputs = reader.oneHotOutputs;
        } else {
            //it is a csv file
            trainingData = new float[2][];
            trainingDataOutputs = new float[2][];
        }


        network = new DenseNetwork(layers, learningRate, DenseNetwork.UPDATE_SGD, momentum, dropoutProbability);

        long time = System.currentTimeMillis();

        printStats();

        float[] averageDerivatives = new float[network.layers];
        for (int p = 0; p < epochs; p++) {
            System.out.println("Epoch: " + p);
            for (int i = 0; i < trainingData.length; i += batchSize) {

                if (i / batchSize % 1000 == 0) {
                    System.out.println("Iteration: " + i);
                }

                float[][] tempIn = new float[batchSize][];
                float[][] tempOut = new float[batchSize][];
                for (int a = 0; a < batchSize; a++) {
                    tempIn[a] = trainingData[i];
                    tempOut[a] = trainingDataOutputs[i];
                }

                network.getDerivativeOfErrorWithRespectToWeights(tempIn, tempOut);

                //difference in derivatives of different layers
                for (int z = 1; z < network.layers; z++) {
                    averageDerivatives[z] += Math.abs(network.derivativesErrorWithRespectToWeights[z][0][0]);
                }
                System.out.println(network.derivativeOfWeightCheck(tempIn[0], tempOut[0], 1, 0, 0));
                System.out.println(network.derivativesErrorWithRespectToWeights[1][0][0]);

                network.gradientDescent();
            }
        }

        for (int i = 1; i < network.layers; i++) {
            System.out.println("Average derivatives of layer " + i + ": " + averageDerivatives[i]);
        }

        System.out.println("It took " + (System.currentTimeMillis() - time) / 1000f + " seconds to train");
    }

    public void deepDream() throws Exception {
        DenseNetwork.DeepDream dream = network.new DeepDream(network, trainingData[0], 1000);
        for (int i = 0; i < 100; i++) {
            dream.updateImage(2);
        }

        BufferedImage dreamedImage = ArrOperations.convert1dArrayToImage(reader.unpreprocessExample(dream.image), imageWidth, imageHeight);
        BufferedImage previousImage = ArrOperations.convert1dArrayToImage(reader.unpreprocessExample(trainingData[0]), imageWidth, imageHeight);
        System.out.println(network.predictOutput(dream.image));
        new DisplayImage(dreamedImage);
        new DisplayImage(previousImage);
    }

    public void test() throws Exception {
        //testing
        Scanner scan = new Scanner(System.in);
        ImageReader testingReader = new ImageReader(testingDataPath, reader);
        float[][] testingData = reader.get1dColorMatricesFromImages(testingIterations, imageWidth);
        testingData = testingReader.preprocessTrainingSet(testingData);
        float[][] testingOutputs = reader.oneHotOutputs;
        System.out.println("Accuracy on training data: " + network.test(trainingData, trainingDataOutputs));
        System.out.println("Accuracy on testing data: " + network.test(testingData, testingOutputs));
    }

    public void testOnOneExample() throws Exception {

        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);

        if (ret == JFileChooser.APPROVE_OPTION) {

            File file = fc.getSelectedFile();
            String fileName = file.getAbsolutePath();
            float[] outputs = reader.preprocessExample(reader.getImageAs1DMatrix(fileName, 28));
            int maxOutput = 0;
            for (int i = 0; i < outputs.length; i++) {
                if (outputs[i] > outputs[maxOutput]) {
                    maxOutput = i;
                }
            }
            System.out.println(reader.classes[maxOutput]);
        }
    }

    public void saveNetwork() throws Exception {
        String saveLocation = "C:\\Users\\Anonymous\\Documents\\DeepLearningSaves\\";
        //saving network
        System.out.println("Would you like to save the network? (y / n)");
        if (scan.nextLine().equals("y")) {
            System.out.println("Saving...");
            (new NetworkSaver()).saveObject(network, saveLocation + "network");
            (new NetworkSaver()).saveObject(reader, saveLocation + "reader");
            System.out.println("Done saving");
        }
    }

    public DenseRunner setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public DenseRunner setLearningRate(float[] learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public DenseRunner setEpochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    public DenseRunner setIterations(int iterations) {
        this.iterations = iterations;
        return this;
    }

    public DenseRunner setLayers(int[] layers) {
        this.layers = layers;
        return this;
    }

    public DenseRunner setMomentum(float momentum) {
        this.momentum = momentum;
        return this;
    }

    public DenseRunner setTestingIterations(int testingIterations) {
        this.testingIterations = testingIterations;
        return this;
    }

    public DenseRunner setTrainingDataPath(String trainingDataPath) {
        this.trainingDataPath = trainingDataPath;
        return this;
    }

    public DenseRunner setTestingDataPath(String testingDataPath) {
        this.testingDataPath = testingDataPath;
        return this;
    }

    public DenseRunner setIsImage(boolean image) {
        isImage = image;
        return this;
    }

    public DenseRunner setImageWidth(int imageWidth) {
        this.imageWidth = imageWidth;
        return this;
    }

    public DenseRunner setImageHeight(int imageHeight) {
        this.imageHeight = imageHeight;
        return this;
    }

    public DenseRunner setDropout(float dropoutProbability) {
        this.dropoutProbability = dropoutProbability;
        return this;
    }
}
