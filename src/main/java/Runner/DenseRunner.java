package Runner;

import Display.DisplayImage;
import Network.DenseNetwork;
import Util.ArrOperations;
import Reader.ImageReader;
import Util.NetworkSaver;

import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.Scanner;

public class DenseRunner {
    public int batchSize = 1;
    public float[] learningRate;
    public int epochs = 10;
    public int iterations = 1000;
    public int[] layers;

    public int testingIterations = 1000;

    public String trainingDataPath;
    public String testingDataPath;
    public boolean isImage;
    public int imageWidth;
    public int imageHeight;

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
        System.out.println("Nodes in each layer: ");
        for(int i = 0; i < network.nodesPerLayer.length; i++) {
            System.out.print(network.nodesPerLayer[i] + ", ");
        }
        System.out.println("Learning rate: " + network.learningRate);
        System.out.println("Momentum: " + network.momentum);
    }

    public void train() throws Exception {
        System.out.println(System.getenv());

        reader = new ImageReader(trainingDataPath);
        if (isImage) {
            trainingData = reader.get1dColorMatricesFromImages(batchSize, 28);
            reader.setPreprocessParameters(trainingData);
            trainingData = reader.preprocessTrainingSet(trainingData);
            trainingDataOutputs = reader.oneHotOutputs;
        } else {
            //it is a csv file
            trainingData = new float[2][];
            trainingDataOutputs = new float[2][];
        }


        network = new DenseNetwork(layers, learningRate, .9f);

        network.classNames = reader.classes;

        float time = System.currentTimeMillis();

        printStats();

        int batchSize = 1;
        float[] averageDerivatives = new float[network.layers];
        for (int p = 0; p < 1; p++) {
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

                network.setDropout(1);
                network.getDerivativeOfErrorWithRespectToWeights(tempIn, tempOut);

                //difference in derivatives of different layers
                for (int z = 1; z < network.layers; z++) {
                    averageDerivatives[z] += Math.abs(network.derivativesErrorWithRespectToWeights[z][0][0]);
                }

                //test derivative to see if theyre accurate
//                for(int a = 1; a < network.derivativesErrorWithRespectToWeights.length; a++) {
//                    for(int q = 0; q < network.derivativesErrorWithRespectToWeights[a].length; q++) {
//                        for(int v = 0; v < network.derivativesErrorWithRespectToWeights[a][q].length; v++) {
//                            if(Float.parseFloat(df.format(new Float(network.derivativesErrorWithRespectToWeights[a][q][v] / 10))) != Float.parseFloat(df.format(new Float(network.derivativeOfWeightCheck(tempIn[0], tempOut[0], a, q, v))))) {
//                                System.out.println(network.derivativesErrorWithRespectToWeights[a][q][v] + ", " + network.derivativeOfWeightCheck(tempIn[0], tempOut[0], a, q, v));
//                            }
//                        }
//                    }
//                }
                network.gradientDescent();
            }
        }

        for (int i = 1; i < network.layers; i++) {
            System.out.println("Average derivatives of layer " + i + ": " + averageDerivatives[i]);
        }

        System.out.println("It took " + ((float) (System.currentTimeMillis() - time)) / 1000f + " seconds to train");
    }

    public void deepDream() throws Exception{
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

    public void test() throws Exception{
        //testing
        Scanner scan = new Scanner(System.in);
        ImageReader testingReader = new ImageReader();
        float[][] testingData = reader.get1dColorMatricesFromImages(testingIterations, imageWidth);
        testingData = reader.preprocessTrainingSet(testingData);
        float[][] testingOutputs = reader.oneHotOutputs;
        System.out.println("Accuracy on training data: " + network.test(trainingData, trainingDataOutputs));
        System.out.println("Accuracy on testing data: " + network.test(testingData, testingOutputs));
    }

    public void saveNetwork() throws Exception{
        String saveLocation = "C:\\Users\\Anonymous\\Documents\\DeepLearningSaves\\";
        //saving network
        System.out.println("Would you like to save the network? (y / n)");
        if (scan.nextLine().equals("y")) {
            System.out.println("Saving...");
            (new NetworkSaver()).saveObject(network, saveLocation + "mnistMyOwnNetwork");
            (new NetworkSaver()).saveObject(reader, saveLocation + "reader");
            System.out.println("Done saving");
        }
    }

    public static void main(String[] args) throws Exception {

    }





}
