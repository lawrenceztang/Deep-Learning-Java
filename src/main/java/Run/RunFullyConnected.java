package Run;

import Network.DenseNetwork;
import Reader.ImageReader;
import Util.*;


import java.text.DecimalFormat;
import java.util.Scanner;


public class RunFullyConnected {

    public static void main(String[] args) throws Exception {



        DecimalFormat df = new DecimalFormat("#.####");

        System.out.println(System.getenv());

        //get training data
        String trainingDataPath = "C:\\Users\\Anonymous\\Pictures\\Numbers\\mnist_png\\training";
        //"C:\\Users\\Anonymous\\Pictures\\Fortnite\\1\\0.jpg"

        ImageReader reader = new ImageReader(trainingDataPath);

        float[][] trainingData = reader.get1dColorMatricesFromImages(8000, 28);
        reader.setPreprocessParameters(trainingData);
        trainingData = reader.preprocessTrainingSet(trainingData);
        float[][] trainingDataOutputs = reader.oneHotOutputs;


        DenseNetwork network = new DenseNetwork(new int[]{2352, 200, 10}, new float[]{0, 0.05f, .0025f}, .9f);
        network.printStats();
        network.classNames = reader.classes;
        System.out.println("Percentage accurate before training: " + network.test(trainingData, trainingDataOutputs));

        float time = System.currentTimeMillis();

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

        System.out.println("It took " + (System.currentTimeMillis() - time) / (1000f) + " seconds to train");

        //testing
        Scanner scan = new Scanner(System.in);
        float[][] testingData = reader.get1dColorMatricesFromImages(200, 28);
        testingData = reader.preprocessTrainingSet(testingData);
        float[][] testingOutputs = reader.oneHotOutputs;
        System.out.println("Percentage accurate after training on training data: " + network.test(trainingData, trainingDataOutputs));
        System.out.println("Percentage accurate after training on testing data: " + network.test(testingData, testingOutputs));

        //dreaming
        DenseNetwork.DeepDream dream = network.new DeepDream(network, trainingData[0], 1000);
        for (int i = 0; i < 100; i++) {
            dream.updateImage(2);
        }

//        BufferedImage dreamedImage = ArrOperations.convert1dArrayToImage(reader.unpreprocessExample(dream.image), 28, 28);
//        BufferedImage previousImage = ArrOperations.convert1dArrayToImage(reader.unpreprocessExample(trainingData[0]), 28, 28);
//        System.out.println(network.predictOutput(dream.image));
//        new DisplayImage(dreamedImage);
//        new DisplayImage(previousImage);


        //specific example
        while (true) {
            System.out.println("Test?");
            if (!scan.nextLine().equals("y")) {
                break;
            }

            System.out.println("Prediction: " + network.test(reader));
        }


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
}
