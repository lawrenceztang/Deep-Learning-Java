package Run;

import Network.FullyConnectedNetwork;
import Util.*;


import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Scanner;


public class RunFullyConnected {

    public static void main(String[] args) throws Exception {

        System.out.println(System.getenv());

        //get training data
        String trainingDataPath = "C:\\Users\\Anonymous\\Pictures\\Numbers\\mnist_png\\training";

        ImageReader reader = new ImageReader(trainingDataPath);
  //      BufferedImage imaged = arrOperations.convert1dArrayToImage(reader.getImageAs1DMatrix("C:\\Users\\Anonymous\\Pictures\\Fortnite\\1\\0.jpg", 100), 100, 100);

        double[][] trainingData = reader.get1dColorMatricesFromImages(4000, 28);
        reader.setPreprocessParameters(trainingData);
        trainingData = reader.preprocessTrainingSet(trainingData);
        double[][] trainingDataOutputs = reader.oneHotOutputs;


        FullyConnectedNetwork network = new FullyConnectedNetwork(new int[]{2352, 200, 10}, new double[]{0, 0.02, .001}, .9);
        network.printStats();
        network.classNames = reader.classes;
        System.out.println("Percentage accurate before training: " + network.test(trainingData, trainingDataOutputs));

        double time = System.currentTimeMillis();

        int batchSize = 1;
        double averageFirstLayerDerivatives = 0;
        double averageSecondLayerDerivatives = 0;
        for (int p = 0; p < 1; p++) {
            for (int i = 0; i < trainingData.length; i += batchSize) {
                double[][] tempIn = new double[batchSize][];
                double[][] tempOut = new double[batchSize][];
                for (int a = 0; a < batchSize; a++) {
                    tempIn[a] = trainingData[i];
                    tempOut[a] = trainingDataOutputs[i];
                }

                network.setDropout(1);
                network.getDerivativeOfErrorWithRespectToWeights(tempIn, tempOut);

                //test derivative to see if theyre accurate
//                System.out.println(network.derivativesErrorWithRespectToWeights[1][190][1900]);
//                System.out.println(network.derivativeOfWeightCheck(tempIn[0], tempOut[0], 1, 190, 1900));
                averageFirstLayerDerivatives += Math.abs(network.derivativesErrorWithRespectToWeights[1][190][1900]);

//                System.out.println("second layer");
//                System.out.println(network.derivativesErrorWithRespectToWeights[2][7][100]);
//                System.out.println(network.derivativeOfWeightCheck(tempIn[0], tempOut[0], 2, 7, 100));
                averageSecondLayerDerivatives += Math.abs(network.derivativesErrorWithRespectToWeights[2][7][100]);

                network.gradientDescent();

            }
            System.out.println(averageFirstLayerDerivatives + ", " + averageSecondLayerDerivatives);
        }

        System.out.println("It took " + (System.currentTimeMillis() - time) / (1000) + " seconds to train");

        //testing
        Scanner scan = new Scanner(System.in);
        double[][] testingData = reader.get1dColorMatricesFromImages(200, 28);
        testingData = reader.preprocessTrainingSet(testingData);
        double[][] testingOutputs = reader.oneHotOutputs;
        System.out.println("Percentage accurate after training on training data: " + network.test(trainingData, trainingDataOutputs));
        System.out.println("Percentage accurate after training on testing data: " + network.test(testingData, testingOutputs));

        //dreaming
        FullyConnectedNetwork.DeepDream dream = network.new DeepDream(network, trainingData[0], 1000);
        for (int i = 0; i < 100; i++) {
            dream.updateImage(2);
        }

        BufferedImage dreamedImage = arrOperations.convert1dArrayToImage(reader.unpreprocessExample(dream.image), 28, 28);
        BufferedImage previousImage = arrOperations.convert1dArrayToImage(reader.unpreprocessExample(trainingData[0]), 28, 28);
        System.out.println(network.predictOutput(dream.image));
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
