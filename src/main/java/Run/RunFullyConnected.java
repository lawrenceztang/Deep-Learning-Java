package Run;

import Display.DisplayImage;
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
        ArrayList<ArrayList<Double>> trainingData = new ArrayList<ArrayList<Double>>();
        ImageReader reader = new ImageReader(trainingDataPath);
  //      BufferedImage imaged = Util.convert1dArrayToImage(reader.getImageAs1DMatrix("C:\\Users\\Anonymous\\Pictures\\Fortnite\\1\\0.jpg", 100), 100, 100);

        trainingData = reader.get1dColorMatricesFromImages(8000, 28);
        reader.setPreprocessParameters(trainingData);
        trainingData = reader.preprocessTrainingSet(trainingData);
        ArrayList<ArrayList<Double>> trainingDataOutputs = reader.oneHotOutputs;


        FullyConnectedNetwork network = new FullyConnectedNetwork(4, 200 , 1932, 10, .05, .9);
        network.printStats();
        network.classNames = reader.classes;
        System.out.println("Percentage accurate before training: " + network.test(trainingData, trainingDataOutputs));

        double time = System.currentTimeMillis();

        int batchSize = 1;
        for (int p = 0; p < 1; p++) {
            for (int i = 0; i < trainingData.size(); i += batchSize) {
                ArrayList<ArrayList<Double>> tempIn = new ArrayList<ArrayList<Double>>();
                ArrayList<ArrayList<Double>> tempOut = new ArrayList<ArrayList<Double>>();
                for (int a = 0; a < batchSize; a++) {
                    tempIn.add(trainingData.get(i));
                    tempOut.add(trainingDataOutputs.get(i));
                }

                network.setDropout(.8);
                network.getDerivativeOfErrorWithRespectToWeights(tempIn, tempOut);

                //test derivative to see if theyre accurate
//                System.out.println(network.derivativesErrorWithRespectToWeights.get(1).get(190).get(1900));
//                System.out.println(network.derivativeOfWeightCheck(tempIn.get(0), tempOut.get(0), 1, 190, 1900));

                network.gradientDescent();


            }
        }

        System.out.println("It took " + (System.currentTimeMillis() - time) / (1000) + " seconds to train");

        //testing
        Scanner scan = new Scanner(System.in);
        ArrayList<ArrayList<Double>> testingData = reader.get1dColorMatricesFromImages(20, 28);
        testingData = reader.preprocessTrainingSet(testingData);
        ArrayList<ArrayList<Double>> testingOutputs = reader.oneHotOutputs;
        System.out.println("Percentage accurate after training on training data: " + network.test(trainingData, trainingDataOutputs));
        System.out.println("Percentage accurate after training on testing data: " + network.test(testingData, testingOutputs));

        //dreaming
        FullyConnectedNetwork.DeepDream dream = network.new DeepDream(network, trainingData.get(0), 1000);
        for (int i = 0; i < 100; i++) {
            dream.updateImage(2);
        }

        BufferedImage dreamedImage = Util.convert1dArrayToImage(reader.unpreprocessExample(dream.image), 28, 28);
        BufferedImage previousImage = Util.convert1dArrayToImage(reader.unpreprocessExample(trainingData.get(0)), 28, 28);
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
