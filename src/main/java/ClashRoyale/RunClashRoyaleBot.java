package ClashRoyale;

import Network.DenseNetwork;
import Reader.ImageReader;

import java.awt.*;
import java.awt.event.InputEvent;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class RunClashRoyaleBot {

    static int[] layers;
    static float[] learningRate;
    static float momentum = .9f;
    static float dropoutProbability = 1;
    static int batchSize = 1;
    static int iterations = 100;
    static int imageWidth = 50;
    static int epochs = 10;

    public static void main(String[] args) throws Exception{

        learningRate = new float[]{0, .0000000004f, .00025f};
        layers = new int[]{3450, 200, 9};

        ImageReader reader = new ImageReader("C:\\Users\\Anonymous\\Pictures\\Clash_Royale\\Supervised_Bot");
        float[][] trainingData = reader.get1dColorMatricesFromImages(batchSize * iterations, imageWidth);
        reader.setPreprocessParameters(trainingData);
        trainingData = reader.preprocessTrainingSet(trainingData);
        float[][] trainingDataOutputs = reader.oneHotOutputs;

        ClashDenseNetwork network = new ClashDenseNetwork(layers, learningRate, .0025f, ClashDenseNetwork.UPDATE_NESTEROV, momentum, dropoutProbability);

        long time = System.currentTimeMillis();

        float[] averageDerivatives = new float[network.layers];
        for (int p = 0; p < epochs; p++) {
            System.out.println("Epoch: " + p);
            for (int i = 0; i < trainingData.length; i += batchSize) {

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
//                System.out.println(network.derivativeOfWeightCheck(tempIn[0], tempOut[0], 1, 0, 0));
//                System.out.println(network.derivativesErrorWithRespectToWeights[1][0][0]);
                if(i % 10 == 0) {
                    System.out.println("Error at iteration " + i + ": " + network.testLoss(trainingData, trainingDataOutputs));
                }

                network.gradientDescent();
            }
        }

        for (int i = 1; i < network.layers; i++) {
            System.out.println("Average derivatives of layer " + i + ": " + averageDerivatives[i]);
        }

        System.out.println("It took " + (System.currentTimeMillis() - time) / 1000f + " seconds to train");


        System.out.println("Start bot?");
        Scanner scan = new Scanner(System.in);
        String string = scan.nextLine();
        System.out.println("Starting bot...");


        Robot bot = new Robot();
        int screenTopx = 694;
        int screenTopy = 48;
        int screenBottomx = 1223;
        int screenBottomy = 990;

        time = System.currentTimeMillis();
        while (System.currentTimeMillis() - time < 240000) {
            //eventually could make it output wait time
            //also could use card selected as input which may be necessary
            TimeUnit.SECONDS.sleep(4);
            float[] image = reader.resizeImage(bot.createScreenCapture(new Rectangle(screenTopx, screenTopy, screenBottomx - screenTopx, screenBottomy - screenTopy)), imageWidth);
            float[] out = network.predictOutput(image);
            int num = getGreatest(Arrays.copyOfRange(out, 0, 3));
            if(num == 1) {
                if(out[3] < 0 || out[3] > screenBottomx - screenTopx || out[4] < 0 || out[4] > screenBottomy - screenTopy) {

                }
                else {
                    mouseClick(screenTopx + (int) out[3], screenTopy + (int) out[4]);
                }
            }
            else if (num == 2) {
                if(getGreatest(Arrays.copyOfRange(out, 5, out.length)) == 0) {
                    mouseClick(857, 898);
                }
                else if(getGreatest(Arrays.copyOfRange(out, 5, out.length)) == 1) {
                    mouseClick(949, 898);
                }
                else if(getGreatest(Arrays.copyOfRange(out, 5, out.length)) == 2) {
                    mouseClick(1049, 898);
                }
                else if(getGreatest(Arrays.copyOfRange(out, 5, out.length)) == 3) {
                    mouseClick(1155, 898);
                }
            }
            else {
                // do nothing
            }
            System.out.println("");
            for(int i = 0; i < out.length; i++) {
                System.out.print(out[i] + ",");
            }
        }
    }

    public static int getGreatest(float[] in) {
        int greatest = 0;
        for(int i = 0; i < in.length; i++) {
            if(in[i] > in[greatest]) {
                greatest = i;
            }
        }
        return greatest;
    }

    public static void mouseClick(int x, int y) throws Exception{
        Robot robot = new Robot();
        robot.mouseMove(x, y);
        robot.mousePress(InputEvent.BUTTON1_MASK);
        TimeUnit.MILLISECONDS.sleep(20);
        robot.mouseRelease(InputEvent.BUTTON1_MASK);

    }

}
