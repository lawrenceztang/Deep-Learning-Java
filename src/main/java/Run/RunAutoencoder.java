package Run;

import Display.DisplayImage;
import Network.AutoEncoder;
import Network.DenseNetwork;
import Reader.ImageReader;
import Util.ArrOperations;

import java.util.Random;
import java.util.Scanner;

public class RunAutoencoder {

    public static void main(String[] args) throws Exception {
        Scanner scan = new Scanner(System.in);
        Random rand = new Random();

        String trainingDataPath = "C:\\Users\\Anonymous\\Pictures\\Numbers\\mnist_png\\training";
        ImageReader reader = new ImageReader(trainingDataPath);

        float[][] trainingData = reader.get1dColorMatricesFromImages(10000, 28);
        reader.setPreprocessParameters(trainingData);
        trainingData = reader.preprocessTrainingSet(trainingData);

        AutoEncoder encoder = new AutoEncoder(new int[]{2352, 10, 2352}, new float[]{0, .15f, 30f}, .9f);
        encoder.setNoise(.9f);

        for(int i = 0; i < 10000; i++) {
            encoder.getDerivativeOfErrorWithRespectToWeights(new float[][]{trainingData[i]});
            encoder.gradientDescent();
            if(i % 1000 == 0) {
                System.out.println(encoder.testLoss(trainingData, trainingData));
            }
        }

        while(true) {
            System.out.println("Test? (y/n)");
            if(!scan.nextLine().equals("n")) {
                int n = rand.nextInt(trainingData.length);
                new DisplayImage(ArrOperations.convertArrayToImage(reader.unpreproccessExample(encoder.predictOutput(encoder.addNoise(trainingData[n]))), 28, 28));
                new DisplayImage(ArrOperations.convertArrayToImage(reader.unpreproccessExample(encoder.addNoise(trainingData[n])), 28, 28));
            }
            else {
                System.out.println("Terminating program. ");
                break;
            }
        }

    }

}
