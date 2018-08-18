import Network.DenseNetwork;
import Reader.ImageReader;
import org.junit.Assert;
import org.junit.Test;

import java.awt.*;
import java.text.DecimalFormat;

public class FullyConnectedTest {
    String path = "C:\\Users\\Anonymous\\Pictures\\Numbers\\mnist_png\\testing";
    float[][] trainingData;
    float[][] trainingDataOutputs;
    DenseNetwork network;

    @Test
    public void testDerivatives() throws Exception{
        DecimalFormat df = new DecimalFormat("#.####");

        System.out.println(System.getenv());


        ImageReader reader = new ImageReader(path);
        trainingData = reader.get1dColorMatricesFromImages(100, 28);
        reader.setPreprocessParameters(trainingData);
        trainingData = reader.preprocessTrainingSet(trainingData);
        trainingDataOutputs = reader.oneHotOutputs;


        network = new DenseNetwork(new int[]{2352, 20, 10}, new float[]{0, .05f, .0025f}, DenseNetwork.UPDATE_SGD, .9f, 1);

        int batchSize = 100;
        for (int p = 0; p < 1; p++) {
            for (int i = 0; i < trainingData.length; i += batchSize) {

                float[][] tempIn = new float[batchSize][];
                float[][] tempOut = new float[batchSize][];
                for (int a = 0; a < batchSize; a++) {
                    tempIn[a] = trainingData[i];
                    tempOut[a] = trainingDataOutputs[i];
                }

                network.getDerivativeOfErrorWithRespectToWeights(tempIn, tempOut);

                boolean pass = true;
                for(int a = 1; a < network.derivativesErrorWithRespectToWeights.length; a++) {
                    for(int q = 0; q < network.derivativesErrorWithRespectToWeights[a].length; q++) {
                        for(int v = 0; v < network.derivativesErrorWithRespectToWeights[a][q].length; v++) {
                            if(Float.parseFloat(df.format(new Float(network.derivativesErrorWithRespectToWeights[a][q][v]))) != Float.parseFloat(df.format(new Float(network.derivativeOfWeightCheck(tempIn, tempOut, a, q, v))))) {
                                pass = false;
                                System.out.println("Test Failed!");
                                System.out.println(network.derivativesErrorWithRespectToWeights[a][q][v]);
                                System.out.println(network.derivativeOfWeightCheck(tempIn, tempOut, a, q, v));
                            }
                            else {

                            }
                        }
                    }
                }
                Assert.assertEquals(pass, true);
            }
        }
    }
}
