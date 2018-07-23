import Network.DenseNetwork;
import Reader.ImageReader;
import org.junit.Assert;
import org.junit.Test;

import java.awt.*;
import java.text.DecimalFormat;

public class FullyConnectedTest {
    String path;
    float[][] trainingData;
    float[][] trainingDataOutputs;
    DenseNetwork network;

    @Test
    public void testDerivatives() throws Exception{
        DecimalFormat df = new DecimalFormat("#.####");

        System.out.println(System.getenv());


        ImageReader reader = new ImageReader(path);
        trainingData = reader.get1dColorMatricesFromImages(1, 28);
        reader.setPreprocessParameters(trainingData);
        trainingData = reader.preprocessTrainingSet(trainingData);
        trainingDataOutputs = reader.oneHotOutputs;


        network = new DenseNetwork(new int[]{2352, 20, 10}, new float[]{0, .05f, .0025f}, .9f);

        network.classNames = reader.classes;

        int batchSize = 1;
        for (int p = 0; p < 1; p++) {
            for (int i = 0; i < trainingData.length; i += batchSize) {

                float[][] tempIn = new float[batchSize][];
                float[][] tempOut = new float[batchSize][];
                for (int a = 0; a < batchSize; a++) {
                    tempIn[a] = trainingData[i];
                    tempOut[a] = trainingDataOutputs[i];
                }

                network.setDropout(1);
                network.getDerivativeOfErrorWithRespectToWeights(tempIn, tempOut);

                boolean pass = true;
                for(int a = 1; a < network.derivativesErrorWithRespectToWeights.length; a++) {
                    for(int q = 0; q < network.derivativesErrorWithRespectToWeights[a].length; q++) {
                        for(int v = 0; v < network.derivativesErrorWithRespectToWeights[a][q].length; v++) {
                            if(Float.parseFloat(df.format(new Float(network.derivativesErrorWithRespectToWeights[a][q][v] / 10))) != Float.parseFloat(df.format(new Float(network.derivativeOfWeightCheck(tempIn[0], tempOut[0], a, q, v))))) {
                                pass = false;
                            }
                        }
                    }
                }
                Assert.assertEquals(pass, true);
            }
        }
    }
}
