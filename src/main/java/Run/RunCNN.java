package Run;

import Network.ConvNetwork;
import Network.DenseNetwork;
import Reader.ImageReader;

public class RunCNN {

    public static void main(String[] args) throws Exception {
        int layers = 3;

        int[] filterSizes = new int[]{0, 3, 3};
        int[] strideSizes = new int[]{0, 1, 1};
        int[] numFilters = new int[]{0, 32, 64};
        int[][] padding = new int[][]{new int[]{0, 0}, new int[]{0, 0}, new int[]{0, 0}};
        int[] poolingStride = new int[]{1, 1, 1};

        //must implement auto input size calculation, right now user has to calculate it out
        int[] nodesPerLayer = new int[]{36864, 200, 10};
        float[] learningRate = new float[]{0, .5f, .025f};
        DenseNetwork fNetwork = new DenseNetwork(nodesPerLayer, learningRate, DenseNetwork.UPDATE_MOMENTUM, .9f, 1);

        ConvNetwork network = new ConvNetwork(new int[]{3, 28, 28}, new float[]{0f, .5f, .5f}, filterSizes, strideSizes, numFilters, padding, poolingStride, fNetwork);
        network.initializeWeights();
        network.setPadding();

        String trainingDataPath = "C:\\Users\\Anonymous\\Pictures\\Numbers\\mnist_png\\training";
        float[][][][] trainingData;
        ImageReader reader = new ImageReader(trainingDataPath);

        trainingData = reader.get3dColorMatrices(1000);
        reader.setPreprocessParameters(trainingData);
        reader.preprocessTrainingSet(trainingData);
        float[][] trainingDataOutputs = reader.oneHotOutputs;

        System.out.println(network.test(trainingData, trainingDataOutputs));

        int batchSize = 1;
        for (int u = 0; u < 1; u++) {
            for (int i = 0; i < trainingData.length; i += batchSize) {
                float[][][][] inputs = new float[batchSize][][][];
                float[][] outputs = new float[batchSize][];
                for (int q = 0; q < batchSize; q++) {
                    inputs[q] = trainingData[i + q];
                    outputs[q] = trainingDataOutputs[i + q];
                }
                network.getGradientsWeightsWithRespectToError(inputs, outputs);

//                System.out.println("Derivatives: ");
//                System.out.println(network.derivativeCheck(inputs[0], outputs[0], 1, 0, 0, 0, 0));
//                System.out.println(network.derivativeErrorWithRespectToWeight[1][0][0][0][0]);

                network.gradientDescent();
                network.denseNetwork.gradientDescent();
            }
        }

        System.out.println(network.test(trainingData, trainingDataOutputs));
    }


}
