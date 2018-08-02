import Network.ConvNetwork;
import Network.DenseNetwork;
import Reader.ImageReader;

public class CNNTest {
    public static void main(String[] args) throws Exception{

        int[] filterSizes = new int[]{0, 1};
        int[] strideSizes = new int[]{0, 1};
        int[] numFilters = new int[]{0, 1};
        int[][] padding = new int[][]{new int[]{0, 0}, new int[]{0, 0}};
        int[] poolingStride = new int[]{1, 1};

        //must implement auto input size calculation, right now user has to calculate it out
        int[] nodesPerLayer = new int[]{4, 1, 2};
        float[] learningRate = new float[]{0, 0f, 0f};
        DenseNetwork fNetwork = new DenseNetwork(nodesPerLayer, learningRate, DenseNetwork.UPDATE_MOMENTUM, .9f, 1);

        ConvNetwork network = new ConvNetwork(new int[]{3, 2, 2}, new float[]{0f, .5f}, filterSizes, strideSizes, numFilters, padding, poolingStride, fNetwork);
        network.initializeWeights();
        network.setPadding();

        float[][][][] trainingData = new float[][][][]{{ { {1, 2}, {3, 4} }, { {1, 2}, {3, 4} }, { {1, 2}, {3, 4} } }};

        float[][] trainingDataOutputs = new float[][]{{2, 3}};

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

                System.out.println("Derivatives: ");
                System.out.println(network.derivativeCheck(inputs[0], outputs[0], 1, 0, 0, 0, 0));
                System.out.println(network.derivativeErrorWithRespectToWeight[1][0][0][0][0]);

                network.gradientDescent();
                network.denseNetwork.gradientDescent();
            }
        }

        System.out.println(network.test(trainingData, trainingDataOutputs));
    }
}
