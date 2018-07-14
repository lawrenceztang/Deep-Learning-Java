package Run;

import Network.ConvNetwork;
import Network.FullyConnectedNetwork;
import Util.ImageReader;

import java.util.ArrayList;

public class RunCNN {

    public static void main(String[] args)throws Exception {

        ArrayList<Integer> filterSizes = new ArrayList<Integer>();
        filterSizes.add(0);
        filterSizes.add(4);
        filterSizes.add(4);

        ArrayList<Integer> strideSizes = new ArrayList<Integer>();
        strideSizes.add(0);
        strideSizes.add(2);
        strideSizes.add(2);

        ArrayList<Integer> numFilters = new ArrayList<Integer>();
        numFilters.add(0);
        numFilters.add(4);
        numFilters.add(8);

        ArrayList<Integer> padding = new ArrayList<Integer>();
        padding.add(0);
        padding.add(0);
        padding.add(0);

        ArrayList<Integer> poolingStride = new ArrayList<Integer>();
        poolingStride.add(0);
        poolingStride.add(0);
        poolingStride.add(0);

        //must implement auto input size calculation, right now user has to calculate it out
        int[] nodesPerLayer = new int[]{200, 200, 10};
        double[] learningRate = new double[]{.05, .05};
        FullyConnectedNetwork fNetwork = new FullyConnectedNetwork(nodesPerLayer, learningRate, .9);

        ConvNetwork network = new ConvNetwork(new int[]{3, 28, 28}, new double[]{.05, .05}, filterSizes, strideSizes, numFilters, padding, poolingStride, fNetwork);
        network.initializeWeights();

        String trainingDataPath = "C:\\Users\\Anonymous\\Pictures\\Numbers\\mnist_png\\training";
        ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> trainingData = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();
        ImageReader reader = new ImageReader(trainingDataPath);

        trainingData = reader.get2dColorMatrices(100);
//        reader.setPreprocessParameters(trainingData);
//        trainingData = reader.preprocessTrainingSet(trainingData);
        ArrayList<ArrayList<Double>> trainingDataOutputs = reader.oneHotOutputs;

        System.out.println(network.predictOutput(trainingData.get(0)));

        int batchSize = 1;
        for(int u = 0; u < 1; u++) {
            for (int i = 0; i < trainingData.size(); i += batchSize) {
                ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> inputs = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();
                ArrayList<ArrayList<Double>> outputs = new ArrayList<ArrayList<Double>>();
                for(int q = 0; q < batchSize; q++) {
                    inputs.add(trainingData.get(i + q));
                    outputs.add(trainingDataOutputs.get(i + q));
                }
                network.getGradientsWeightsWithRespectToError(inputs, outputs);
                network.gradientDescent();
            }
        }

        System.out.println(network.predictOutput(trainingData.get(0)));
    }


}
