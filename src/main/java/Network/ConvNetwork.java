package Network;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Random;

import Util.ArrOperations;

public class ConvNetwork {

    Random rand;

    //constants
    int[] inputSize;

    //hyperparameters
    float[] convLearningRates;
    int[] filterSizes;
    int[] strideSizes;
    int[] numberOfFilters;
    int[][] padding;
    int[] poolingStride;
    int numLayers;


    //weights: numlayers x numberFilters x inputDepth x filtersize x filtersize

    float[][][][][] weights;
    float[][] biases;
    float[][][][] outputsInLayers;
    float[][][][] derivativeErrorWithRespectToInputToActivation;
    public float[][][][][] derivativeErrorWithRespectToWeight;


    public DenseNetwork denseNetwork;

    //no max pooling in final layer
    public ConvNetwork(int[] inputSize, float[] convLearningRates, int[] filterSizes, int[] strideSizes, int[] numberOfFilters, int[][] padding, int[] poolingStride, DenseNetwork denseNetwork) {

        rand = new Random();

        this.inputSize = inputSize;

        this.convLearningRates = convLearningRates;
        this.filterSizes = filterSizes;
        this.strideSizes = strideSizes;
//        this.numInputs = numInputs;
//        this.numOutputs = numOutputs;
        this.numberOfFilters = numberOfFilters;
        this.padding = padding;
        this.poolingStride = poolingStride;

        this.denseNetwork = denseNetwork;

        initializeWeights();
        derivativeErrorWithRespectToInputToActivation = new float[filterSizes.length][][][];
        numLayers = filterSizes.length;

        numberOfFilters[0] = 3;
    }

    public void gradientDescent() {
        for (int i = 1; i < derivativeErrorWithRespectToWeight.length; i++) {
            for (int n = 0; n < derivativeErrorWithRespectToWeight[i].length; n++) {
                for (int z = 0; z < derivativeErrorWithRespectToWeight[i][n].length; z++) {
                    for (int s = 0; s < derivativeErrorWithRespectToWeight[i][n][z].length; s++) {
                        for (int f = 0; f < derivativeErrorWithRespectToWeight[i][n][z][s].length; f++) {
                            weights[i][n][z][s][f] -= derivativeErrorWithRespectToWeight[i][n][z][s][f] * convLearningRates[i];
                        }
                    }
                }
            }
        }
    }

    //i think backprop is correctly done
    public void getDerivativeOfWeights(float[][][][] inputs, float[][] outputs) throws Exception {

        forwardPass(inputs[0]);
        derivativeErrorWithRespectToInputToActivation = initializeActivationInputs();

        float[][] inputToFullyConnected = new float[1][];
        inputToFullyConnected[0] = ArrOperations.convert3Dto1D(outputsInLayers[outputsInLayers.length - 1]);

        denseNetwork.getDerivativeOfErrorWithRespectToWeights(inputToFullyConnected, outputs);

        float[] derivativeErrorWithRespectToInputsToFullyConnected = new float[denseNetwork.nodesPerLayer[0]];
        initializeWeightDerivatives();

        for (int u = 0; u < denseNetwork.numInputs; u++) {
            for (int i = 0; i < denseNetwork.derivativesErrorWithRespectToInputsToActivation[1].length; i++) {
                derivativeErrorWithRespectToInputsToFullyConnected[u] += denseNetwork.derivativesErrorWithRespectToInputsToActivation[1][i] * denseNetwork.weights[1][i][u];
            }
        }

        int last = derivativeErrorWithRespectToInputToActivation.length - 1;

        //must be put through activation
        derivativeErrorWithRespectToInputToActivation[last] = ArrOperations.matrixMatrixProduct(ArrOperations.convert1Dto3D(derivativeErrorWithRespectToInputsToFullyConnected), ArrOperations.getDerivativeFromSigmoid(outputsInLayers[last]));
        float[][][] outputAfterPoolings = ArrOperations.pad(maxPooling(outputsInLayers[last - 1], poolingStride[last - 1]), padding[last]);

        for (int b = 0; b < derivativeErrorWithRespectToInputToActivation[last].length; b++) {
            for (int h = 0; h < derivativeErrorWithRespectToInputToActivation[last][b].length; h++) {
                for (int z = 0; z < derivativeErrorWithRespectToInputToActivation[last][b][h].length; z++) {

                    for (int x = 0; x < numberOfFilters[last - 1]; x++) {
                        for (int w = 0; w < filterSizes[last]; w++) {
                            for (int y = 0; y < filterSizes[last]; y++) {
                                //inputtoactivation
                                derivativeErrorWithRespectToWeight[last][b][x][w][y] += derivativeErrorWithRespectToInputToActivation[last][b][h][z] * outputAfterPoolings[x][h * strideSizes[last] + w][z * strideSizes[last] + y];
                            }
                        }
                    }
                    biases[last][b] = biases[last][b] - derivativeErrorWithRespectToInputToActivation[last][b][h][z];
                }
            }
        }

        for (int a = numLayers - 2; a > 0; a--) {

            //undo convolution
            for (int b = 0; b < numberOfFilters[a + 1]; b++) {
                for (int x = 0; x < outputsInLayers[a].length; x++) {
                    for (int h = 0; h < derivativeErrorWithRespectToInputToActivation[a + 1][b].length; h++) {
                        for (int q = 0; q < derivativeErrorWithRespectToInputToActivation[a + 1][b][h].length; q++) {


                            for (int w = 0; w < filterSizes[a + 1]; w++) {
                                for (int y = 0; y < filterSizes[a + 1]; y++) {
                                    derivativeErrorWithRespectToInputToActivation[a][x][h * strideSizes[a + 1] + w][q * strideSizes[a + 1] + y] += derivativeErrorWithRespectToInputToActivation[a + 1][b][h][q] * weights[a + 1][b][x][w][y];
                                }
                            }
                        }
                    }
                }
            }

            //undo padding
            derivativeErrorWithRespectToInputToActivation[a] = ArrOperations.unpad(derivativeErrorWithRespectToInputToActivation[a], padding[a + 1]);

            //undo pooling
            derivativeErrorWithRespectToInputToActivation[a] = getDerivativeFromMaxPooling(derivativeErrorWithRespectToInputToActivation[a], outputsInLayers[a]);

            //undo sigmoid
            derivativeErrorWithRespectToInputToActivation[a] = ArrOperations.matrixMatrixProduct(derivativeErrorWithRespectToInputToActivation[a], ArrOperations.getDerivativeFromSigmoid(outputsInLayers[a]));

            float[][][] outputAfterPooling = ArrOperations.pad(maxPooling(outputsInLayers[a - 1], poolingStride[a - 1]), padding[a]);

            for (int b = 0; b < derivativeErrorWithRespectToInputToActivation[a].length; b++) {
                for (int h = 0; h < derivativeErrorWithRespectToInputToActivation[a][b].length; h++) {
                    for (int z = 0; z < derivativeErrorWithRespectToInputToActivation[a][b][h].length; z++) {

                        for (int x = 0; x < numberOfFilters[a - 1]; x++) {
                            for (int w = 0; w < filterSizes[a]; w++) {
                                for (int y = 0; y < filterSizes[a]; y++) {
                                    //inputtoactivation
                                    derivativeErrorWithRespectToWeight[a][b][x][w][y] += derivativeErrorWithRespectToInputToActivation[a][b][h][z] * outputAfterPooling[x][h * strideSizes[a] + w][z * strideSizes[a] + y];
                                }
                            }
                        }
                        biases[a][b] -= derivativeErrorWithRespectToInputToActivation[a][b][h][z];
                    }
                }
            }

        }
    }

    //working
    public float[] predictOutput(float[][][] inputs) throws Exception {
        float[][][] layerInputs = inputs;
        for (int e = 1; e < numLayers; e++) {
            layerInputs = ArrOperations.pad(layerInputs, padding[e]);
            float[][][] layerOutputs = initializeOutputs(numberOfFilters[e], (layerInputs[0].length - filterSizes[e]) / strideSizes[e] + 1, (layerInputs[0][0].length - filterSizes[e]) / strideSizes[e] + 1);
            for (int i = 0; i <= layerInputs[0].length - filterSizes[e]; i += strideSizes[e]) {
                for (int b = 0; b <= layerInputs[0][i].length - filterSizes[e]; b += strideSizes[e]) {


                    float[][][] inputToNeuron = new float[layerInputs.length][][];

                    for (int v = 0; v < layerInputs.length; v++) {
                        inputToNeuron[v] = new float[filterSizes[e]][];
                        for (int u = 0; u < filterSizes[e]; u++) {
                            inputToNeuron[v][u] = new float[filterSizes[e]];
                            for (int t = 0; t < filterSizes[e]; t++) {

                                inputToNeuron[v][u][t] = layerInputs[v][i + u][b + t];
                            }
                        }
                    }
                    for (int l = 0; l < weights[e].length; l++) {
                        layerOutputs[l][i / strideSizes[e]][b / strideSizes[e]] += ArrOperations.sigmoidFunction(ArrOperations.matrixProductSum(inputToNeuron, weights[e][l]));
                    }
                }
            }
            layerInputs = maxPooling(layerOutputs, poolingStride[e]);
        }

        float[] inputToFullyConnected = ArrOperations.convert3Dto1D(layerInputs);
        return denseNetwork.predictOutput(inputToFullyConnected);
    }

    //input layer is counted as layer 0
    public void forwardPass(float[][][] inputs) throws Exception {
        outputsInLayers = new float[numLayers][][][];

        float[][][] layerInputs = inputs;
        outputsInLayers[0] = layerInputs;
        for (int e = 1; e < numLayers; e++) {
            layerInputs = ArrOperations.pad(layerInputs, padding[e]);
            float[][][] layerOutputs = initializeOutputs(numberOfFilters[e], (layerInputs[0].length - filterSizes[e]) / strideSizes[e] + 1, (layerInputs[0][0].length - filterSizes[e]) / strideSizes[e] + 1);
            for (int i = 0; i <= layerInputs[0].length - filterSizes[e]; i += strideSizes[e]) {
                for (int b = 0; b <= layerInputs[0][i].length - filterSizes[e]; b += strideSizes[e]) {


                    float[][][] inputToNeuron = new float[layerInputs.length][][];

                    for (int v = 0; v < layerInputs.length; v++) {
                        inputToNeuron[v] = new float[filterSizes[e]][];
                        for (int u = 0; u < filterSizes[e]; u++) {
                            inputToNeuron[v][u] = new float[filterSizes[e]];
                            for (int t = 0; t < filterSizes[e]; t++) {
                                inputToNeuron[v][u][t] = layerInputs[v][i + u][b + t];
                            }
                        }
                    }
                    for (int l = 0; l < numberOfFilters[e]; l++) {
                        layerOutputs[l][i / strideSizes[e]][b / strideSizes[e]] += ArrOperations.sigmoidFunction(ArrOperations.matrixProductSum(inputToNeuron, weights[e][l]));
                    }
                }
            }
            outputsInLayers[e] = layerOutputs;

            layerInputs = maxPooling(outputsInLayers[e], poolingStride[e]);
        }

    }


    public void initializeWeights() {
        weights = new float[filterSizes.length][][][][];
        biases = new float[filterSizes.length][];

        for (int i = 1; i < filterSizes.length; i++) {
            weights[i] = new float[numberOfFilters[i]][][][];
            biases[i] = new float[numberOfFilters[i]];
            for (int u = 0; u < numberOfFilters[i]; u++) {

                int inputDepth;
                if (i == 1) {
                    inputDepth = inputSize[0];
                } else {
                    inputDepth = numberOfFilters[i - 1];
                }
                weights[i][u] = new float[inputDepth][][];
                biases[i][u] = ArrOperations.gaussianRandomVariable(.25f, 0);
                for (int q = 0; q < inputDepth; q++) {
                    weights[i][u][q] = new float[filterSizes[i]][];
                    for (int v = 0; v < filterSizes[i]; v++) {
                        weights[i][u][q][v] = new float[filterSizes[i]];
                        for (int c = 0; c < filterSizes[i]; c++) {
                            weights[i][u][q][v][c] = ArrOperations.gaussianRandomVariable(.25f, 0);
                        }
                    }
                }
            }
        }
        return;
    }

    public void initializeWeightDerivatives() {
        derivativeErrorWithRespectToWeight = new float[filterSizes.length][][][][];
        for (int i = 1; i < filterSizes.length; i++) {
            derivativeErrorWithRespectToWeight[i] = new float[numberOfFilters[i]][][][];
            for (int u = 0; u < numberOfFilters[i]; u++) {

                int inputDepth;
                if (i == 1) {
                    inputDepth = inputSize[0];
                } else {
                    inputDepth = numberOfFilters[i - 1];
                }
                derivativeErrorWithRespectToWeight[i][u] = new float[inputDepth][][];
                for (int q = 0; q < inputDepth; q++) {
                    derivativeErrorWithRespectToWeight[i][u][q] = new float[filterSizes[i]][];
                    for (int v = 0; v < filterSizes[i]; v++) {
                        derivativeErrorWithRespectToWeight[i][u][q][v] = new float[filterSizes[i]];
                        for (int c = 0; c < filterSizes[i]; c++) {
                            derivativeErrorWithRespectToWeight[i][u][q][v][c] = 0;
                        }
                    }
                }
            }
        }
        return;
    }

    public float[][][] initializeOutputs(int numFilters, int width, int length) {
        float[][][] outputs = new float[numFilters][][];
        for (int i = 0; i < numFilters; i++) {
            outputs[i] = new float[width][];
            for (int d = 0; d < width; d++) {
                outputs[i][d] = new float[length];
                for (int f = 0; f < length; f++) {
                    outputs[i][d][f] = 0f;
                }
            }
        }
        return outputs;
    }

    public float[][][][] initializeActivationInputs() {

        float[][][][] out = new float[outputsInLayers.length][][][];
        for (int i = 0; i < outputsInLayers.length - 1; i++) {
            out[i] = new float[outputsInLayers[i].length][][];
            for (int u = 0; u < outputsInLayers[i].length; u++) {
                out[i][u] = new float[outputsInLayers[i][u].length / poolingStride[i] + padding[i + 1][0]][];
                for (int y = 0; y < out[i][u].length; y++) {
                    out[i][u][y] = new float[outputsInLayers[i][u][0].length / poolingStride[i] + padding[i + 1][1]];
                    for (int a = 0; a < out[i][u][y].length; a++) {
                        out[i][u][y][a] = 0f;
                    }
                }
            }
        }
        return out;
    }

    public void setPadding() {
        int[] inputSizes = new int[2];
        inputSizes[0] = inputSize[1];
        inputSizes[1] = inputSize[2];
        for (int i = 1; i < filterSizes.length; i++) {
            for (int p = 0; p < 2; p++) {
                if ((inputSizes[p] - filterSizes[i]) % strideSizes[i] != 0) {
                    padding[i][p] = (int) Math.ceil((double) (inputSizes[p] - filterSizes[i]) / (double) strideSizes[i]) * strideSizes[i] - (inputSizes[p] - filterSizes[i]);
                }
                inputSizes[p] = (inputSizes[p] - filterSizes[i]) / strideSizes[i] + 1;
            }
        }
    }

    public float[][][] maxPooling(float[][][] in, int stride) {
        if (stride == 1) {
            return in;
        }
        float[][][] out = new float[in.length][][];
        for (int u = 0; u < in.length; u++) {
            out[u] = new float[in[u].length][];

            for (int a = 0; a < in[u].length; a += stride) {
                out[u][a] = new float[in[u][a].length];
                for (int q = 0; q < in[u][a].length; q += stride) {

                    float max = -9999;
                    for (int p = 0; p < stride; p++) {
                        for (int y = 0; y < stride; y++) {
                            if (in[u][a + p][q + y] > max) {
                                max = in[u][a + p][q + y];
                            }
                        }
                    }

                    out[u][a][q] = max;

                }
            }
        }
        return out;
    }

    //strides should not overlap
    public float[][][] getDerivativeFromMaxPooling(float[][][] derivativeErrorWithRespectToPoolingOutput, float[][][] poolingInput) {

        if (poolingInput[0].length == derivativeErrorWithRespectToPoolingOutput[0].length) {
            return derivativeErrorWithRespectToPoolingOutput;
        }

        float[][][] derivativeErrorWithRespectToInputToPooling = new float[poolingInput.length][][];
        int stride = derivativeErrorWithRespectToPoolingOutput[0].length / poolingInput[0].length;
        for (int i = 0; i < derivativeErrorWithRespectToPoolingOutput.length; i++) {
            derivativeErrorWithRespectToInputToPooling[i] = new float[poolingInput[i].length][];
            for (int a = 0; a < derivativeErrorWithRespectToPoolingOutput[i].length; a++) {
                for (int o = 0; o < derivativeErrorWithRespectToPoolingOutput[i][a].length; o++) {

                    float max = -99999;
                    for (int c = 0; c < stride; c++) {
                        for (int u = 0; u < stride; u++) {
                            if (poolingInput[i][a * stride + c][o * stride + u] > max) {
                                max = poolingInput[i][a * stride + c][o * stride + u];
                            }
                        }
                    }

                    for (int y = 0; y < stride; y++) {
                        derivativeErrorWithRespectToInputToPooling[i][y + a * stride] = new float[poolingInput[i][y + a * stride].length];
                        for (int q = 0; q < stride; q++) {
                            if (max == poolingInput[i][y + a * stride][o * stride + q]) {
                                derivativeErrorWithRespectToInputToPooling[i][y + a * stride][q + o * stride] = derivativeErrorWithRespectToPoolingOutput[i][a][o];
                            } else {
                                derivativeErrorWithRespectToInputToPooling[i][y + a * stride][q + o * stride] = 0f;
                            }
                        }
                    }

                }
            }
        }
        return derivativeErrorWithRespectToInputToPooling;
    }

    //tests on dataset, returns percentage accurate
    public float test(float[][][][] in, float[][] out) throws Exception {
        int numCorrect = 0;
        for (int i = 0; i < in.length; i++) {
            float[] array = predictOutput(in[i]);
            int prediction = 0;

            for (int p = 0; p < array.length; p++) {
                if (array[p] > array[prediction]) {
                    prediction = p;
                }
            }

            int output = 0;
            for (int p = 0; p < array.length; p++) {
                if (out[i][p] > out[i][output]) {
                    output = p;
                }
            }

            if (prediction == output) {
                numCorrect++;
            }
        }
        return numCorrect / (float) in.length;
    }

    int aPoolWidth;
    int aPoolLength;

    public float[] averagePooling(float[][][] in) {
        float[] out = new float[in.length];
        for (int p = 0; p < in.length; p++) {
            for (int c = 0; c < in[p].length; c++) {
                for (int d = 0; d < in[p][c].length; d++) {
                    out[p] += in[p][c][d];
                }
            }
            out[p] = out[p] / (float) in[p].length / (float) in[p][0].length;
        }
        aPoolWidth = in[0].length;
        aPoolLength = in[0][0].length;
        return out;
    }

    public float[][][] getDerivativeFromAveragePooling(float[] in) {
        float[][][] out = new float[in.length][][];
        for (int d = 0; d < in.length; d++) {
            out[d] = new float[aPoolWidth][];
            for (int i = 0; i < aPoolWidth; i++) {
                out[d][i] = new float[aPoolLength];
                for (int u = 0; u < aPoolLength; u++) {
                    out[d][i][u] = in[d] / aPoolLength / aPoolWidth;
                }
            }
        }
        return out;
    }

    public float derivativeCheck(float[][][] in, float[] out, int layer, int filter, int z, int x, int y) throws Exception {
        float intervals = .0001f;
        weights[layer][filter][z][x][y] -= intervals;
        float losss = ArrOperations.meanSquaredError(predictOutput(in), out);
        weights[layer][filter][z][x][y] += intervals * 2;
        float output = (ArrOperations.meanSquaredError(predictOutput(in), out) - losss) / intervals / 2;
        weights[layer][filter][z][x][y] -= intervals;

        return output;
    }

    public BufferedImage visualizeWeights(int layer, int filterNum, int z) {
        return ArrOperations.convertArrayToImage(weights[layer][filterNum][z]);
    }

    public class DeepDream {
        public float[][][] image;
        float[] derivativeErrorWithRespectToInputs;

        float learningRate = 1;
        float max = -9999;
        float min = 9999;

        public DeepDream(float[][][] image, float learningRate) {

            this.image = ArrOperations.makeCopy(image);
            this.learningRate = learningRate;


            for (int i = 0; i < image.length; i++) {
                for(int d = 0; d < image[i].length; d++) {
                    for(int u = 0; u < image[i][d].length; u++) {
                        if (image[i][d][u] > max) {
                            max = image[i][d][u];
                        }
                        if (image[i][d][u] < min) {
                            min = image[i][d][u];
                        }
                    }
                }
            }
        }

        public void updateImage(int desiredOutput) throws Exception {

            derivativeErrorWithRespectToInputs = new float[image.length];
            for (int i = 0; i < image.length; i++) {
                derivativeErrorWithRespectToInputs[i] = 0f;
            }

            float[] outputs = new float[ConvNetwork.this.denseNetwork.numOutputs];
            for (int i = 0; i < outputs.length; i++) {
                if (i == desiredOutput) {
                    outputs[i] = 1f;
                } else {
                    outputs[i] = 0f;
                }
            }
            float[][][][] inputsToNetwork = new float[1][][][];
            inputsToNetwork[0] = image;

            float[][] outputsOfNetwork = new float[1][];
            outputsOfNetwork[0] = outputs;


            ConvNetwork.this.getDerivativeOfWeights(inputsToNetwork, outputsOfNetwork);

            for (int b = 0; b < numberOfFilters[1]; b++) {
                for (int x = 0; x < outputsInLayers[0].length; x++) {
                    for (int h = 0; h < derivativeErrorWithRespectToInputToActivation[1][b].length; h++) {
                        for (int q = 0; q < derivativeErrorWithRespectToInputToActivation[1][b][h].length; q++) {

                            for (int w = 0; w < filterSizes[1]; w++) {
                                for (int y = 0; y < filterSizes[1]; y++) {
                                    image[x][h * strideSizes[1] + w][q * strideSizes[1] + y] -= derivativeErrorWithRespectToInputToActivation[1][b][h][q] * weights[1][b][x][w][y] * learningRate;
                                }
                            }

                        }
                    }
                }
            }

            image = ArrOperations.keepInRange(image, 0, 256);

        }
    }


}
