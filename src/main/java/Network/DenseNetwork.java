package Network;

import java.io.Serializable;
import java.util.Random;

import Util.*;


public class DenseNetwork implements Serializable {
    Random rand;

    //constants
    public final static int UPDATE_SGD = 0;
    public final static int UPDATE_MOMENTUM = 1;
    public final static int UPDATE_NESTEROV = 2;
    public final static int UPDATE_RMSPROP = 3;

    final static float WEIGHT_DROPPED_OUT = 0;


    //hyperparameters
    public int layers;
    public int[] nodesPerLayer;
    public int numInputs;
    public int numOutputs;
    public float learningRate[];
    public float momentum;
    float dropoutProbability = 1;
    int updateRule = 0;


    //learned parameters
    float[][][] weights;
    float[][] biases;
    float[][] derivativesErrorWithRespectToBiases;
    public float[][][] derivativesErrorWithRespectToWeights;
    float[][] outputsInAllLayers;
    float[][][] weightMomentumUpdate;
    float[][] derivativesErrorWithRespectToInputsToActivation;
    float[][][] weightsAfterDropout;

    public DenseNetwork(int[] nodesPerLayer, float[] learningRate, int updateRule, float momentum, float dropoutProbability) {
        rand = new Random();
        outputsInAllLayers = new float[nodesPerLayer.length][];
        weightsAfterDropout = new float[nodesPerLayer.length][][];

        //layers includes output and input layer
        this.layers = nodesPerLayer.length;
        this.nodesPerLayer = nodesPerLayer;
        this.numInputs = nodesPerLayer[0];
        this.numOutputs = nodesPerLayer[nodesPerLayer.length - 1];
        this.learningRate = learningRate;
        this.updateRule = updateRule;
        this.momentum = momentum;
        this.dropoutProbability = dropoutProbability;

        setWeights();
    }

    public void gradientDescent() {

        for (int i = 1; i < weights.length; i++) {
            int nodesPreviousLayer = nodesPerLayer[i - 1];
            for (int t = 0; t < weights[i].length; t++) {
                for (int y = 0; y < weights[i][t].length; y++) {
                    if (weightsAfterDropout[i][t][y] != WEIGHT_DROPPED_OUT) {
                        weightMomentumUpdate[i][t][y] = weightMomentumUpdate[i][t][y] * momentum + (1 - momentum) * learningRate[i] * derivativesErrorWithRespectToWeights[i][t][y];
                        weights[i][t][y] = weights[i][t][y] - weightMomentumUpdate[i][t][y];
                    }

                    biases[i][t] = biases[i][t] - derivativesErrorWithRespectToBiases[i][t] * learningRate[i] * .01f;
                }
            }
        }

        //ensure that if dropout isnt set weightsAfterDropout is still updated - weightsAfterDropout is used for forwardPass
        weightsAfterDropout = weights;
    }

    //todo: gradients are 10 times larger than actually in last layer
    public void getDerivativeOfErrorWithRespectToWeights(float[][] inputs, float[][] outputs) throws Exception {
        //no need to worry about dropped out weights because they automatically have 0 derivative so they arent updated in gradient descent
        //try mean squared error
        setDropout();
        setUpdateRule();

        derivativesErrorWithRespectToWeights = getNewDerivativeWeights();
        derivativesErrorWithRespectToBiases = getNewDerivativeBiasesAndOutputs();
        for (int w = 0; w < inputs.length; w++) {
            derivativesErrorWithRespectToInputsToActivation = getNewDerivativeBiasesAndOutputs();
            forwardPass(inputs[w]);
            for (int p = layers - 1; p > 0; p--) {
                if (p == layers - 1) {
                    derivativesErrorWithRespectToInputsToActivation[p] = ArrOperations.getDerivativeFromSoftmax(outputsInAllLayers[p], ArrOperations.getDerivativeFromMSE(outputs[w], outputsInAllLayers[layers - 1]));
                    for (int i = 0; i < numOutputs; i++) {
                        for (int u = 0; u < nodesPerLayer[p - 1]; u++) {
                            derivativesErrorWithRespectToWeights[p][i][u] = derivativesErrorWithRespectToWeights[p][i][u] + derivativesErrorWithRespectToInputsToActivation[p][i] * outputsInAllLayers[p - 1][u] / inputs.length;
                        }
                        derivativesErrorWithRespectToBiases[p][i] = derivativesErrorWithRespectToBiases[p][i] + derivativesErrorWithRespectToInputsToActivation[p][i] / inputs.length;
                    }
                } else {
                    for (int i = 0; i < nodesPerLayer[p]; i++) {

                        for (int nextLayerNode = 0; nextLayerNode < nodesPerLayer[p + 1]; nextLayerNode++) {
                            derivativesErrorWithRespectToInputsToActivation[p][i] = derivativesErrorWithRespectToInputsToActivation[p][i] + derivativesErrorWithRespectToInputsToActivation[p + 1][nextLayerNode] * weights[p + 1][nextLayerNode][i] * (ArrOperations.getDerivativeFromSigmoid(outputsInAllLayers[p][i]));
                        }
                        for (int u = 0; u < nodesPerLayer[p - 1]; u++) {
                            derivativesErrorWithRespectToWeights[p][i][u] = derivativesErrorWithRespectToWeights[p][i][u] + derivativesErrorWithRespectToInputsToActivation[p][i] * outputsInAllLayers[p - 1][u] / inputs.length;
                        }
                        derivativesErrorWithRespectToBiases[p][i] = derivativesErrorWithRespectToBiases[p][i] + derivativesErrorWithRespectToInputsToActivation[p][i] / inputs.length;
                    }
                }

            }
        }
        return;
    }


    //forward pass - first thing
    //stores outputs of each neuron in hidden layers
    public void forwardPass(float[] inputs) throws Exception {
        float[] in = inputs;
        float[][] outputsInHiddenLayersTemp = new float[layers][];

        for (int i = 0; i < layers; i++) {
            if (i == layers - 1) {
                float[] temp = new float[numOutputs];
                for (int u = 0; u < numOutputs; u++) {
                    //outputs = numOutputs in last layer
                    temp[u] = ArrOperations.dotProductNoGPU(weightsAfterDropout[i][u], in) + biases[i][u];
                }
                temp = ArrOperations.softmax(temp);
                outputsInHiddenLayersTemp[i] = temp;
                in = temp;
            } else if (i == 0) {
                //output is input in first layer
                outputsInHiddenLayersTemp[i] = in;
            } else {
                float[] temp = new float[nodesPerLayer[i]];
                for (int u = 0; u < nodesPerLayer[i]; u++) {
                    temp[u] = ArrOperations.sigmoidFunction(ArrOperations.dotProductNoGPU(weightsAfterDropout[i][u], in) + biases[i][u]);
                }
                outputsInHiddenLayersTemp[i] = temp;
                in = temp;
            }
        }

        outputsInAllLayers = outputsInHiddenLayersTemp;
    }

    //output as array of probabililities
    public float[] predictOutput(float[] array) throws Exception {
        float[] in = array;

        for (int i = 0; i < layers; i++) {
            if (i == layers - 1) {
                float[] temp = new float[numOutputs];
                for (int u = 0; u < numOutputs; u++) {
                    //outputs = numOutputs in last layer
                    temp[u] = ArrOperations.dotProductNoGPU(ArrOperations.vectorScalarProduct(weights[i][u], dropoutProbability), in) + biases[i][u];
                }
                in = ArrOperations.softmax(temp);
            } else if (i == 0) {

            } else {
                float[] temp = new float[nodesPerLayer[i]];
                for (int u = 0; u < nodesPerLayer[i]; u++) {
                    temp[u] = ArrOperations.sigmoidFunction(ArrOperations.dotProductNoGPU(ArrOperations.vectorScalarProduct(weights[i][u], dropoutProbability), in) + biases[i][u]);
                }
                in = temp;
            }
        }
        return in;
    }

    //weight initialization can be improved
    //intialize weights at very beginning
    public void setWeights() {
        weights = new float[layers][][];
        biases = new float[layers][];
        for (int i = 0; i < layers; i++) {
            weights[i] = new float[nodesPerLayer[i]][];
            biases[i] = new float[nodesPerLayer[i]];
            //last layer - numOutputs outputs

            if (i == 0) {
                //no weights first layer or input layer
            } else {
                for (int a = 0; a < nodesPerLayer[i]; a++) {
                    weights[i][a] = new float[nodesPerLayer[i - 1]];
                    biases[i][a] = ArrOperations.gaussianRandomVariable(.01f, 0);
                    for (int u = 0; u < nodesPerLayer[i - 1]; u++) {
                        weights[i][a][u] = ArrOperations.gaussianRandomVariable((float) Math.sqrt(2 / ((float) nodesPerLayer[i - 1] + (float) nodesPerLayer[i])), 0);
                    }
                }
            }
        }
        weightMomentumUpdate = getNewDerivativeWeights();
        weightsAfterDropout = weights;
    }

    public float[][] getNewDerivativeBiasesAndOutputs() {
        float[][] derivativesErrorWithRespectToBiasesCopy = new float[layers][];
        for (int i = 0; i < layers; i++) {
            derivativesErrorWithRespectToBiasesCopy[i] = new float[nodesPerLayer[i]];
            if (i == 0) {
                //no weights first layer or input layer
            } else {
                for (int a = 0; a < nodesPerLayer[i]; a++) {
                    derivativesErrorWithRespectToBiasesCopy[i][a] = 0f;
                }
            }
        }
        return derivativesErrorWithRespectToBiasesCopy;
    }

    public float[][][] getNewDerivativeWeights() {
        float[][][] derivativesErrorWithRespectToWeightsCopy = new float[layers][][];
        for (int i = 0; i < layers; i++) {
            derivativesErrorWithRespectToWeightsCopy[i] = new float[nodesPerLayer[i]][];

            if (i == 0) {
                //no weights first layer or input layer
            } else {
                for (int a = 0; a < nodesPerLayer[i]; a++) {
                    derivativesErrorWithRespectToWeightsCopy[i][a] = new float[nodesPerLayer[i - 1]];
                    for (int u = 0; u < nodesPerLayer[i - 1]; u++) {
                        derivativesErrorWithRespectToWeightsCopy[i][a][u] = 0f;
                    }
                }
            }
        }
        return derivativesErrorWithRespectToWeightsCopy;
    }


    public void setDropout() {
        if (dropoutProbability == 1) {
            weightsAfterDropout = weights;
            return;
        }
        dropoutProbability = dropoutProbability;
        weightsAfterDropout = getNewDerivativeWeights();
        for (int i = 1; i < weights.length; i++) {
            for (int a = 0; a < weights[i].length; a++) {
                for (int q = 0; q < weights[i][a].length; q++) {
                    if (rand.nextFloat() >= dropoutProbability) {
                        //will be ignored when doing backpropogation and gradient descent
                        weightsAfterDropout[i][a][q] = WEIGHT_DROPPED_OUT;
                    }
                }
            }
        }
    }

    public void setUpdateRule() {
        if (updateRule == UPDATE_NESTEROV) {
            for (int i = 1; i < weightsAfterDropout.length; i++) {
                for (int u = 0; u < weightsAfterDropout[i].length; u++) {
                    for (int a = 0; a < weightsAfterDropout[i][u].length; a++) {
                        if (weightsAfterDropout[i][u][a] != WEIGHT_DROPPED_OUT) {
                            weightsAfterDropout[i][u][a] += weightMomentumUpdate[i][u][a] * momentum;
                        }
                    }
                }
            }
        }
    }

    //tests on dataset, returns percentage accurate
    public float test(float[][] in, float[][] out) throws Exception {
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


    //NOTE: convenience methods not central to the neural network

    //not working
    public void setLearningRateAutomatically(float[][] trainingDataOutputs) {
        float maxOutput = -9999;
        float minOutput = 9999;
        for (int i = 0; i < 100; i++) {
            for (int p = 0; p < trainingDataOutputs[i].length; p++) {
                if (trainingDataOutputs[i][p] > maxOutput) {
                    maxOutput = trainingDataOutputs[i][p];
                } else if (trainingDataOutputs[i][p] < minOutput) {
                    minOutput = trainingDataOutputs[i][p];
                }
            }
        }
        for (int i = 1; i < learningRate.length - 1; i++) {
            learningRate[i] = 10 / nodesPerLayer[i];
        }
    }

    //doesnt work if dropout is used
    public float derivativeOfWeightCheck(float[] in, float[] out, int layer, int node, int previousLayerNode) throws Exception {
        float interval = .000001f;
        weights[layer][node][previousLayerNode] -= interval;
        float loss = ArrOperations.meanSquaredError(predictOutput(in), out);
        weights[layer][node][previousLayerNode] += interval * 2;
        float output = (ArrOperations.meanSquaredError(predictOutput(in), out) - loss) / interval / 2;
        weights[layer][node][previousLayerNode] -= interval;
        return output;
    }

    public class DeepDream {
        DenseNetwork network;
        public float[] image;
        float[] derivativeErrorWithRespectToInputs;

        float learningRate = 1;
        float max = -9999;
        float min = 9999;

        public DeepDream(DenseNetwork network, float[] image, float learningRate) {
            this.network = network;
            this.image = ArrOperations.makeCopy(image);
            this.learningRate = learningRate;


            for (int i = 0; i < image.length; i++) {
                if (image[i] > max) {
                    max = image[i];
                }
                if (image[i] < min) {
                    min = image[i];
                }
            }
        }

        public void updateImage(int desiredOutput) throws Exception {

            derivativeErrorWithRespectToInputs = new float[image.length];
            for (int i = 0; i < image.length; i++) {
                derivativeErrorWithRespectToInputs[i] = 0f;
            }

            float[] outputs = new float[network.numOutputs];
            for (int i = 0; i < network.numOutputs; i++) {
                if (i == desiredOutput) {
                    outputs[i] = 0f;
                } else {
                    outputs[i] = 0f;
                }
            }
            float[][] inputsToNetwork = new float[1][];
            inputsToNetwork[0] = image;

            float[][] outputsOfNetwork = new float[1][];
            outputsOfNetwork[0] = outputs;


            network.getDerivativeOfErrorWithRespectToWeights(inputsToNetwork, outputsOfNetwork);

            for (int i = 0; i < network.derivativesErrorWithRespectToInputsToActivation[1].length; i++) {
                for (int u = 0; u < network.numInputs; u++) {
                    derivativeErrorWithRespectToInputs[u] += network.derivativesErrorWithRespectToInputsToActivation[1][i] * network.weights[1][i][u];
                }
            }

            for (int i = 0; i < derivativeErrorWithRespectToInputs.length; i++) {
                image[i] = image[i] - derivativeErrorWithRespectToInputs[i] * learningRate;
                if (image[i] < min) {
                    image[i] = min;
                }
                if (image[i] > max) {
                    image[i] = max;
                }
            }
        }
    }


}
