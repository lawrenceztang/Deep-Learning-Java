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

    final static float WEIGHT_DROPPED_OUT = 1;
    final static float WEIGHT_KEPT = 0;

    //hyperparameters
    public int layers;
    public int[] nodesPerLayer;
    public int numInputs;
    public int numOutputs;
    public float learningRate[];
    public float momentum;
    public float dropoutProbability = 1;
    public int updateRule = 0;


    //learned parameters
    public float[][][] weights;
    public float[][] biases;
    public float[][] derivativesErrorWithRespectToBiases;
    public float[][][] derivativesErrorWithRespectToWeights;
    public float[][] outputsInAllLayers;
    public float[][][] weightMomentumUpdate;
    public float[][] derivativesErrorWithRespectToInputsToActivation;
    public float[][] weightsAfterDropout;

    public DenseNetwork(int[] nodesPerLayer, float[] learningRate, int updateRule, float momentum, float dropoutProbability) {
        rand = new Random();
        outputsInAllLayers = new float[nodesPerLayer.length][];
        weightsAfterDropout = new float[nodesPerLayer.length][];

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
            for (int t = 0; t < weights[i].length; t++) {
                if (weightsAfterDropout[i][t] == WEIGHT_KEPT) {
                    for (int y = 0; y < weights[i][t].length; y++) {
                        weightMomentumUpdate[i][t][y] = weightMomentumUpdate[i][t][y] * momentum + (1 - momentum) * learningRate[i] * derivativesErrorWithRespectToWeights[i][t][y];
                        weights[i][t][y] = weights[i][t][y] - weightMomentumUpdate[i][t][y];
                    }
                    biases[i][t] = biases[i][t] - derivativesErrorWithRespectToBiases[i][t] * learningRate[i] * 10f;
                }
            }
        }

        //ensure that if dropout isnt set weightsAfterDropout is still updated - weightsAfterDropout is used for forwardPass
    }

    //todo: gradients are 10 times larger than actually in last layer
    public void getDerivativeOfErrorWithRespectToWeights(float[][] inputs, float[][] outputs) throws Exception {
        //no need to worry about dropped out weights because they automatically have 0 derivative so they arent updated in gradient descent
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
                        if(weightsAfterDropout[p][i] == WEIGHT_KEPT) {
                            for (int u = 0; u < nodesPerLayer[p - 1]; u++) {
                                derivativesErrorWithRespectToWeights[p][i][u] = derivativesErrorWithRespectToWeights[p][i][u] + derivativesErrorWithRespectToInputsToActivation[p][i] * outputsInAllLayers[p - 1][u] / inputs.length;
                            }
                            derivativesErrorWithRespectToBiases[p][i] = derivativesErrorWithRespectToBiases[p][i] + derivativesErrorWithRespectToInputsToActivation[p][i] / inputs.length;
                        }
                    }

                    for(int i = 0; i < derivativesErrorWithRespectToInputsToActivation[p].length; i++) {
                        if(weightsAfterDropout[p][i] != WEIGHT_KEPT) {
                            derivativesErrorWithRespectToInputsToActivation[p][i] = 0;
                        }
                    }

                } else {
                    for (int i = 0; i < nodesPerLayer[p]; i++) {
                        if (weightsAfterDropout[p][i] == WEIGHT_KEPT) {

                            for (int nextLayerNode = 0; nextLayerNode < nodesPerLayer[p + 1]; nextLayerNode++) {

                                derivativesErrorWithRespectToInputsToActivation[p][i] = derivativesErrorWithRespectToInputsToActivation[p][i] + derivativesErrorWithRespectToInputsToActivation[p + 1][nextLayerNode] * weights[p + 1][nextLayerNode][i] * (ArrOperations.getDerivativeFromSigmoid(outputsInAllLayers[p][i]));
                            }
                            for (int u = 0; u < nodesPerLayer[p - 1]; u++) {
                                derivativesErrorWithRespectToWeights[p][i][u] = derivativesErrorWithRespectToWeights[p][i][u] + derivativesErrorWithRespectToInputsToActivation[p][i] * outputsInAllLayers[p - 1][u] / inputs.length;
                            }
                            derivativesErrorWithRespectToBiases[p][i] = derivativesErrorWithRespectToBiases[p][i] + derivativesErrorWithRespectToInputsToActivation[p][i] / inputs.length;
                        } else {
                            derivativesErrorWithRespectToInputsToActivation[p][i] = 0;
                        }
                    }
                }

            }
        }
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
                    if (weightsAfterDropout[i][u] != WEIGHT_DROPPED_OUT) {
                        temp[u] = ArrOperations.dotProductNoGPU(weights[i][u], in) + biases[i][u];
                    } else {
                        temp[u] = 0;
                    }
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
                    if (weightsAfterDropout[i][u] != WEIGHT_DROPPED_OUT) {
                        temp[u] = ArrOperations.sigmoidFunction(ArrOperations.dotProductNoGPU(weights[i][u], in) + biases[i][u]);
                    } else {
                        temp[u] = 0;
                    }
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
                    temp[u] = (ArrOperations.dotProductNoGPU(weights[i][u], in) + biases[i][u]) * dropoutProbability;
                }
                in = ArrOperations.softmax(temp);
            } else if (i == 0) {

            } else {
                float[] temp = new float[nodesPerLayer[i]];
                for (int u = 0; u < nodesPerLayer[i]; u++) {
                    temp[u] = ArrOperations.sigmoidFunction((ArrOperations.dotProductNoGPU(weights[i][u], in) + biases[i][u]) * dropoutProbability);
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

        weightsAfterDropout = new float[weights.length][];

        if (dropoutProbability == 1) {
            for (int i = 0; i < weightsAfterDropout.length; i++) {
                weightsAfterDropout[i] = new float[weights[i].length];
                for (int a = 0; a < weightsAfterDropout[i].length; a++) {
                    weightsAfterDropout[i][a] = WEIGHT_KEPT;
                }
            }
            return;
        }

        for (int i = 1; i < weights.length; i++) {
            weightsAfterDropout[i] = new float[weights[i].length];
            for (int a = 0; a < weights[i].length; a++) {
                if (rand.nextFloat() >= dropoutProbability) {
                    //will be ignored when doing backpropogation and gradient descent
                    weightsAfterDropout[i][a] = WEIGHT_DROPPED_OUT;
                } else {
                    weightsAfterDropout[i][a] = WEIGHT_KEPT;
                }
            }
        }
    }

    public void setUpdateRule() {
        if (updateRule == UPDATE_NESTEROV) {
            for (int i = 1; i < weights.length; i++) {
                for (int u = 0; u < weights[i].length; u++) {
                    if (weightsAfterDropout[i][u] != WEIGHT_DROPPED_OUT) {
                        for (int a = 0; a < weights[i][u].length; a++) {
                            weights[i][u][a] += weightMomentumUpdate[i][u][a] * momentum;
                        }
                    }
                }
            }
        } else {

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

    public float testLoss(float[][] in, float[][] out) throws Exception {
        int loss = 0;
        for (int i = 0; i < in.length; i++) {
            loss += ArrOperations.meanSquaredError(predictOutput(in[i]), out[i]);
        }
        return loss / (float) in.length;
    }

    public AutoEncoder[] encoders;

    public void initializeEncoders() {
        encoders = new AutoEncoder[layers - 2];
        for (int i = 0; i < layers - 2; i++) {
            encoders[i] = new AutoEncoder(new int[]{nodesPerLayer[i], nodesPerLayer[i + 1], nodesPerLayer[i]}, learningRate, .9f);
        }
    }

    public void preTrain(float[][] in, float[] learningRate, int batchSize, float noise) throws Exception {
        for (int i = 0; i < layers - 2; i++) {

            encoders[i].setNoise(noise);

            for (int u = 0; u < in.length; u += batchSize) {
                float[][] temp = new float[batchSize][];
                for (int d = 0; d < batchSize; d++) {
                    temp[d] = in[u + d];
                }
                encoders[i].getDerivativeOfErrorWithRespectToWeights(temp);
                encoders[i].gradientDescent();
            }

            this.weights[i] = encoders[i].weights[1];
        }
    }


    //NOTE: convenience methods not central to the neural network


    public float derivativeOfWeightCheck(float[][] in, float[][] out, int layer, int node, int previousLayerNode) throws Exception {

        float interval = .001f;
        float derivative = 0;

        for (int i = 0; i < in.length; i++) {
            weights[layer][node][previousLayerNode] -= interval;
            float loss = ArrOperations.meanSquaredError(predictOutput(in[i]), out[i]);
            weights[layer][node][previousLayerNode] += interval * 2;
            derivative += (ArrOperations.meanSquaredError(predictOutput(in[i]), out[i]) - loss) / interval / 2;
            weights[layer][node][previousLayerNode] -= interval;
        }

        return derivative / in.length;
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
                    outputs[i] = 1f;
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
