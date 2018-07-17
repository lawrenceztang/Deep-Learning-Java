package Network;

import javax.swing.*;
import java.io.File;
import java.io.Serializable;
import java.util.Random;
import Util.*;


public class FullyConnectedNetwork implements Serializable {
    Random rand;

    int layers;
    int[] nodesPerLayer;
    int numInputs;
    int numOutputs;
    double learningRate[];
    double momentum;
    double probabilityNeuronRetained = 1;


    double[][][] weights;
    double[][] biases;
    double[][] derivativesErrorWithRespectToBiases;
    public double[][][] derivativesErrorWithRespectToWeights;
    double[][] outputsInAllLayers;
    double[][][] weightMomentumUpdate;
    double[][] derivativesErrorWithRespectToInputsToActivation;
    public String[] classNames;
    double[][][] weightsAfterDropout;

    public FullyConnectedNetwork(int[] nodesPerLayer, double[] learningRate, double momentum) {
        rand = new Random();
        outputsInAllLayers = new double[nodesPerLayer.length][];
        weightsAfterDropout = new double[nodesPerLayer.length][][];

        //layers includes output and input layer
        this.layers = nodesPerLayer.length;
        this.nodesPerLayer = nodesPerLayer;
        this.numInputs = nodesPerLayer[0];
        this.numOutputs = nodesPerLayer[nodesPerLayer.length - 1];
        this.learningRate = learningRate;
        this.momentum = momentum;

        setWeights();
    }

    public void gradientDescent() {

        for (int i = 1; i < layers; i++) {
            int nodesPreviousLayer = nodesPerLayer[i - 1];
            int nodesThisLayer = nodesPerLayer[i];

            for (int t = 0; t < nodesThisLayer; t++) {
                for (int y = 0; y < nodesPreviousLayer; y++) {
                    if(weightsAfterDropout[i][t][y] != 0) {
                        weights[i][t][y] = weights[i][t][y] - weightMomentumUpdate[i][t][y] * learningRate[i];
                    }

                    biases[i][t] = biases[i][t] - derivativesErrorWithRespectToBiases[i][t] * learningRate[i] * .01;
                }
            }
        }

        //ensure that if dropout isnt set weightsAfterDropout is still updated - weightsAfterDropout is used for forwardPass
        weightsAfterDropout = weights;
    }

    //todo: gradients are 10 times larger than actually in last layer
    public void getDerivativeOfErrorWithRespectToWeights(double[][] inputs, double[][] outputs) throws Exception{
        //no need to worry about dropped out weights because they automatically have 0 derivative so they arent updated in gradient descent
        //try mean squared error
        derivativesErrorWithRespectToWeights = getNewDerivativeWeights();
        derivativesErrorWithRespectToBiases = getNewDerivativeBiasesAndOutputs();
        for (int w = 0; w < inputs.length; w++) {
            derivativesErrorWithRespectToInputsToActivation = getNewDerivativeBiasesAndOutputs();
            forwardPass(inputs[w]);
            for (int p = layers - 1; p > 0; p--) {
                if (p == layers - 1) {
                    derivativesErrorWithRespectToInputsToActivation[p] = arrOperations.getDerivativeFromSoftmax(outputsInAllLayers[p], arrOperations.getDerivativeFromMSE(outputs[w], outputsInAllLayers[layers - 1]));
                    for (int i = 0; i < numOutputs; i++) {
                        for (int u = 0; u < nodesPerLayer[p]; u++) {
                            derivativesErrorWithRespectToWeights[p][i][u] = derivativesErrorWithRespectToWeights[p][i][u] + derivativesErrorWithRespectToInputsToActivation[p][i] * outputsInAllLayers[p - 1][u] / inputs.length;
                        }
                        derivativesErrorWithRespectToBiases[p][i] = derivativesErrorWithRespectToBiases[p][i] + derivativesErrorWithRespectToInputsToActivation[p][i] / inputs.length;
                    }
                } else {
                    for (int i = 0; i < nodesPerLayer[p]; i++) {

                        for (int nextLayerNode = 0; nextLayerNode < nodesPerLayer[p + 1]; nextLayerNode++) {
                            derivativesErrorWithRespectToInputsToActivation[p][i] = derivativesErrorWithRespectToInputsToActivation[p][i] + derivativesErrorWithRespectToInputsToActivation[p + 1][nextLayerNode] * weights[p + 1][nextLayerNode][i] * (arrOperations.getDerivativeFromSigmoid(outputsInAllLayers[p][i]));
                        }
                        for (int u = 0; u < nodesPerLayer[p - 1]; u++) {
                            derivativesErrorWithRespectToWeights[p][i][u] = derivativesErrorWithRespectToWeights[p][i][u] + derivativesErrorWithRespectToInputsToActivation[p][i] * outputsInAllLayers[p - 1][u] / inputs.length;
                        }
                        derivativesErrorWithRespectToBiases[p][i] = derivativesErrorWithRespectToBiases[p][i] + derivativesErrorWithRespectToInputsToActivation[p][i] / inputs.length;
                    }
                }

            }
        }

        for(int i = 1; i < derivativesErrorWithRespectToWeights.length; i++) {
            for(int u = 0; u < derivativesErrorWithRespectToWeights[i].length; u++) {
                for(int g = 0; g < derivativesErrorWithRespectToWeights[i][u].length; g++) {
                    weightMomentumUpdate[i][u][g] = weightMomentumUpdate[i][u][g] * momentum + (1 - momentum) * derivativesErrorWithRespectToWeights[i][u][g];
                }
            }
        }

    }


    //forward pass - first thing
    //stores outputs of each neuron in hidden layers
    public void forwardPass(double[] inputs) throws Exception{
        double[] in = inputs;
        double[][] outputsInHiddenLayersTemp = new double[layers][];

        for (int i = 0; i < layers; i++) {
            if (i == layers - 1) {
                double[] temp = new double[numOutputs];
                for (int u = 0; u < numOutputs; u++) {
                    //outputs = numOutputs in last layer
                    temp[u] = arrOperations.dotProductNoGPU(weightsAfterDropout[i][u], in) + biases[i][u];
                }
                temp = arrOperations.softmax(temp);
                outputsInHiddenLayersTemp[i] = temp;
                in = temp;
            } else if (i == 0) {
                //output is input in first layer
                outputsInHiddenLayersTemp[i] = in;
            } else {
                double[] temp = new double[nodesPerLayer[i]];
                for (int u = 0; u < nodesPerLayer[i]; u++) {
                    temp[u] = arrOperations.sigmoidFunction(arrOperations.dotProductNoGPU(weightsAfterDropout[i][u], in) + biases[i][u]);
                }
                outputsInHiddenLayersTemp[i] = temp;
                in = temp;
            }
        }

        outputsInAllLayers = outputsInHiddenLayersTemp;
    }

    //output as array of probabililities
    public double[] predictOutput(double[] array) throws Exception{
        double[] in = array;

        for (int i = 0; i < layers; i++) {
            if (i == layers - 1) {
                double[] temp = new double[numOutputs];
                for (int u = 0; u < numOutputs; u++) {
                    //outputs = numOutputs in last layer
                    temp[u] = arrOperations.dotProductNoGPU(arrOperations.vectorScalarProduct(weights[i][u], probabilityNeuronRetained), in) + biases[i][u];
                }
                in = arrOperations.softmax(temp);
            } else if (i == 0) {

            } else {
                double[] temp = new double[nodesPerLayer[i]];
                for (int u = 0; u < nodesPerLayer[i]; u++) {
                    temp[u] = arrOperations.sigmoidFunction(arrOperations.dotProductNoGPU(arrOperations.vectorScalarProduct(weights[i][u], probabilityNeuronRetained), in) + biases[i][u]);
                }
                in = temp;
            }
        }
        return in;
    }

    //weight initialization can be improved
    //intialize weights at very beginning
    public void setWeights() {
        weights = new double[layers][][];
        biases = new double[layers][];
        for (int i = 0; i < layers; i++) {
            weights[i] = new double[nodesPerLayer[i]][];
            biases[i] = new double[nodesPerLayer[i]];
            //last layer - numOutputs outputs

            if (i == 0) {
                //no weights first layer or input layer
            }

            else {
                for (int a = 0; a < nodesPerLayer[i]; a++) {
                    weights[i][a] = new double[nodesPerLayer[i - 1]];
                    biases[i][a] = 0d;
                    for (int u = 0; u < nodesPerLayer[i - 1]; u++) {
                        weights[i][a][u] = rand.nextDouble() * .1 - .05;
                    }
                }
            }
        }
        weightMomentumUpdate = getNewDerivativeWeights();
        weightsAfterDropout = weights;
    }

    public double[][] getNewDerivativeBiasesAndOutputs() {
        double[][] derivativesErrorWithRespectToBiasesCopy = new double[layers][];
        for (int i = 0; i < layers; i++) {
            derivativesErrorWithRespectToBiasesCopy[i] = new double[nodesPerLayer[i]];
            if (i == 0) {
                //no weights first layer or input layer
            }
            else {
                for (int a = 0; a < nodesPerLayer[i]; a++) {
                    derivativesErrorWithRespectToBiasesCopy[i][a] = 0d;
                }
            }
        }
        return derivativesErrorWithRespectToBiasesCopy;
    }

    public double[][][] getNewDerivativeWeights() {
        double[][][] derivativesErrorWithRespectToWeightsCopy = new double[layers][][];
        for (int i = 0; i < layers; i++) {
            derivativesErrorWithRespectToWeightsCopy[i] = new double[nodesPerLayer[i]][];

            if (i == 0) {
                //no weights first layer or input layer
            }
            else {
                for (int a = 0; a < nodesPerLayer[i]; a++) {
                    derivativesErrorWithRespectToWeightsCopy[i][a] = new double[nodesPerLayer[i - 1]];
                    for (int u = 0; u < nodesPerLayer[i - 1]; u++) {
                        derivativesErrorWithRespectToWeightsCopy[i][a][u] = 0d;
                    }
                }
            }
        }
        return derivativesErrorWithRespectToWeightsCopy;
    }

    public void setDropout (double probability) {
        probabilityNeuronRetained = probability;
        weightsAfterDropout = getNewDerivativeWeights();
        for(int i = 1; i < weights.length; i++) {
            for(int a = 0; a < weights[i].length; a++) {
                    for (int q = 0; q < weights[i][a].length; q++) {
                        if (rand.nextDouble() < probability) {
                            weightsAfterDropout[i][a][q] = weights[i][a][q];
                        } else {
                            weightsAfterDropout[i][a][q] = 0d;
                        }
                    }
            }
        }
    }

    //tests on dataset, returns percentage accurate
    public double test(double[][] in, double[][] out) throws Exception{
        int numCorrect = 0;
        for (int i = 0; i < in.length; i++) {
            double[] array = predictOutput(in[i]);
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
        return numCorrect / (double) in.length;
    }


    //NOTE: convenience methods not central to the neural network


    //output as highest probability classname
    public String getPredictionClass(double[] in) throws Exception{
        double[] out = predictOutput(in);

        int maxOutput = 0;
        for (int i = 0; i < out.length; i++) {
            if (out[i] > out[maxOutput]) {
                maxOutput = i;
            }
        }
        return classNames[maxOutput];
    }


    //tests on single file that is chosen, returns predicted class
    public String test(ImageReader reader) throws Exception {


        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);

        if (ret == JFileChooser.APPROVE_OPTION) {

            File file = fc.getSelectedFile();
            String fileName = file.getAbsolutePath();
            return this.getPredictionClass(reader.preprocessExample(reader.getImageAs1DMatrix(fileName, 28)));

        } else {
            return null;
        }
    }

    //not working
    public void setLearningRateAutomatically(double[][] trainingDataOutputs) {
        double maxOutput = -9999;
        double minOutput = 9999;
        for (int i = 0; i < 100; i++) {
            for (int p = 0; p < trainingDataOutputs[i].length; p++) {
                if (trainingDataOutputs[i][p] > maxOutput) {
                    maxOutput = trainingDataOutputs[i][p];
                } else if (trainingDataOutputs[i][p] < minOutput) {
                    minOutput = trainingDataOutputs[i][p];
                }
            }
        }
        for(int i = 1; i < learningRate.length - 1; i++) {
            learningRate[i] = 10 / nodesPerLayer[i];
        }
    }

    //doesnt work if dropout is used
    public double derivativeOfWeightCheck(double[] in, double[] out, int layer, int node, int previousLayerNode) throws Exception{
        double interval = .000001;
        weights[layer][node][previousLayerNode] -= interval;
        double loss = arrOperations.meanSquaredError(predictOutput(in), out);
        weights[layer][node][previousLayerNode] += interval * 2;
        double output = (arrOperations.meanSquaredError(predictOutput(in), out) - loss) / interval / 2;
        weights[layer][node][previousLayerNode] -= interval;
        return output;
    }



    public void printStats() {
        System.out.println("Neural Network Parameters:");
        System.out.println("Layers: " + layers);
        System.out.println("Nodes in each layer: " + nodesPerLayer);
        System.out.println("Number of inputs: " + numInputs);
        System.out.println("Number of outputs: " + numOutputs);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Momentum: " + momentum);
    }

    public class DeepDream {
        FullyConnectedNetwork network;
        public double[] image;
        double[] derivativeErrorWithRespectToInputs;

        double learningRate = 1;
        double max = -9999;
        double min = 9999;

        public DeepDream(FullyConnectedNetwork network, double[] image, double learningRate) {
            this.network = network;
            this.image = arrOperations.makeCopy(image);
            this.learningRate = learningRate;


            for(int i = 0; i < image.length; i++) {
                if(image[i] > max) {
                    max = image[i];
                }
                if(image[i] < min) {
                    min = image[i];
                }
            }
        }

        public void updateImage(int desiredOutput) throws  Exception{

            derivativeErrorWithRespectToInputs = new double[image.length];
            for(int i = 0; i < image.length; i++) {
                derivativeErrorWithRespectToInputs[i] = 0d;
            }

            double[] outputs = new double[network.numOutputs];
            for (int i = 0; i < network.numOutputs; i++) {
                if (i == desiredOutput) {
                    outputs[i] = 0d;
                } else {
                    outputs[i] = 0d;
                }
            }
            double[][] inputsToNetwork = new double[1][];
            inputsToNetwork[0] = image;

            double[][] outputsOfNetwork = new double[1][];
            outputsOfNetwork[0] = outputs;


            network.getDerivativeOfErrorWithRespectToWeights(inputsToNetwork, outputsOfNetwork);

            for (int i = 0; i < network.derivativesErrorWithRespectToInputsToActivation[1].length; i++) {
                for (int u = 0; u < network.numInputs; u++) {
                    derivativeErrorWithRespectToInputs[u] = derivativeErrorWithRespectToInputs[u] + network.derivativesErrorWithRespectToInputsToActivation[1][i] * network.weights[1][i][u];
                }
            }

            for(int i = 0; i < derivativeErrorWithRespectToInputs.length; i++) {
                image[i] = image[i] - derivativeErrorWithRespectToInputs[i] * learningRate;
                if(image[i] < min) {
                    image[i] = min;
                }
                if(image[i] > max) {
                    image[i] = max;
                }
            }
        }
    }


}
