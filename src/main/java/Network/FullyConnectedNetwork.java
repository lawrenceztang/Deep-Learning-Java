package Network;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import Util.*;


public class FullyConnectedNetwork implements Serializable {
    Random rand;

    int layers;
    int nodesPerLayer;
    int numInputs;
    int numOutputs;
    double learningRate;
    double momentum;
    double probabilityNeuronRetained;


    ArrayList<ArrayList<ArrayList<Double>>> weights;
    ArrayList<ArrayList<Double>> biases;
    ArrayList<ArrayList<Double>> derivativesErrorWithRespectToBiases;
    public ArrayList<ArrayList<ArrayList<Double>>> derivativesErrorWithRespectToWeights;
    ArrayList<ArrayList<Double>> outputsInAllLayers;
    ArrayList<ArrayList<ArrayList<Double>>> weightMomentumUpdate;
    ArrayList<ArrayList<Double>> derivativesErrorWithRespectToInputsToActivation;
    public ArrayList<String> classNames;
    ArrayList<ArrayList<ArrayList<Double>>> weightsAfterDropout;

    public FullyConnectedNetwork(int layers, int nodesPerLayer, int numInputs, int numOutputs, double learningRate, double momentum) {
        rand = new Random();
        outputsInAllLayers = new ArrayList<ArrayList<Double>>();
        weightsAfterDropout = new ArrayList<ArrayList<ArrayList<Double>>>();

        //layers includes output and input layer
        this.layers = layers;
        this.nodesPerLayer = nodesPerLayer;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.learningRate = learningRate;
        this.momentum = momentum;

        setWeights();
    }

    public void gradientDescent() {

        for (int i = 1; i < layers; i++) {
            int nodesPreviousLayer = nodesPerLayer;
            int nodesThisLayer = nodesPerLayer;
            if (i == layers - 1) {
                nodesThisLayer = numOutputs;
            }
            if (i == 1) {
                nodesPreviousLayer = numInputs;
            }

            for (int t = 0; t < nodesThisLayer; t++) {
                for (int y = 0; y < nodesPreviousLayer; y++) {
                    if(weightsAfterDropout.get(i).get(t).get(y) != 0) {
                        weights.get(i).get(t).set(y, weights.get(i).get(t).get(y) - weightMomentumUpdate.get(i).get(t).get(y) * learningRate);
                    }

                    biases.get(i).set(t, biases.get(i).get(t) - derivativesErrorWithRespectToBiases.get(i).get(t) * learningRate * .01);
                }
            }
        }
    }

    //todo: gradients are 10 times larger than actually in last layer, gradients completely innacurate in other layers
    public void getDerivativeOfErrorWithRespectToWeights(ArrayList<ArrayList<Double>> inputs, ArrayList<ArrayList<Double>> outputs) throws Exception{

        //try mean squared error
        derivativesErrorWithRespectToWeights = getNewDerivativeWeights();
        derivativesErrorWithRespectToBiases = getNewDerivativeBiasesAndOutputs();
        for (int w = 0; w < inputs.size(); w++) {
            derivativesErrorWithRespectToInputsToActivation = getNewDerivativeBiasesAndOutputs();
            forwardPass(inputs.get(w));
            for (int p = layers - 1; p > 0; p--) {
                if (p == layers - 1) {
                    derivativesErrorWithRespectToInputsToActivation.set(p, Util.getDerivativeFromSoftmax(outputsInAllLayers.get(p), Util.getDerivativeFromMSE(outputs.get(w), outputsInAllLayers.get(layers - 1))));
                    for (int i = 0; i < numOutputs; i++) {
                        for (int u = 0; u < nodesPerLayer; u++) {
                            derivativesErrorWithRespectToWeights.get(p).get(i).set(u, derivativesErrorWithRespectToWeights.get(p).get(i).get(u) + derivativesErrorWithRespectToInputsToActivation.get(p).get(i) * outputsInAllLayers.get(p - 1).get(u) / inputs.size());
                        }
                        derivativesErrorWithRespectToBiases.get(p).set(i, derivativesErrorWithRespectToBiases.get(p).get(i) + derivativesErrorWithRespectToInputsToActivation.get(p).get(i) / inputs.size());
                    }
                } else {
                    for (int i = 0; i < nodesPerLayer; i++) {
                        int numberOfNeuronsInNextLayer;
                        int numberofNeuronsPreviousLayer;
                        if (p == layers - 2) {
                            numberOfNeuronsInNextLayer = numOutputs;
                        } else {
                            numberOfNeuronsInNextLayer = nodesPerLayer;
                        }
                        if (p == 1) {
                            numberofNeuronsPreviousLayer = numInputs;
                        }
                        else {
                            numberofNeuronsPreviousLayer = nodesPerLayer;
                        }
                        for (int nextLayerNode = 0; nextLayerNode < numberOfNeuronsInNextLayer; nextLayerNode++) {
                            derivativesErrorWithRespectToInputsToActivation.get(p).set(i, derivativesErrorWithRespectToInputsToActivation.get(p).get(i) + derivativesErrorWithRespectToInputsToActivation.get(p + 1).get(nextLayerNode) * weights.get(p + 1).get(nextLayerNode).get(i) * (Util.getDerivativeFromSigmoid(outputsInAllLayers.get(p).get(i))));
                        }
                        for (int u = 0; u < numberofNeuronsPreviousLayer; u++) {
                            derivativesErrorWithRespectToWeights.get(p).get(i).set(u, derivativesErrorWithRespectToWeights.get(p).get(i).get(u) + derivativesErrorWithRespectToInputsToActivation.get(p).get(i) * outputsInAllLayers.get(p - 1).get(u) / inputs.size());
                        }
                        derivativesErrorWithRespectToBiases.get(p).set(i, derivativesErrorWithRespectToBiases.get(p).get(i) + derivativesErrorWithRespectToInputsToActivation.get(p).get(i) / inputs.size());
                    }
                }

            }
        }

        for(int i = 0; i < derivativesErrorWithRespectToWeights.size(); i++) {
            for(int u = 0; u < derivativesErrorWithRespectToWeights.get(i).size(); u++) {
                for(int g = 0; g < derivativesErrorWithRespectToWeights.get(i).get(u).size(); g++) {
                    weightMomentumUpdate.get(i).get(u).set(g, weightMomentumUpdate.get(i).get(u).get(g) * momentum + (1 - momentum) * derivativesErrorWithRespectToWeights.get(i).get(u).get(g));
                }
            }
        }

    }


    //TODO: dropout
    //forward pass - first thing
    //stores outputs of each neuron in hidden layers
    public void forwardPass(ArrayList<Double> inputs) throws Exception{
        ArrayList<Double> in = inputs;
        ArrayList<ArrayList<Double>> outputsInHiddenLayersTemp = new ArrayList<ArrayList<Double>>();

        for (int i = 0; i < layers; i++) {
            if (i == layers - 1) {
                ArrayList<Double> temp = new ArrayList<Double>();
                for (int u = 0; u < numOutputs; u++) {
                    //outputs = numOutputs in last layer
                    temp.add(Util.dotProduct(weightsAfterDropout.get(i).get(u), in, 2) + biases.get(i).get(u));
                }
                temp = Util.softmax(temp);
                outputsInHiddenLayersTemp.add(temp);
                in = temp;
            } else if (i == 0) {
                //output is input in first layer
                outputsInHiddenLayersTemp.add(in);
            } else {
                ArrayList<Double> temp = new ArrayList<Double>();
                for (int u = 0; u < nodesPerLayer; u++) {
                    temp.add(Util.sigmoidFunction(Util.dotProduct(weightsAfterDropout.get(i).get(u), in, 2) + biases.get(i).get(u)));
                }
                outputsInHiddenLayersTemp.add(temp);
                in = temp;
            }
        }

        outputsInAllLayers = outputsInHiddenLayersTemp;
    }

    //output as array of probabililities
    public ArrayList<Double> predictOutput(ArrayList<Double> array) throws Exception{
        ArrayList<Double> in = array;

        for (int i = 0; i < layers; i++) {
            if (i == layers - 1) {
                ArrayList<Double> temp = new ArrayList<Double>();
                for (int u = 0; u < numOutputs; u++) {
                    //outputs = numOutputs in last layer
                    temp.add(Util.dotProduct(Util.vectorScalarProduct(weights.get(i).get(u), probabilityNeuronRetained), in, 2) + biases.get(i).get(u));
                }
                in = Util.softmax(temp);
            } else if (i == 0) {

            } else {
                ArrayList<Double> temp = new ArrayList<Double>();
                for (int u = 0; u < nodesPerLayer; u++) {
                    temp.add(Util.sigmoidFunction(Util.dotProduct(Util.vectorScalarProduct(weights.get(i).get(u), probabilityNeuronRetained), in, 2) + biases.get(i).get(u)));
                }
                in = temp;
            }
        }
        return in;
    }

    //weight initialization can be improved
    //intialize weights at very beginning
    public void setWeights() {
        weights = new ArrayList<ArrayList<ArrayList<Double>>>();
        biases = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < layers; i++) {
            weights.add(new ArrayList<ArrayList<Double>>());
            biases.add(new ArrayList<Double>());
            //last layer - numOutputs outputs
            if (i == layers - 1) {
                for (int a = 0; a < numOutputs; a++) {
                    weights.get(i).add(new ArrayList<Double>());
                    biases.get(i).add(0d);
                    for (int u = 0; u < nodesPerLayer; u++) {
                        weights.get(i).get(a).add(rand.nextDouble() * .1 - .05);
                    }
                }
            } else if (i == 0) {
                //no weights first layer or input layer
            }

            //2nd layer - numInput inputs
            else if (i == 1) {
                for (int a = 0; a < nodesPerLayer; a++) {
                    weights.get(i).add(new ArrayList<Double>());
                    biases.get(i).add(0d);
                    for (int u = 0; u < numInputs; u++) {
                        weights.get(i).get(a).add(rand.nextDouble() * .1 - .05);
                    }
                }
            }

            //all other layers
            else {
                for (int a = 0; a < nodesPerLayer; a++) {
                    weights.get(i).add(new ArrayList<Double>());
                    biases.get(i).add(0d);
                    for (int u = 0; u < nodesPerLayer; u++) {
                        weights.get(i).get(a).add(rand.nextDouble() * .1 - .05);
                    }
                }
            }
        }
        weightMomentumUpdate = new ArrayList<ArrayList<ArrayList<Double>>>(getNewDerivativeWeights());
    }

    public ArrayList<ArrayList<Double>> getNewDerivativeBiasesAndOutputs() {
        ArrayList<ArrayList<Double>> derivativesErrorWithRespectToBiasesCopy = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < layers; i++) {
            derivativesErrorWithRespectToBiasesCopy.add(new ArrayList<Double>());
            //last layer - numOutputs outputs
            if (i == layers - 1) {
                for (int a = 0; a < numOutputs; a++) {
                    derivativesErrorWithRespectToBiasesCopy.get(i).add(0d);
                }
            } else if (i == 0) {
                //no weights first layer or input layer
            }
            //2nd layer - numInput inputs
            else if (i == 1) {
                for (int a = 0; a < nodesPerLayer; a++) {
                    derivativesErrorWithRespectToBiasesCopy.get(i).add(0d);
                }
            }
            //all other layers
            else {
                for (int a = 0; a < nodesPerLayer; a++) {
                    derivativesErrorWithRespectToBiasesCopy.get(i).add(0d);
                }
            }
        }
        return derivativesErrorWithRespectToBiasesCopy;
    }

    public ArrayList<ArrayList<ArrayList<Double>>> getNewDerivativeWeights() {
        ArrayList<ArrayList<ArrayList<Double>>> derivativesErrorWithRespectToWeightsCopy = new ArrayList<ArrayList<ArrayList<Double>>>();
        for (int i = 0; i < layers; i++) {
            derivativesErrorWithRespectToWeightsCopy.add(new ArrayList<ArrayList<Double>>());
            //last layer - numOutputs outputs
            if (i == layers - 1) {
                for (int a = 0; a < numOutputs; a++) {
                    derivativesErrorWithRespectToWeightsCopy.get(i).add(new ArrayList<Double>());
                    for (int u = 0; u < nodesPerLayer; u++) {
                        derivativesErrorWithRespectToWeightsCopy.get(i).get(a).add(0d);
                    }
                }
            } else if (i == 0) {
                //no weights first layer or input layer
            }
            //2nd layer - numInput inputs
            else if (i == 1) {
                for (int a = 0; a < nodesPerLayer; a++) {
                    derivativesErrorWithRespectToWeightsCopy.get(i).add(new ArrayList<Double>());
                    for (int u = 0; u < numInputs; u++) {
                        derivativesErrorWithRespectToWeightsCopy.get(i).get(a).add(0d);
                    }
                }
            }
            //all other layers
            else {
                for (int a = 0; a < nodesPerLayer; a++) {
                    derivativesErrorWithRespectToWeightsCopy.get(i).add(new ArrayList<Double>());
                    for (int u = 0; u < nodesPerLayer; u++) {
                        derivativesErrorWithRespectToWeightsCopy.get(i).get(a).add(0d);
                    }
                }
            }
        }
        return derivativesErrorWithRespectToWeightsCopy;
    }

    public void setDropout (double probability) {
        probabilityNeuronRetained = probability;
        weightsAfterDropout = getNewDerivativeWeights();
        for(int i = 0; i < weights.size(); i++) {
            for(int a = 0; a < weights.get(i).size(); a++) {
                for(int q = 0; q < weights.get(i).get(a).size(); q++) {
                    if(rand.nextDouble() < probability) {
                        weightsAfterDropout.get(i).get(a).set(q, weights.get(i).get(a).get(q));
                    }
                    else {
                        weightsAfterDropout.get(i).get(a).set(q, 0d);
                    }
                }
            }
        }
    }

    //tests on dataset, returns percentage accurate
    public double test(ArrayList<ArrayList<Double>> in, ArrayList<ArrayList<Double>> out) throws Exception{
        int numCorrect = 0;
        for (int i = 0; i < in.size(); i++) {
            ArrayList<Double> array = predictOutput(in.get(i));
            int prediction = 0;

            for (int p = 0; p < array.size(); p++) {
                if (array.get(p) > array.get(prediction)) {
                    prediction = p;
                }
            }

            int output = 0;
            for (int p = 0; p < array.size(); p++) {
                if (out.get(i).get(p) > out.get(i).get(output)) {
                    output = p;
                }
            }

            if (prediction == output) {
                numCorrect++;
            }
        }
        return numCorrect / (double) in.size();
    }


    //NOTE: convenience methods not central to the neural network


    //output as highest probability classname
    public String getPredictionClass(ArrayList<Double> in) throws Exception{
        ArrayList<Double> out = predictOutput(in);

        int maxOutput = 0;
        for (int i = 0; i < out.size(); i++) {
            if (out.get(i) > out.get(maxOutput)) {
                maxOutput = i;
            }
        }
        return classNames.get(maxOutput);
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

    public void setLearningRateAutomatically(ArrayList<ArrayList<Double>> trainingDataOutputs) {
        double maxOutput = -9999;
        double minOutput = 9999;
        for (int i = 0; i < 100; i++) {
            for (int p = 0; p < trainingDataOutputs.get(i).size(); p++) {
                if (trainingDataOutputs.get(i).get(p) > maxOutput) {
                    maxOutput = trainingDataOutputs.get(i).get(p);
                } else if (trainingDataOutputs.get(i).get(p) < minOutput) {
                    minOutput = trainingDataOutputs.get(i).get(p);
                }
            }
        }
        double range = maxOutput - minOutput;
        learningRate = range / nodesPerLayer;
    }

    public double derivativeOfWeightCheck(ArrayList<Double> in, ArrayList<Double> out, int layer, int node, int previousLayerNode) throws Exception{
        double interval = .000001;
        weights.get(layer).get(node).set(previousLayerNode, weights.get(layer).get(node).get(previousLayerNode) - interval);
        double loss = Util.meanSquaredError(predictOutput(in), out);
        weights.get(layer).get(node).set(previousLayerNode, weights.get(layer).get(node).get(previousLayerNode) + interval * 2);
        double output = (Util.meanSquaredError(predictOutput(in), out) - loss) / interval / 2;
        weights.get(layer).get(node).set(previousLayerNode, weights.get(layer).get(node).get(previousLayerNode) - interval);
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
        public ArrayList<Double> image;
        ArrayList<Double> derivativeErrorWithRespectToInputs;

        double learningRate = 1;
        double max = -9999;
        double min = 9999;

        public DeepDream(FullyConnectedNetwork network, ArrayList<Double> image, double learningRate) {
            this.network = network;
            this.image = Util.makeCopy1D(image);
            this.learningRate = learningRate;


            for(int i = 0; i < image.size(); i++) {
                if(image.get(i) > max) {
                    max = image.get(i);
                }
                if(image.get(i) < min) {
                    min = image.get(i);
                }
            }
        }

        public void updateImage(int desiredOutput) throws  Exception{

            derivativeErrorWithRespectToInputs = new ArrayList<Double>();
            for(int i = 0; i < image.size(); i++) {
                derivativeErrorWithRespectToInputs.add(0d);
            }

            ArrayList<Double> outputs = new ArrayList<Double>();
            for (int i = 0; i < network.numOutputs; i++) {
                if (i == desiredOutput) {
                    outputs.add(1d);
                } else {
                    outputs.add(0d);
                }
            }
            ArrayList<ArrayList<Double>> inputsToNetwork = new ArrayList<ArrayList<Double>>();
            inputsToNetwork.add(image);

            ArrayList<ArrayList<Double>> outputsOfNetwork = new ArrayList<ArrayList<Double>>();
            outputsOfNetwork.add(outputs);


            network.getDerivativeOfErrorWithRespectToWeights(inputsToNetwork, outputsOfNetwork);

            for (int i = 0; i < network.derivativesErrorWithRespectToInputsToActivation.size(); i++) {
                for (int u = 0; u < network.numInputs; u++) {
                    derivativeErrorWithRespectToInputs.set(u, derivativeErrorWithRespectToInputs.get(u) + network.derivativesErrorWithRespectToInputsToActivation.get(1).get(i) * network.weights.get(1).get(i).get(u));
                }
            }

            System.out.println("hi");
            for(int i = 0; i < derivativeErrorWithRespectToInputs.size(); i++) {
                image.set(i, image.get(i) - derivativeErrorWithRespectToInputs.get(i) * learningRate);
                if(image.get(i) < min) {
                    image.set(i, min);
                }
                if(image.get(i) > max) {
                    image.set(i, max);
                }
            }
        }
    }


}
