package Network;

import java.util.ArrayList;
import java.util.Random;

import Util.ArrListOperations;

public class ConvNetwork {

    Random rand;

    //constants
    int[] inputSize;

    //hyperparameters
    double[] convLearningRates;
    ArrayList<Integer> filterSizes;
    ArrayList<Integer> strideSizes;
    ArrayList<Integer> numberOfFilters;
    ArrayList<Integer> padding;
    ArrayList<Integer> poolingStride;
    int numLayers;


    //weights: numlayers x numberFilters x inputDepth x filtersize x filtersize

    ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>> weights;
    ArrayList<ArrayList<Double>> biases;
    ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> outputsInLayers;
    ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> derivativeErrorWithRespectToInputToActivation;
    ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>> derivativeErrorWithRespectToWeight;


    DenseNetwork denseNetwork;

    public ConvNetwork(int[] inputSize, double[] convLearningRates, ArrayList<Integer> filterSizes, ArrayList<Integer> strideSizes, ArrayList<Integer> numberOfFilters, ArrayList<Integer> padding, ArrayList<Integer> poolingStride, DenseNetwork denseNetwork) {

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
        derivativeErrorWithRespectToWeight = new ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>>();
        derivativeErrorWithRespectToInputToActivation = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();
        numLayers = filterSizes.size();



    }

    public void gradientDescent() {
        for (int i = 0; i < derivativeErrorWithRespectToWeight.size(); i++) {
            for (int n = 0; n < derivativeErrorWithRespectToWeight.get(i).size(); n++) {
                for (int z = 0; z < derivativeErrorWithRespectToWeight.get(i).get(n).size(); z++) {
                    for (int s = 0; s < derivativeErrorWithRespectToWeight.get(i).get(n).get(z).size(); s++) {
                        for (int f = 0; f < derivativeErrorWithRespectToWeight.get(i).get(n).get(z).get(s).size(); f++) {
                            weights.get(i).get(n).get(z).get(s).set(f, weights.get(i).get(n).get(z).get(s).get(f) - derivativeErrorWithRespectToWeight.get(i).get(n).get(z).get(s).get(f) * convLearningRates[i]);
                        }
                    }
                }
            }
        }
    }

    //i think backprop is correctly done
    public void getGradientsWeightsWithRespectToError(ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> inputs, ArrayList<ArrayList<Double>> outputs) throws Exception {

        forwardPass(inputs.get(0));
        derivativeErrorWithRespectToInputToActivation = initializeActivationInputs();

        ArrayList<ArrayList<Double>> inputToFullyConnected = new ArrayList<ArrayList<Double>>();
        inputToFullyConnected.add(ArrListOperations.convert3Dto1D(outputsInLayers.get(outputsInLayers.size() - 1)));

        denseNetwork.getDerivativeOfErrorWithRespectToWeights(inputToFullyConnected, outputs);

        ArrayList<Double> derivativeErrorWithRespectToInputsToFullyConnected = new ArrayList<Double>();


        for (int u = 0; u < denseNetwork.numInputs; u++) {
            derivativeErrorWithRespectToInputsToFullyConnected.add(0d);
            for (int i = 0; i < denseNetwork.derivativesErrorWithRespectToInputsToActivation.get(1).size(); i++) {
                derivativeErrorWithRespectToInputsToFullyConnected.set(u, derivativeErrorWithRespectToInputsToFullyConnected.get(u) + denseNetwork.derivativesErrorWithRespectToInputsToActivation.get(1).get(i) * denseNetwork.weights.get(1).get(i).get(u));
            }
        }
        //must be put through activation
        derivativeErrorWithRespectToInputToActivation.set(derivativeErrorWithRespectToInputToActivation.size() - 1, ArrListOperations.getDerivativeFromSigmoid3d(ArrListOperations.convert1Dto3D(derivativeErrorWithRespectToInputsToFullyConnected), outputsInLayers.get(outputsInLayers.size() - 1)));


        for (int a = numLayers - 2; a > 0; a--) {

            derivativeErrorWithRespectToInputToActivation.set(a, initializeOutputs(numberOfFilters.get(a), outputsInLayers.get(a).get(0).size(), outputsInLayers.get(a).get(0).get(0).size()));
            for (int b = 0; b < numberOfFilters.get(a + 1); b++) {
                for (int x = 0; x < outputsInLayers.get(a).size(); x++) {
                    for (int h = 0; h <= outputsInLayers.get(a).get(x).size() - filterSizes.get(a + 1); h += strideSizes.get(a + 1)) {
                        for (int q = 0; q <= outputsInLayers.get(a).get(x).get(h).size() - filterSizes.get(a + 1); q += strideSizes.get(a + 1)) {


                            for (int w = 0; w < filterSizes.get(a + 1); w++) {
                                for (int y = 0; y < filterSizes.get(a + 1); y++) {
                                        derivativeErrorWithRespectToInputToActivation.get(a).get(x).get(h + w).set(q + y, derivativeErrorWithRespectToInputToActivation.get(a).get(x).get(h + w).get(q + y) + derivativeErrorWithRespectToInputToActivation.get(a + 1).get(b).get(h / strideSizes.get(a + 1)).get(q / strideSizes.get(a + 1)) * weights.get(a + 1).get(b).get(x).get(w).get(y));
                                }
                            }


                        }
                    }
                }
            }

            derivativeErrorWithRespectToInputToActivation.set(a, getDerivativeFromMaxPooling(derivativeErrorWithRespectToInputToActivation.get(a), outputsInLayers.get(a)));


            for (int u = 0; u < derivativeErrorWithRespectToInputToActivation.get(a).size(); u++) {
                for (int m = 0; m < derivativeErrorWithRespectToInputToActivation.get(a).get(u).size(); m++) {
                    for (int p = 0; p < derivativeErrorWithRespectToInputToActivation.get(a).get(u).get(m).size(); p++) {
                        derivativeErrorWithRespectToInputToActivation.get(a).get(u).get(m).set(p, derivativeErrorWithRespectToInputToActivation.get(a).get(u).get(m).get(p) * ArrListOperations.getDerivativeFromSigmoid(outputsInLayers.get(a).get(u).get(m).get(p)));
                    }
                }
            }

            ArrayList<ArrayList<ArrayList<Double>>> outputAfterPooling = maxPooling(outputsInLayers.get(a - 1), poolingStride.get(a - 1));
            derivativeErrorWithRespectToInputToActivation.set(a, ArrListOperations.unpad(derivativeErrorWithRespectToInputToActivation.get(a), padding.get(a)));

            for (int b = 0; b < derivativeErrorWithRespectToInputToActivation.get(a).size(); b++) {
                for (int h = 0; h < derivativeErrorWithRespectToInputToActivation.get(a).get(b).size(); h++) {
                    for (int q = 0; q < derivativeErrorWithRespectToInputToActivation.get(a).get(b).get(h).size(); q++) {

                        for (int x = 0; x < numberOfFilters.get(a - 1); x++) {
                            for (int w = 0; w < filterSizes.get(a); w++) {
                                for (int y = 0; y < filterSizes.get(a); y++) {
                                    //inputtoactivation
                                    derivativeErrorWithRespectToWeight.get(a).get(b).get(x).get(w).set(y, derivativeErrorWithRespectToWeight.get(a).get(b).get(x).get(w).get(y) + derivativeErrorWithRespectToInputToActivation.get(a).get(b).get(h).get(q) * outputsInLayers.get(a - 1).get(b).get(h + w).get(q + y));
                                }
                            }
                        }
                        try {
                            biases.get(a).set(b, biases.get(a).get(b) - derivativeErrorWithRespectToInputToActivation.get(a).get(b).get(h).get(q));
                        }
                        catch(Exception e) {
                            break;
                        }
                    }
                }
            }

        }
    }

    //working
    public ArrayList<Double> predictOutput(ArrayList<ArrayList<ArrayList<Double>>> inputs) throws Exception {
        ArrayList<ArrayList<ArrayList<Double>>> layerInputs = inputs;
        for (int e = 1; e < numLayers; e++) {
            layerInputs = ArrListOperations.pad(layerInputs, padding.get(e));
            ArrayList<ArrayList<ArrayList<Double>>> layerOutputs = initializeOutputs(numberOfFilters.get(e), (layerInputs.get(0).size() - filterSizes.get(e)) / strideSizes.get(e) + 1, (layerInputs.get(0).get(0).size() - filterSizes.get(e)) / strideSizes.get(e) + 1);
            for (int i = 0; i <= layerInputs.get(0).size() - filterSizes.get(e); i += strideSizes.get(e)) {
                for (int b = 0; b <= layerInputs.get(0).get(i).size() - filterSizes.get(e); b += strideSizes.get(e)) {


                    ArrayList<ArrayList<ArrayList<Double>>> inputToNeuron = new ArrayList<ArrayList<ArrayList<Double>>>();

                    for (int v = 0; v < layerInputs.size(); v++) {
                        inputToNeuron.add(new ArrayList<ArrayList<Double>>());
                        for (int u = 0; u < filterSizes.get(e); u++) {
                            inputToNeuron.get(v).add(new ArrayList<Double>());
                            for (int t = 0; t < filterSizes.get(e); t++) {

                                inputToNeuron.get(v).get(u).add(layerInputs.get(v).get(i + u).get(b + t));
                            }
                        }
                    }
                    for (int l = 0; l < numberOfFilters.get(e); l++) {
                        layerOutputs.get(l).get(i / strideSizes.get(e)).set(b / strideSizes.get(e), ArrListOperations.matrixProductSum(inputToNeuron, weights.get(e).get(l)));
                    }
                }
            }

            layerInputs = maxPooling(layerOutputs, poolingStride.get(e));
        }

        ArrayList<Double> inputToFullyConnected = ArrListOperations.convert3Dto1D(layerInputs);
        return denseNetwork.predictOutput(inputToFullyConnected);
    }

    //input layer is counted as layer 0
    public void forwardPass(ArrayList<ArrayList<ArrayList<Double>>> inputs) throws Exception {
        outputsInLayers = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();

        ArrayList<ArrayList<ArrayList<Double>>> layerInputs = inputs;
        outputsInLayers.add(layerInputs);
        for (int e = 1; e < numLayers; e++) {
            layerInputs = ArrListOperations.pad(layerInputs, padding.get(e));
            ArrayList<ArrayList<ArrayList<Double>>> layerOutputs = initializeOutputs(numberOfFilters.get(e), (layerInputs.get(0).size() - filterSizes.get(e)) / strideSizes.get(e) + 1, (layerInputs.get(0).get(0).size() - filterSizes.get(e)) / strideSizes.get(e) + 1);
            for (int i = 0; i <= layerInputs.get(0).size() - filterSizes.get(e); i += strideSizes.get(e)) {
                for (int b = 0; b <= layerInputs.get(0).get(i).size() - filterSizes.get(e); b += strideSizes.get(e)) {


                    ArrayList<ArrayList<ArrayList<Double>>> inputToNeuron = new ArrayList<ArrayList<ArrayList<Double>>>();

                    for (int v = 0; v < layerInputs.size(); v++) {
                        inputToNeuron.add(new ArrayList<ArrayList<Double>>());
                        for (int u = 0; u < filterSizes.get(e); u++) {
                            inputToNeuron.get(v).add(new ArrayList<Double>());
                            for (int t = 0; t < filterSizes.get(e); t++) {

                                inputToNeuron.get(v).get(u).add(layerInputs.get(v).get(i + u).get(b + t));
                            }
                        }
                    }
                    for (int l = 0; l < numberOfFilters.get(e); l++) {
                        layerOutputs.get(l).get(i / strideSizes.get(e)).set(b / strideSizes.get(e), ArrListOperations.matrixProductSum(inputToNeuron, weights.get(e).get(l)));
                    }
                }
            }
            outputsInLayers.add(layerOutputs);
            layerInputs = maxPooling(layerOutputs, poolingStride.get(e));
        }

    }

    public void initializeWeights() {
        weights = new ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>>();
        biases = new ArrayList<ArrayList<Double>>();
        weights.add(new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>());
        biases.add(new ArrayList<Double>());
        for (int i = 1; i < filterSizes.size(); i++) {
            weights.add(new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>());
            biases.add(new ArrayList<Double>());
            for (int u = 0; u < numberOfFilters.get(i); u++) {
                weights.get(i).add(new ArrayList<ArrayList<ArrayList<Double>>>());
                biases.get(i).add(0d);
                int inputDepth;
                if (i == 1) {
                    inputDepth = inputSize[0];
                } else {
                    inputDepth = numberOfFilters.get(i - 1);
                }
                for (int q = 0; q < inputDepth; q++) {
                    weights.get(i).get(u).add(new ArrayList<ArrayList<Double>>());
                    for (int v = 0; v < filterSizes.get(i); v++) {
                        weights.get(i).get(u).get(q).add(new ArrayList<Double>());
                        for (int c = 0; c < filterSizes.get(i); c++) {
                            weights.get(i).get(u).get(q).get(v).add(rand.nextDouble() * .01 - .005);
                        }
                    }
                }
            }
        }
        return;
    }

    public ArrayList<ArrayList<ArrayList<Double>>> initializeOutputs(int numFilters, int width, int length) {
        ArrayList<ArrayList<ArrayList<Double>>> outputs = new ArrayList<ArrayList<ArrayList<Double>>>();
        for (int i = 0; i < numFilters; i++) {
            outputs.add(new ArrayList<ArrayList<Double>>());
            for (int d = 0; d < width; d++) {
                outputs.get(i).add(new ArrayList<Double>());
                for (int f = 0; f < length; f++) {
                    outputs.get(i).get(d).add(0d);
                }
            }
        }
        return outputs;
    }

    public ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> initializeActivationInputs() {

        ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> out = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();
        for (int i = 0; i < outputsInLayers.size(); i++) {
            out.add(new ArrayList<ArrayList<ArrayList<Double>>>());
            for (int u = 0; u < outputsInLayers.get(i).size(); u++) {
                out.get(i).add(new ArrayList<ArrayList<Double>>());
                for (int y = 0; y < outputsInLayers.get(i).get(u).size(); y++) {
                    out.get(i).get(u).add(new ArrayList<Double>());
                    for (int a = 0; a < outputsInLayers.get(i).get(u).get(y).size(); a++) {
                        out.get(i).get(u).get(y).add(0d);
                    }
                }
            }
        }
        return out;
    }

    public ArrayList<ArrayList<ArrayList<Double>>> maxPooling(ArrayList<ArrayList<ArrayList<Double>>> in, int stride) {
        if (stride == 0) {
            return in;
        }
        ArrayList<ArrayList<ArrayList<Double>>> out = new ArrayList<ArrayList<ArrayList<Double>>>();
        for (int u = 0; u < in.size(); u++) {
            out.add(new ArrayList<ArrayList<Double>>());

            for (int a = 0; a < in.get(u).size(); a += stride) {
                out.get(u).add(new ArrayList<Double>());
                for (int q = 0; q < in.get(u).get(a).size(); q += stride) {

                    double max = -9999;
                    for (int p = 0; p < stride; p++) {
                        for (int y = 0; y < stride; y++) {
                            if (in.get(u).get(a + p).get(q + y) > max) {
                                max = in.get(u).get(a + p).get(q + y);
                            }
                        }
                    }

                    out.get(u).get(a).add(max);

                }
            }
        }
        return out;
    }

    //not working
    public ArrayList<ArrayList<ArrayList<Double>>> getDerivativeFromMaxPooling(ArrayList<ArrayList<ArrayList<Double>>> derivativeErrorWithRespectToPoolingOutput, ArrayList<ArrayList<ArrayList<Double>>> poolingInput) {

        ArrayList<ArrayList<ArrayList<Double>>> derivativeErrorWithRespectToInputToPooling = new ArrayList<ArrayList<ArrayList<Double>>>();
        int stride = derivativeErrorWithRespectToPoolingOutput.get(0).size() / poolingInput.get(0).size();
        for (int i = 0; i < derivativeErrorWithRespectToPoolingOutput.size(); i++) {
            derivativeErrorWithRespectToInputToPooling.add(new ArrayList<ArrayList<Double>>());
            for (int a = 0; a < derivativeErrorWithRespectToPoolingOutput.get(i).size(); a++) {
                for (int o = 0; o < derivativeErrorWithRespectToPoolingOutput.get(i).get(a).size(); o++) {

                    double max = -99999;
                    for (int c = 0; c < stride; c++) {
                        for (int u = 0; u < stride; u++) {
                            if (poolingInput.get(i).get(a * stride + c).get(o * stride + u) > max) {
                                max = poolingInput.get(i).get(a * stride + c).get(o * stride + u);
                            }
                        }
                    }

                    for (int y = 0; y < stride; y++) {
                        derivativeErrorWithRespectToInputToPooling.get(i).add(new ArrayList<Double>());
                        for (int q = 0; q < stride; q++) {
                            if (max == poolingInput.get(i).get(y + a * stride).get(o * stride + q)) {
                                derivativeErrorWithRespectToInputToPooling.get(i).get(y + a * stride).add(derivativeErrorWithRespectToPoolingOutput.get(i).get(a).get(o));
                            } else {
                                derivativeErrorWithRespectToInputToPooling.get(i).get(y + a * stride).add(0d);
                            }
                        }
                    }

                }
            }
        }
        return derivativeErrorWithRespectToInputToPooling;
    }

    //tests on dataset, returns percentage accurate
    public double test(ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> in, ArrayList<ArrayList<Double>> out) throws Exception {
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


}
