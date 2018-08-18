package Network;

import Util.ArrOperations;

public class AutoEncoder extends DenseNetwork{

    float noise;

    public AutoEncoder(int[] nodesPerLayer, float[] learningRate, float momentum) {
        super(nodesPerLayer, learningRate, DenseNetwork.UPDATE_MOMENTUM, momentum, 1);
    }

    public void setNoise(float in) {
        noise = in;
    }

    public float[] addNoise(float[] in) {

        float[] out = new float[in.length];
        for(int i = 0 ; i < in.length; i++) {
            if(rand.nextFloat() < noise) {
                out[i] = 0;
            }
            else {
                out[i] = in[i];
            }
        }

        return out;
    }


    //todo: gradients are 10 times larger than actually in last layer
    public void getDerivativeOfErrorWithRespectToWeights(float[][] inputs) throws Exception {
        //no need to worry about dropped out weights because they automatically have 0 derivative so they arent updated in gradient descent
        setDropout();
        setUpdateRule();

        derivativesErrorWithRespectToWeights = getNewDerivativeWeights();
        derivativesErrorWithRespectToBiases = getNewDerivativeBiasesAndOutputs();
        for (int w = 0; w < inputs.length; w++) {
            derivativesErrorWithRespectToInputsToActivation = getNewDerivativeBiasesAndOutputs();
            forwardPass(addNoise(inputs[w]));
            for (int p = layers - 1; p > 0; p--) {
                if (p == layers - 1) {
                    derivativesErrorWithRespectToInputsToActivation[p] = ArrOperations.getDerivativeFromSigmoid(outputsInAllLayers[p]);
                    float[] temp = ArrOperations.getDerivativeFromMSE(inputs[w], outputsInAllLayers[p]);
                    for(int d = 0; d < derivativesErrorWithRespectToInputsToActivation[p].length; d++) {
                        derivativesErrorWithRespectToInputsToActivation[p][d] *= temp[d];
                    }
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

    @Override
    //forward pass - first thing
    //stores outputs of each neuron in hidden layers
    public void forwardPass(float[] inputs) throws Exception {
        float[] in = inputs;
        float[][] outputsInHiddenLayersTemp = new float[layers][];

        for (int i = 0; i < layers; i++) {

            if (i == 0) {
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


    @Override
    public float[] predictOutput(float[] inputs) throws Exception {
        float[] in = inputs;
        for (int i = 0; i < layers; i++) {

            if (i == 0) {
                //output is input in first layer
            } else {
                float[] temp = new float[nodesPerLayer[i]];
                for (int u = 0; u < nodesPerLayer[i]; u++) {
                    temp[u] = ArrOperations.sigmoidFunction(ArrOperations.dotProductNoGPU(weightsAfterDropout[i][u], in) + biases[i][u]);
                }
                in = temp;
            }
        }

        return in;
    }

}
