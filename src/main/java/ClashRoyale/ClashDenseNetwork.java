package ClashRoyale;

import Network.DenseNetwork;
import Util.ArrOperations;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;


public class ClashDenseNetwork extends DenseNetwork {

    float learningRateRegression;

    public ClashDenseNetwork(int[] nodesPerLayer, float[] learningRate, float learningRateRegression, int updateRule, float momentum, float dropoutProbability) {
        super(nodesPerLayer, learningRate, updateRule, momentum, dropoutProbability);
        this.learningRateRegression = learningRateRegression;
    }

    public void gradientDescent() {

        for (int i = 1; i < weights.length; i++) {
            for (int t = 0; t < weights[i].length; t++) {
                for (int y = 0; y < weights[i][t].length; y++) {
                    float rate;
                    if(i == weights.length - 1 && t == 3 || t == 4) {
                        rate = learningRateRegression;
                    }
                    else {
                        rate = learningRate[i];
                    }

                    weightMomentumUpdate[i][t][y] = weightMomentumUpdate[i][t][y] * momentum + (1 - momentum) * rate * derivativesErrorWithRespectToWeights[i][t][y];
                    weights[i][t][y] = weights[i][t][y] - weightMomentumUpdate[i][t][y];

                    biases[i][t] = biases[i][t] - derivativesErrorWithRespectToBiases[i][t] * rate * .01f;
                }
            }
        }

        //ensure that if dropout isnt set weightsAfterDropout is still updated - weightsAfterDropout is used for forwardPass
        weightsAfterDropout = weights;
    }

      /*File name format:
    All in the same folder
        notClick,placeCard,selectCard,xCoord,yCoord,select1,select2,select3,select4
        File number separated by a "-"
     */

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

            if (outputs[0][0] == 1) {
                for (int i = 3; i < outputs[0].length; i++) {
                    outputsInAllLayers[layers - 1][i] = outputs[0][i];
                }
            } else if (outputs[0][1] == 1) {
                for (int i = 5; i < outputs[0].length; i++) {
                    outputsInAllLayers[layers - 1][i] = outputs[0][i];
                }
            } else if (outputs[0][2] == 1) {
                outputsInAllLayers[layers - 1][3] = outputs[0][3];
                outputsInAllLayers[layers - 1][4] = outputs[0][4];
            }

            for (int p = layers - 1; p > 0; p--) {
                if (p == layers - 1) {

                    float[] temp1 = ArrOperations.getDerivativeFromSoftmax(outputsInAllLayers[p], ArrOperations.getDerivativeFromMSE(outputs[w], outputsInAllLayers[layers - 1]), 0, 3);
                    float[] temp2 = ArrOperations.getDerivativeFromSoftmax(outputsInAllLayers[p], ArrOperations.getDerivativeFromMSE(outputs[w], outputsInAllLayers[layers - 1]), 5, outputs[0].length);
                    for (int j = 0; j < 3; j++) {
                        derivativesErrorWithRespectToInputsToActivation[p][j] = temp1[j];
                    }
                    for (int j = 5; j < outputs[0].length; j++) {
                        derivativesErrorWithRespectToInputsToActivation[p][j] = temp2[j];
                    }
                    derivativesErrorWithRespectToInputsToActivation[p][3] = ArrOperations.getDerivativeFromMSE(outputs[w], outputsInAllLayers[layers - 1])[3];
                    derivativesErrorWithRespectToInputsToActivation[p][4] = ArrOperations.getDerivativeFromMSE(outputs[w], outputsInAllLayers[layers - 1])[4];

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

                float[] temp1 = ArrOperations.softmax(temp, 0, 3);
                float[] temp2 = ArrOperations.softmax(temp, 5, temp.length);
                for (int j = 0; j < 3; j++) {
                    temp[j] = temp1[j];
                }
                for (int j = 5; j < temp.length; j++) {
                    temp[j] = temp2[j];
                }


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

                float[] temp1 = ArrOperations.softmax(temp, 0, 3);
                float[] temp2 = ArrOperations.softmax(temp, 5, temp.length);
                for (int j = 0; j < 3; j++) {
                    temp[j] = temp1[j];
                }
                for (int j = 5; j < temp.length; j++) {
                    temp[j] = temp2[j];
                }
                in = temp;

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

    public float testLoss(float[][] in, float[][] out) throws Exception {
        int loss = 0;
        for (int a = 0; a < out.length; a++) {
            float[] prediction = predictOutput(in[a]);

            if (out[a][0] == 1) {
                for (int i = 3; i < out[a].length; i++) {
                    prediction[i] = out[a][i];
                }
            } else if (out[a][1] == 1) {
                for (int i = 5; i < out[a].length; i++) {
                    prediction[i] = out[a][i];
                }
            } else if (out[a][2] == 1) {
                prediction[3] = out[a][3];
                prediction[4] = out[a][4];
            }

            loss += ArrOperations.meanSquaredError(prediction, out[a]);
        }
        return loss / (float) in.length;
    }


}
