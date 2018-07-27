package Util;

import Kernels.MatrixVectorKernel;
import com.aparapi.Range;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Random;

public class ArrOperations {

    static float e = (float) Math.E;
    static float pi = (float) Math.PI;


    public static float[] matrixVectorProduct(float[] vector, float[][] matrix) {

        int groupSize = 2;
        int chunkSize = (int) Math.ceil((float) vector.length / groupSize);

//        KernelManager.setKernelManager(new JTPKernelManager());
//        Device device = KernelManager.instance().bestDevice();
//        MatrixVectorKernel kernel = new MatrixVectorKernel(matrix, vector, chunkSize);
//        kernel.execute(device.createRange(groupSize * matrix.length, groupSize));


        MatrixVectorKernel kernel = new MatrixVectorKernel(matrix, vector, chunkSize);
        kernel.execute(Range.create(groupSize * matrix.length, groupSize));

        float[] out = new float[matrix.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = kernel.in1[i][0];
        }

        return out;
    }

    //elementwise product
    public static float[][][] matrixMatrixProduct (float[][][] matrix, float matrix1[][][]) {
        float[][][] out = new float[matrix.length][][];
        for(int i = 0; i < out.length; i++) {
            out[i] = new float[matrix[i].length][];
            for(int u = 0; u < out[i].length; u++) {
                out[i][u] = new float[matrix[i][u].length];
                for(int e = 0; e < out[i][u].length; e++) {
                    out[i][u][e] = matrix[i][u][e] * matrix1[i][u][e];
                }
            }
        }
        return out;
    }


    public static float dotProductNoGPU(float[] in1, float[] in2) {
        float sum = 0;
        for (int i = 0; i < in1.length; i++) {
            sum += in1[i] * in2[i];
        }
        return sum;
    }


    public static float matrixProductSum(float[][][] in, float[][][] anotherIn) {

        float out = 0;
        for (int i = 0; i < in.length; i++) {

            for (int n = 0; n < in[i].length; n++) {
                for (int q = 0; q < in[i][n].length; q++) {
                    out += in[i][n][q] * anotherIn[i][n][q];
                }
            }
        }
        return out;
    }


    public static float[][] matrixScalarProduct(float[][] matrix, float scalar) {
        float[][] out = new float[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            out[i] = new float[matrix[i].length];
            for (int p = 0; p < matrix[i].length; p++) {
                out[i][p] = matrix[i][p] * scalar;
            }
        }
        return out;
    }


    public static float[] vectorScalarProduct(float[] vector, float scalar) {
        float[] output = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            output[i] = vector[i] * scalar;
        }
        return output;
    }


    public static float sigmoidFunction(float in) {
        return 2 / (float) (1 + Math.pow(e, -in)) - 1;
    }

    public static float getDerivativeFromSigmoid(float y) {
        float out = (y + 1) * (1 - y) / 2;
        return out;
    }

    public static float[] getDerivativeFromSigmoid(float[] y) {
        float[] out = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = ((y[i] + 1) * (1 - y[i]) / 2);
        }
        return out;
    }

    public static float[][][] getDerivativeFromSigmoid(float[][][] y) {
        float[][][] out = new float[y.length][][];
        for (int i = 0; i < y.length; i++) {
            out[i] = new float[y[i].length][];
            for (int u = 0; u < y[i].length; u++) {
                out[i][u] = new float[y[i][u].length];
                for (int a = 0; a < y[i][u].length; a++) {
                    out[i][u][a] = (y[i][u][a] + 1) * (1 - y[i][u][a]) / 2;
                }
            }
        }
        return out;
    }

    public static float meanSquaredError(float[] prediction, float[] trueClass) {
        float sum = 0;
        for (int i = 0; i < prediction.length; i++) {
            sum += Math.pow(prediction[i] - trueClass[i], 2);
        }
        return sum / prediction.length;
    }

    //mean squared error derivative
    public static float getDerivativeFromMSE(float trueOutput, float predictedOutput) {
        return (predictedOutput - trueOutput) * 2;
    }

    public static float[] getDerivativeFromMSE(float[] trueOutput, float[] predictedOutput) {
        float[] derivatives = new float[trueOutput.length];
        for (int i = 0; i < trueOutput.length; i++) {
            derivatives[i] = (predictedOutput[i] - trueOutput[i]) * 2;
        }
        return derivatives;
    }

    public static float[] softmax(float[] in) {
        float sum = 0;
        for (int i = 0; i < in.length; i++) {
            sum += Math.pow(e, in[i]);
        }
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = (float) Math.pow(e, in[i]) / sum;
        }
        return out;
    }

    public static float[] getDerivativeFromSoftmax(float[] outputsOfSoftmax, float[] derivativeErrorWithRespectToOutputsOfSoftmax) {
        float[] out = new float[outputsOfSoftmax.length];
        for (int i = 0; i < outputsOfSoftmax.length; i++) {
            out[i] = 0f;
        }

        for (int i = 0; i < outputsOfSoftmax.length; i++) {
            for (int j = 0; j < outputsOfSoftmax.length; j++) {
                if (j == i) {
                    out[j] = out[j] + outputsOfSoftmax[i] * (1 - outputsOfSoftmax[j]) * derivativeErrorWithRespectToOutputsOfSoftmax[i];
                } else {
                    out[j] = out[j] - outputsOfSoftmax[i] * outputsOfSoftmax[j] * derivativeErrorWithRespectToOutputsOfSoftmax[i];
                }
            }
        }
        return out;
    }

    //TODO: keeps parameters to use for 1Dto3D
    static int width;
    static int length;
    static int depth;

    public static float[] convert3Dto1D(float[][][] in) {
        width = in.length;
        length = in[0].length;
        depth = in[0][0].length;
        float[] out = new float[in.length * in[0].length * in[0][0].length];
        for (int i = 0; i < in.length; i++) {
            for (int y = 0; y < in[i].length; y++) {
                for (int b = 0; b < in[i][y].length; b++) {
                    out[i * y * b + y * b + b] = in[i][y][b];
                }
            }
        }
        return out;
    }

    public static float[][][] convert1Dto3D(float[] in, int width, int length, int depth) {
        float[][][] out = new float[width][][];
        for (int i = 0; i < width; i++) {
            out[i] = new float[length][];
            for (int u = 0; u < length; u++) {
                out[i][u] = new float[depth];
                for (int a = 0; a < depth; a++) {
                    out[i][u][a] = in[i * length * depth + u * depth + a];
                }
            }
        }
        return out;
    }


    public static float[][][] convert1Dto3D(float[] in) {
        float[][][] out = new float[width][][];
        for (int i = 0; i < width; i++) {
            out[i] = new float[length][];
            for (int u = 0; u < length; u++) {
                out[i][u] = new float[depth];
                for (int a = 0; a < depth; a++) {
                    out[i][u][a] = in[i * length * depth + u * depth + a];
                }
            }
        }
        return out;
    }

    //padding is on both sides - final dimension = initial + padding
    //priority is on the left and top sides
    public static float[][][] pad(float[][][] in, int[] padding) {
        float[][][] out = new float[in.length][][];
        for (int i = 0; i < in.length; i++) {
            out[i] = new float[in[i].length + padding[0]][];
            for (int a = 0; a < out[i].length; a++) {
                out[i][a] = new float[in[i][0].length + padding[1]];
            }
        }

        for (int i = 0; i < out.length; i++) {
            for (int a = 0; a < in[i].length; a++) {
                for (int p = 0; p < in[i][a].length; p++) {
                    out[i][a + (int) Math.ceil((double) padding[0] / 2d)][p + (int) Math.ceil((double) padding[1] / 2d)] = in[i][a][p];
                }
            }
        }
        return in;
    }

    public static float[][][] unpad(float[][][] in, int[] padding) {
        float[][][] out = new float[in.length][][];

        for (int i = 0; i < in.length; i++) {
            out[i] = new float[in[i].length - padding[0]][];
            for (int e = 0; e < out[i].length; e++) {
                out[i][e] = new float[in[i][e].length - padding[1]];
                for (int u = 0; u < out[i][e].length; u++) {
                    out[i][e][u] = in[i][e + (int) Math.ceil((double) padding[0] / 2d)][u + (int) Math.ceil((double) padding[1] / 2d)];
                }
            }
        }
        return in;
    }


    public static float returnMax(float[][] in) {
        float max = -9999;
        for (int i = 0; i < in.length; i++) {
            for (int r = 0; r < in[i].length; r++) {
                if (in[i][r] > max) {
                    max = in[i][r];
                }
            }
        }
        return max;
    }

    public static BufferedImage convert1dArrayToImage(float[] in, int width, int height) {
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < width; i++) {
            for (int u = 0; u < height; u++) {

                Color color = new Color((int) in[i * height * 3 + u * 3], (int) in[i * height * 3 + u * 3 + 1], (int) in[i * height * 3 + u * 3 + 2]);
                out.setRGB(i, u, color.getRGB());
            }
        }

        return out;
    }

    public static float[][] makeCopy(float[][] in) {
        float[][] out = new float[in.length][];
        for (int i = 0; i < in.length; i++) {
            for (int p = 0; p < in[i].length; p++) {
                out[i][p] = in[i][p];
            }
        }
        return out;
    }

    public static float[] makeCopy(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = in[i];
        }
        return out;
    }

    public static BufferedImage resizeImage(BufferedImage image, int newWidth) {
        int chunkSize = image.getWidth() / newWidth;
        BufferedImage out = new BufferedImage(newWidth, image.getHeight() / chunkSize, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < image.getWidth(); i += chunkSize) {
            for (int u = 0; u < image.getHeight(); u += chunkSize) {
                if (i + chunkSize > image.getWidth()) {
                    return out;
                } else if (u + chunkSize > image.getHeight()) {
                    break;
                }

                int averageR = 0;
                int averageG = 0;
                int averageB = 0;
                for (int p = 0; p < chunkSize; p++) {
                    for (int a = 0; a < chunkSize; a++) {
                        Color color = new Color(image.getRGB(p + i, a + u));
                        averageB += color.getBlue();
                        averageG += color.getGreen();
                        averageR += color.getRed();
                    }
                }
                Color outColor = new Color(averageR / chunkSize / chunkSize, averageG / chunkSize / chunkSize, averageB / chunkSize / chunkSize);
                out.setRGB(i / chunkSize, u / chunkSize, outColor.getRGB());
            }
        }
        return out;
    }

    public static float gaussianRandomVariable(float standardDeviation, float center) {
        Random rand = new Random();
        return standardDeviation * (float) (Math.sqrt(-2 * Math.log(rand.nextFloat())) * Math.cos(2 * pi * rand.nextFloat())) + center;
    }
}
