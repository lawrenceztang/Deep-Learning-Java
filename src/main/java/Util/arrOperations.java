package Util;

import Kernels.MatrixVectorKernel;
import com.aparapi.Range;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Random;

public class arrOperations {

    static double e = Math.E;
    static double pi = Math.PI;


    public static double[] matrixVectorProduct(double[] vector, double[][] matrix) {

        int groupSize = 2;
        int chunkSize = (int) Math.ceil((double) vector.length / groupSize);

//        KernelManager.setKernelManager(new JTPKernelManager());
//        Device device = KernelManager.instance().bestDevice();
//        MatrixVectorKernel kernel = new MatrixVectorKernel(matrix, vector, chunkSize);
//        kernel.execute(device.createRange(groupSize * matrix.length, groupSize));


        MatrixVectorKernel kernel = new MatrixVectorKernel(matrix, vector, chunkSize);
        kernel.execute(Range.create(groupSize * matrix.length, groupSize));

        double[] out = new double[matrix.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = kernel.in1[i][0];
        }

        return out;
    }


    public static double dotProductNoGPU(double[] in1, double[] in2) {
        double sum = 0;
        for (int i = 0; i < in1.length; i++) {
            sum += in1[i] * in2[i];
        }
        return sum;
    }


    public static double matrixProductSum(double[][][] in, double[][][] anotherIn) {

        double out = 0;
        for (int i = 0; i < in.length; i++) {

            for (int n = 0; n < in[i].length; n++) {
                for (int q = 0; q < in[i][n].length; q++) {
                    out += in[i][n][q] * anotherIn[i][n][q];
                }
            }
        }
        return out;
    }


    public static double[][] matrixScalarProduct(double[][] matrix, double scalar) {
        double[][] out = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            out[i] = new double[matrix[i].length];
            for (int p = 0; p < matrix[i].length; p++) {
                out[i][p] = matrix[i][p] * scalar;
            }
        }
        return out;
    }


    public static double[] vectorScalarProduct(double[] vector, double scalar) {
        double[] output = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            output[i] = vector[i] * scalar;
        }
        return output;
    }


    public static double sigmoidFunction(double in) {
        return 2 / (1 + Math.pow(e, -in)) - 1;
    }

    public static double getDerivativeFromSigmoid(double y) {
        double out = (y + 1) * (1 - y) / 2;
        return out;
    }

    public static double[] getDerivativeFromSigmoid(double[] y) {
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = ((y[i] + 1) * (1 - y[i]) / 2);
        }
        return out;
    }

    public static double[][][] getDerivativeFromSigmoid3d(double[][][] in, double[][][] y) {
        double[][][] out = new double[in.length][][];
        for (int i = 0; i < in.length; i++) {
            out[i] = new double[in[i].length][];
            for (int u = 0; u < in[i].length; u++) {
                out[i][u] = new double[in[i][u].length];
                for (int a = 0; a < in[i][u].length; a++) {
                    out[i][u][a] = (y[i][u][a] + 1) * (1 - y[i][u][a]) / 2;
                }
            }
        }
        return out;
    }

    public static double meanSquaredError(double[] prediction, double[] trueClass) {
        double sum = 0;
        for (int i = 0; i < prediction.length; i++) {
            sum += Math.pow(prediction[i] - trueClass[i], 2);
        }
        return sum / prediction.length;
    }

    //mean squared error derivative
    public static double getDerivativeFromMSE(double trueOutput, double predictedOutput) {
        return (predictedOutput - trueOutput) * 2;
    }

    public static double[] getDerivativeFromMSE(double[] trueOutput, double[] predictedOutput) {
        double[] derivatives = new double[trueOutput.length];
        for (int i = 0; i < trueOutput.length; i++) {
            derivatives[i] = (predictedOutput[i] - trueOutput[i]) * 2;
        }
        return derivatives;
    }

    public static double[] softmax(double[] in) {
        double sum = 0;
        for (int i = 0; i < in.length; i++) {
            sum += Math.pow(e, in[i]);
        }
        double[] out = new double[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = Math.pow(e, in[i]) / sum;
        }
        return out;
    }

    public static double[] getDerivativeFromSoftmax(double[] outputsOfSoftmax, double[] derivativeErrorWithRespectToOutputsOfSoftmax) {
        double[] out = new double[outputsOfSoftmax.length];
        for (int i = 0; i < outputsOfSoftmax.length; i++) {
            out[i] = 0d;
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

    public static double[] convert3Dto1D(double[][][] in) {
        width = in.length;
        length = in[0].length;
        depth = in[0][0].length;
        double[] out = new double[in.length * in[0].length * in[0][0].length];
        for (int i = 0; i < in.length; i++) {
            for (int y = 0; y < in[i].length; y++) {
                for (int b = 0; b < in[i][y].length; b++) {
                    out[i * y * b + y * b + b] = in[i][y][b];
                }
            }
        }
        return out;
    }

    public static double[][][] convert1Dto3D(double[] in, int width, int length, int depth) {
        double[][][] out = new double[width][][];
        for (int i = 0; i < width; i++) {
            out[i] = new double[length][];
            for (int u = 0; u < length; u++) {
                out[i][u] = new double[depth];
                for (int a = 0; a < depth; a++) {
                    out[i][u][a] = in[i * length * depth + u * depth + a];
                }
            }
        }
        return out;
    }


    public static double[][][] convert1Dto3D(double[] in) {
        double[][][] out = new double[width][][];
        for (int i = 0; i < width; i++) {
            out[i] = new double[length][];
            for (int u = 0; u < length; u++) {
                out[i][u] = new double[depth];
                for (int a = 0; a < depth; a++) {
                    out[i][u][a] = in[i * length * depth + u * depth + a];
                }
            }
        }
        return out;
    }

    //padding is on both sides - final dimension = initial + padding * 2
    public static double[][][] pad(double[][][] in, int padding) {
        double[][][] out = new double[in.length + padding * 2][][];
        for (int i = 0; i < in.length; i++) {
            for (int a = 0; a < in[i].length; a++) {
                for (int p = 0; p < in[i][a].length; p++) {
                    out[i][a + padding][p + padding] = in[i][a][p];
                }
            }
        }
        return in;
    }

    public static double[][][] unpad(double[][][] in, int padding) {
        double[][][] out = new double[in.length - 2 * padding][][];

        for (int i = 0; i < in.length; i++) {
            for (int e = 0; e < in[i].length; e++) {
                for(int u = 0; u < in[i][e].length; u++) {

                }
            }
        }
        return in;
    }


    public static double returnMax(double[][] in) {
        double max = -9999;
        for (int i = 0; i < in.length; i++) {
            for (int r = 0; r < in[i].length; r++) {
                if (in[i][r] > max) {
                    max = in[i][r];
                }
            }
        }
        return max;
    }

    //doesnt work
    public static BufferedImage convert1dArrayToImage(double[] in, int width, int height) {
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < width; i++) {
            for (int u = 0; u < height; u++) {
                try {
                    Color color = new Color((int) in[i * height * 3 + u * 3], (int) in[i * height * 3 + u * 3 + 1], (int) in[i * height * 3 + u * 3 + 2]);
                    out.setRGB(i, u, color.getRGB());
                } catch (Exception e) {

                }

            }
        }

        return out;
    }

    public static double[][] makeCopy(double[][] in) {
        double[][] out = new double[in.length][];
        for (int i = 0; i < in.length; i++) {
            for (int p = 0; p < in[i].length; p++) {
                out[i][p] = in[i][p];
            }
        }
        return out;
    }

    public static double[] makeCopy(double[] in) {
        double[] out = new double[in.length];
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

    public static double gaussianRandomVariable (double standardDeviation, double center) {
        Random rand = new Random();
        return  standardDeviation * Math.sqrt(-2 * Math.log(rand.nextDouble())) * Math.cos(2 * pi * rand.nextDouble()) + center;
    }
}
