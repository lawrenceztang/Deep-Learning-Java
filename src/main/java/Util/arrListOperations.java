package Util;

import Kernels.DotProductKernel;
import Kernels.MatrixVectorKernel;
import Kernels.TestKernel;
import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.JavaDevice;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import com.sun.org.apache.xpath.internal.operations.Mult;


import java.awt.*;
import java.awt.image.BufferedImage;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.*;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class arrListOperations {

    static double e = 2.71828;


    public static double dotProductNoGPU(ArrayList<Double> in1, ArrayList<Double> in2) {
        double sum = 0;
        for (int i = 0; i < in1.size(); i++) {
            sum += in1.get(i) * in2.get(i);
        }
        return sum;
    }



    public static double matrixProductSum(ArrayList<ArrayList<ArrayList<Double>>> in, ArrayList<ArrayList<ArrayList<Double>>> anotherIn) {

        double out = 0;
        for (int i = 0; i < in.size(); i++) {

            for (int n = 0; n < in.get(i).size(); n++) {
                for (int q = 0; q < in.get(i).get(n).size(); q++) {
                    out += in.get(i).get(n).get(q) * anotherIn.get(i).get(n).get(q);
                }
            }
        }
        return out;
    }



    public static ArrayList<ArrayList<Double>> matrixScalarProduct(ArrayList<ArrayList<Double>> matrix, double scalar) {
        ArrayList<ArrayList<Double>> out = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < matrix.size(); i++) {
            out.add(new ArrayList<Double>());
            for (int p = 0; p < matrix.get(i).size(); p++) {
                out.get(i).add(matrix.get(i).get(p) * scalar);
            }
        }
        return out;
    }


    public static ArrayList<Double> vectorScalarProduct(ArrayList<Double> vector, double scalar) {
        ArrayList<Double> output = new ArrayList<Double>();
        for (int i = 0; i < vector.size(); i++) {
            output.add(vector.get(i) * scalar);
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

    public static ArrayList<Double> getDerivativeFromSigmoid(ArrayList<Double> y) {
        ArrayList<Double> out = new ArrayList<Double>();
        for (int i = 0; i < y.size(); i++) {
            out.add((y.get(i) + 1) * (1 - y.get(i)) / 2);
        }
        return out;
    }

    public static ArrayList<ArrayList<ArrayList<Double>>> getDerivativeFromSigmoid3d(ArrayList<ArrayList<ArrayList<Double>>> in, ArrayList<ArrayList<ArrayList<Double>>> y) {
        ArrayList<ArrayList<ArrayList<Double>>> out = new ArrayList<ArrayList<ArrayList<Double>>>();
        for(int i = 0; i < in.size(); i++) {
            out.add(new ArrayList<ArrayList<Double>>());
            for(int u = 0; u < in.get(i).size(); u++) {
                out.get(i).add(new ArrayList<Double>());
                for(int a = 0; a < in.get(i).get(u).size(); a++) {
                    out.get(i).get(u).add((y.get(i).get(u).get(a) + 1) * (1 - y.get(i).get(u).get(a)) / 2);
                }
            }
        }
        return out;
    }

    public static double meanSquaredError(ArrayList<Double> prediction, ArrayList<Double> trueClass) {
        double sum = 0;
        for (int i = 0; i < prediction.size(); i++) {
            sum += Math.pow(prediction.get(i) - trueClass.get(i), 2);
        }
        return sum / prediction.size();
    }

    //mean squared error derivative
    public static double getDerivativeFromMSE(double trueOutput, double predictedOutput) {
        return (predictedOutput - trueOutput) * 2;
    }

    public static ArrayList<Double> getDerivativeFromMSE(ArrayList<Double> trueOutput, ArrayList<Double> predictedOutput) {
        ArrayList<Double> derivatives = new ArrayList<Double>();
        for (int i = 0; i < trueOutput.size(); i++) {
            derivatives.add((predictedOutput.get(i) - trueOutput.get(i)) * 2);
        }
        return derivatives;
    }

    public static ArrayList<Double> softmax(ArrayList<Double> in) {
        double sum = 0;
        for (int i = 0; i < in.size(); i++) {
            sum += Math.pow(e, in.get(i));
        }
        ArrayList<Double> out = new ArrayList<Double>();
        for (int i = 0; i < in.size(); i++) {
            out.add(Math.pow(e, in.get(i)) / sum);
        }
        return out;
    }

    public static ArrayList<Double> getDerivativeFromSoftmax(ArrayList<Double> outputsOfSoftmax, ArrayList<Double> derivativeErrorWithRespectToOutputsOfSoftmax) {
        ArrayList<Double> out = new ArrayList<Double>();
        for (int i = 0; i < outputsOfSoftmax.size(); i++) {
            out.add(0d);
        }

        for (int i = 0; i < outputsOfSoftmax.size(); i++) {
            for (int j = 0; j < outputsOfSoftmax.size(); j++) {
                if (j == i) {
                    out.set(j, out.get(j) + outputsOfSoftmax.get(i) * (1 - outputsOfSoftmax.get(j)) * derivativeErrorWithRespectToOutputsOfSoftmax.get(i));
                } else {
                    out.set(j, out.get(j) - outputsOfSoftmax.get(i) * outputsOfSoftmax.get(j) * derivativeErrorWithRespectToOutputsOfSoftmax.get(i));
                }
            }
        }
        return out;
    }

    //TODO: keeps parameters to use for 1Dto3D
    static int width;
    static int length;
    static int depth;

    public static ArrayList<Double> convert3Dto1D(ArrayList<ArrayList<ArrayList<Double>>> in) {
        width = in.size();
        length = in.get(0).size();
        depth = in.get(0).get(0).size();
        ArrayList<Double> out = new ArrayList<Double>();
        for (int i = 0; i < in.size(); i++) {
            for (int y = 0; y < in.get(i).size(); y++) {
                for (int b = 0; b < in.get(i).get(y).size(); b++) {
                    out.add(in.get(i).get(y).get(b));
                }
            }
        }
        return out;
    }

    public static ArrayList<ArrayList<ArrayList<Double>>> convert1Dto3D(ArrayList<Double> in, int width, int length, int depth) {
        ArrayList<ArrayList<ArrayList<Double>>> out = new ArrayList<ArrayList<ArrayList<Double>>>();
        for (int i = 0; i < width; i++) {
            out.add(new ArrayList<ArrayList<Double>>());
            for (int u = 0; u < length; u++) {
                out.get(i).add(new ArrayList<Double>());
                for (int a = 0; a < depth; a++) {
                    out.get(i).get(u).add(in.get(i * length * depth + u * depth + a));
                }
            }
        }
        return out;
    }


    public static ArrayList<ArrayList<ArrayList<Double>>> convert1Dto3D(ArrayList<Double> in) {
        ArrayList<ArrayList<ArrayList<Double>>> out = new ArrayList<ArrayList<ArrayList<Double>>>();
        for (int i = 0; i < width; i++) {
            out.add(new ArrayList<ArrayList<Double>>());
            for (int u = 0; u < length; u++) {
                out.get(i).add(new ArrayList<Double>());
                for (int a = 0; a < depth; a++) {
                    out.get(i).get(u).add(in.get(i * length * depth + u * depth + a));
                }
            }
        }
        return out;
    }

    public static ArrayList<ArrayList<ArrayList<Double>>> pad(ArrayList<ArrayList<ArrayList<Double>>> in, double padding) {
        for (int i = 0; i < in.size(); i++) {
            for (int u = 0; u < padding; u++) {
                for (int a = 0; a < in.get(i).size(); a++) {

                    in.get(i).get(a).add(0d);
                    in.get(i).get(a).add(0, 0d);
                }
            }
            ArrayList<Double> zeroes = new ArrayList<Double>();
            for (int t = 0; t < padding; t++) {

                for (int z = 0; z < in.get(i).get(1).size(); z++) {
                    zeroes.add(0d);
                }
                in.get(i).add(zeroes);
                in.get(i).add(0, zeroes);
            }

        }
        return in;
    }

    public static ArrayList<ArrayList<ArrayList<Double>>> unpad(ArrayList<ArrayList<ArrayList<Double>>> in, int padding) {
        for (int i = 0; i < in.size(); i++) {
            for (int e = 0; e < padding; e++) {
                in.get(i).remove(0);
                in.get(i).remove(in.get(i).size() - 1);
            }

            for (int v = 0; v < in.get(i).size(); v++) {
                for (int c = 0; c < padding; c++) {
                    in.get(i).get(v).remove(0);
                    in.get(i).get(v).remove(in.get(i).get(v).size() - 1);
                }
            }
        }
        return in;
    }


    public static double returnMax(ArrayList<ArrayList<Double>> in) {
        double max = -9999;
        for (int i = 0; i < in.size(); i++) {
            for (int r = 0; r < in.get(i).size(); r++) {
                if (in.get(i).get(r) > max) {
                    max = in.get(i).get(r);
                }
            }
        }
        return max;
    }

    //doesnt work
    public static BufferedImage convert1dArrayToImage(ArrayList<Double> in, int width, int height) {
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < width; i++) {
            for (int u = 0; u < height; u++) {
                try {
                    Color color = new Color(in.get(i * height * 3 + u * 3).intValue(), in.get(i * height * 3 + u * 3 + 1).intValue(), in.get(i * height * 3 + u * 3 + 2).intValue());
                    out.setRGB(i, u, color.getRGB());
                } catch (Exception e) {

                }

            }
        }

        return out;
    }

    public static ArrayList<ArrayList<Double>> makeCopy3D(ArrayList<ArrayList<Double>> in) {
        ArrayList<ArrayList<Double>> out = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < in.size(); i++) {
            for (int p = 0; p < in.get(i).size(); p++) {
                out.get(i).add((in.get(i).get(p)));
            }
        }
        return out;
    }

    public static ArrayList<Double> makeCopy1D(ArrayList<Double> in) {
        ArrayList<Double> out = new ArrayList<Double>();
        for (int i = 0; i < in.size(); i++) {
            out.add(in.get(i));
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

}
