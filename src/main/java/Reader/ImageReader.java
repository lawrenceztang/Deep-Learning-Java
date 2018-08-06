package Reader;

import Util.ArrOperations;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

//reads images in folders with name of class, like MNIST
public class ImageReader implements Serializable {

    String path;
    public String[] classes;
    File file;
    public float[][] oneHotOutputs;
    public int numOutputs;
    Random rand;
    ArrayList<Integer> newImageReferenceForEachClass;
    public File[][] allFiles;
    public File[] allFilesCSVLabel;
    public float mean;
    public float standardDeviation;
    public boolean CSV = false;

    public ImageReader(String path) throws Exception {
        this.path = path;

        file = new File(path);
        rand = new Random();

        File[] files = file.listFiles();

        if (files[0].listFiles() == null) {
            CSV = true;
            numOutputs = getNumCommas(files[0].getPath().substring(files[0].getPath().lastIndexOf('\\'), files[0].getPath().length())) + 1;
        } else {
            classes = new String[files.length];
            for (int i = 0; i < files.length; i++) {

                //get class names from subfolders
                String temp = files[i].toString();
                for (int u = 0; u < temp.length(); u++) {
                    if (temp.charAt(temp.length() - 1 - u) == '\\') {
                        classes[i] = temp.substring(temp.length() - u, temp.length());
                        break;
                    }
                }
            }

            allFiles = new File[files.length][];
            newImageReferenceForEachClass = new ArrayList<Integer>();

            for (int i = 0; i < files.length; i++) {
                allFiles[i] = files[i].listFiles();
                newImageReferenceForEachClass.add(0);
            }
        }
    }

    public ImageReader() {

    }

    public ImageReader(String path, ImageReader reader) throws Exception {
        this(path);
        this.mean = reader.mean;
        this.standardDeviation = reader.standardDeviation;
    }


    public float[][][][] get3dColorMatrices(int batchSize) throws Exception {
        float[][][][] imagesAsMatrices = new float[batchSize][][][];
        oneHotOutputs = new float[batchSize][];

        //get images and save as matrix

        for (int i = 0; i < batchSize; i++) {
            int random = rand.nextInt(allFiles.length);

            if (newImageReferenceForEachClass.get(random) > allFiles[random].length - 1) {
                break;
            }
            imagesAsMatrices[i] = getImageAs3DMatrix(allFiles[random][i].getPath());
            newImageReferenceForEachClass.set(random, newImageReferenceForEachClass.get(random) + 1);

            oneHotOutputs[i] = new float[classes.length];

            for (int t = 0; t < classes.length; t++) {
                if (t == random) {
                    oneHotOutputs[i][t] = 1f;
                } else {
                    oneHotOutputs[i][t] = 0f;
                }
            }
        }

        return imagesAsMatrices;
    }

    public float[][] get1dColorMatricesFromImages(int batchSize, int newWidth) throws Exception {
        float[][] imagesAsMatrices = new float[batchSize][];
        oneHotOutputs = new float[batchSize][];

        if (CSV = true) {
            //CSV format of label - can be used for multiple outputs
            allFilesCSVLabel = file.listFiles();
            for (int a = 0; a < batchSize; a++) {
                int i = rand.nextInt(file.listFiles().length);
                imagesAsMatrices[a] = getImageAs1DMatrix(allFilesCSVLabel[i].getPath(), newWidth);
                oneHotOutputs[a] = ArrOperations.getValuesFromCSVString(allFilesCSVLabel[i].getPath().substring(allFilesCSVLabel[i].getPath().lastIndexOf('\\') + 1, allFilesCSVLabel[i].getPath().length()), numOutputs);
            }

        } else {
            //Folder name is classification

            //get images and save as matrix
            //random chance for each class to be picked, evenly distributed
            for (int i = 0; i < batchSize; i++) {

                int random = rand.nextInt(allFiles.length);

                //if reference is over number of images in class, exit out of loop
                if (newImageReferenceForEachClass.get(random) > allFiles[random].length - 1) {
                    break;
                }

                imagesAsMatrices[i] = getImageAs1DMatrix(allFiles[random][newImageReferenceForEachClass.get(random)].getPath(), newWidth);
                newImageReferenceForEachClass.set(random, newImageReferenceForEachClass.get(random) + 1);

                oneHotOutputs[i] = new float[classes.length];

                for (int t = 0; t < classes.length; t++) {
                    if (t == random) {
                        oneHotOutputs[i][t] = 1f;
                    } else {
                        oneHotOutputs[i][t] = 0f;
                    }
                }

            }
        }
        return imagesAsMatrices;
    }

    //return image as 2d matrix with color
    public float[][][] getImageAs3DMatrix(String filePath) throws Exception {
        BufferedImage image = ImageIO.read(new File(filePath));
        float[][][] matrix = new float[3][][];
        for (int i = 0; i < 3; i++) {
            matrix[i] = new float[image.getWidth()][];
            for (int x = 0; x < image.getWidth(); x++) {
                matrix[i][x] = new float[image.getHeight()];
                for (int y = 0; y < image.getHeight(); y++) {

                    Color color = new Color(image.getRGB(x, y));
                    if (i == 0) {
                        matrix[i][x][y] = (float) color.getRed();
                    } else if (i == 1) {
                        matrix[i][x][y] = (float) color.getGreen();
                    } else {
                        matrix[i][x][y] = (float) color.getBlue();
                    }
                }
            }
        }
        return matrix;
    }

    //return image as 1d matrix with color
    public float[] getImageAs1DMatrix(String filePath, int newWidth) throws Exception {
        BufferedImage image = ImageIO.read(new File(filePath));
        return resizeImage(image, newWidth);
    }

    public float[] resizeImage(BufferedImage image, int newWidth) {
        float[] matrix = new float[image.getWidth() * image.getHeight() * 3];
        int chunkSize = image.getWidth() / newWidth;

        for (int x = 0; x < image.getWidth(); x += chunkSize) {

            for (int y = 0; y < image.getHeight(); y += chunkSize) {
                if (x + chunkSize > image.getWidth()) {
                    return matrix;
                } else if (y + chunkSize > image.getHeight()) {
                    break;
                }

                int averageR = 0;
                int averageG = 0;
                int averageB = 0;
                for (int p = 0; p < chunkSize; p++) {
                    for (int u = 0; u < chunkSize; u++) {
                        Color color = new Color(image.getRGB(x + p, y + u));
                        averageB += color.getBlue();
                        averageG += color.getGreen();
                        averageR += color.getRed();
                    }
                }
                matrix[x / chunkSize * image.getHeight() * 3 + y / chunkSize * 3] = (float) (averageR / chunkSize / chunkSize);
                matrix[x / chunkSize * image.getHeight() * 3 + y / chunkSize * 3 + 1] = (float) (averageG / chunkSize / chunkSize);
                matrix[x / chunkSize * image.getHeight() * 3 + y / chunkSize * 3 + 2] = (float) (averageB / chunkSize / chunkSize);
            }
        }
        return matrix;
    }

    //some preprocessing saved in reader object
    public float[] preprocessExample(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = (in[i] - mean) / standardDeviation;
        }
        return out;
    }

    public float[][] preprocessTrainingSet(float[][] in) {
        float[][] out = new float[in.length][];
        for (int p = 0; p < in.length; p++) {
            out[p] = new float[in[p].length];
            for (int u = 0; u < in[p].length; u++) {
                out[p][u] = (in[p][u] - mean) / standardDeviation;
            }
        }
        return out;
    }

    public void preprocessTrainingSet(float[][][][] in) {
        for (int i = 0; i < in.length; i++) {
            for (int a = 0; a < in[i].length; a++) {
                for (int z = 0; z < in[i][a].length; z++) {
                    for (int u = 0; u < in[i][a][z].length; u++) {
                        in[i][a][z][u] = (in[i][a][u][z] - mean) / standardDeviation;
                    }
                }
            }
        }
    }

    public float[] unpreprocessExample(float[] in) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = in[i] * standardDeviation + mean;
        }
        return out;
    }

    public void setPreprocessParameters(float[][] in) {
        mean = 0;
        for (int i = 0; i < in.length; i++) {
            for (int u = 0; u < in[i].length; u++) {
                mean += in[i][u];
            }
        }
        mean = mean / in.length / in[0].length;

        float variance = 0;

        //calculate variance
        for (int i = 0; i < in.length; i++) {
            for (int u = 0; u < in[i].length; u++) {
                variance += Math.pow(in[i][u] - mean, 2);
            }
        }

        variance = variance / in.length / in[0].length;
        standardDeviation = (float) Math.sqrt(variance);
    }

    public void setPreprocessParameters(float[][][][] in) {
        mean = 0;
        for (int i = 0; i < in.length; i++) {
            for (int u = 0; u < in[i].length; u++) {
                for (int p = 0; p < in[i][u].length; p++) {
                    for (int a = 0; a < in[i][u][p].length; a++) {
                        mean += in[i][u][p][a];
                    }
                }
            }
        }
        mean = mean / in.length / in[0].length / in[0][0].length / in[0][0][0].length;

        float variance = 0;

        for (int i = 0; i < in.length; i++) {
            for (int u = 0; u < in[i].length; u++) {
                for (int p = 0; p < in[i][u].length; p++) {
                    for (int a = 0; a < in[i][u][p].length; a++) {
                        variance += Math.pow(in[i][u][p][a] - mean, 2);
                    }
                }
            }
        }

        variance = variance / in.length / in[0].length / in[0][0].length / in[0][0][0].length;
        standardDeviation = (float) Math.sqrt(variance);
    }

    public int getNumCommas(String in) {
        int count = 0;
        for (int i = 0; i < in.length(); i++) {
            if (in.charAt(i) == ',') {
                count++;
            }
        }
        return count;
    }

}
