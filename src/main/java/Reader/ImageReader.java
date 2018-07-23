package Reader;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

//reads images in folders with name of class, like MNIST
public class ImageReader implements Serializable{

    String path;
    public String[] classes;
    File file;
    public float[][] oneHotOutputs;
    Random rand;
    ArrayList<Integer> newImageReferenceForEachClass;
    public File[][] allFiles;
    float mean;
    float standardDeviation;

    public ImageReader(String path) throws Exception{
        this.path = path;;
        file = new File(path);
        rand = new Random();

        File[] files = file.listFiles();
        classes = new String[files.length];


        for (int i = 0; i < files.length; i++) {

            //get class names from subfolders
            String temp = files[i].toString();
            for (int u = 0; u < temp.length(); u++) {
                if (temp.charAt(temp.length() - 1 - u) == '\\') {
                    classes[u] = temp.substring(temp.length() - u, temp.length());
                    break;
                }
            }
        }

        allFiles = new File[files.length][];
        newImageReferenceForEachClass = new ArrayList<Integer>();

        for(int i = 0; i < files.length; i++) {
            allFiles[i] = files[i].listFiles();
            newImageReferenceForEachClass.add(0);
        }
    }

    public ImageReader () {

    }

    public ImageReader(ImageReader reader, float mean, float standardDeviation) throws Exception{
        this(reader.path);
        this.mean = mean;
        this.standardDeviation = standardDeviation;
    }


    public ArrayList<ArrayList<ArrayList<ArrayList<Float>>>> get3dColorMatrices (int batchSize) throws Exception{
        ArrayList<ArrayList<ArrayList<ArrayList<Float>>>> imagesAsMatrices = new ArrayList<ArrayList<ArrayList<ArrayList<Float>>>>();
        oneHotOutputs = new float[batchSize][];

        //get images and save as matrix

        for(int i = 0; i < batchSize; i++) {
            int random = rand.nextInt(allFiles.length);

            if(newImageReferenceForEachClass.get(random) > allFiles[random].length - 1) {
                break;
            }
            imagesAsMatrices.add(getImageAs2DMatrix(allFiles[random][i].getPath()));
            newImageReferenceForEachClass.set(random, newImageReferenceForEachClass.get(random) + 1);

            oneHotOutputs[i] = new float[classes.length];

            for(int t = 0; t < classes.length; t++) {
                if(t == i) {
                    oneHotOutputs[oneHotOutputs.length - 1][t] = 1f;
                }
                else {
                    oneHotOutputs[oneHotOutputs.length - 1][t] = 0f;
                }
            }
        }

        return imagesAsMatrices;
    }

    public float[][] get1dColorMatricesFromImages(int batchSize, int newWidth) throws Exception{
        float[][] imagesAsMatrices = new float[batchSize][];
        oneHotOutputs = new float[batchSize][];

        //get images and save as matrix
        //random chance for each class to be picked, evenly distributed
        for(int i = 0; i < batchSize; i++) {

            int random = rand.nextInt(allFiles.length);

            //if reference is over number of images in class, exit out of loop
            if(newImageReferenceForEachClass.get(random) > allFiles[random].length - 1) {
                break;
            }

            imagesAsMatrices[i] = getImageAs1DMatrix(allFiles[random][newImageReferenceForEachClass.get(random)].getPath(), newWidth);
            newImageReferenceForEachClass.set(random, newImageReferenceForEachClass.get(random) + 1);

            oneHotOutputs[i] = new float[classes.length];

            for(int t = 0; t < classes.length; t++) {
                if (t == random) {
                    oneHotOutputs[i][t] = 1f;
                } else {
                    oneHotOutputs[i][t] = 0f;
                }
            }

        }
        return imagesAsMatrices;
    }

    //return image as 2d matrix with color
    public ArrayList<ArrayList<ArrayList<Float>>> getImageAs2DMatrix(String filePath) throws Exception {
        BufferedImage image = ImageIO.read(new File(filePath));
        ArrayList<ArrayList<ArrayList<Float>>> matrix = new ArrayList<ArrayList<ArrayList<Float>>>();
        for(int i = 0; i < 3; i++) {
            matrix.add(new ArrayList<ArrayList<Float>>());
            for (int x = 0; x < image.getWidth(); x++) {
                matrix.get(i).add(new ArrayList<Float>());
                for (int y = 0; y < image.getHeight(); y++) {

                    Color color = new Color(image.getRGB(x, y));
                    if(i == 0) {
                        matrix.get(i).get(x).add((float) color.getRed());
                    }
                    else if(i == 1) {
                        matrix.get(i).get(x).add((float) color.getGreen());
                    }
                    else {
                        matrix.get(i).get(x).add((float) color.getBlue());
                    }
                }
            }
        }
        return matrix;
    }

    //return image as 1d matrix with color
    public float[] getImageAs1DMatrix(String filePath, int newWidth) throws Exception {
        BufferedImage image = ImageIO.read(new File(filePath));
        float[] matrix = new float[image.getWidth() * image.getHeight() * 3];
        int chunkSize = image.getWidth() / newWidth;

        for (int x = 0; x < image.getWidth(); x += chunkSize) {

            for (int y = 0; y < image.getHeight(); y += chunkSize) {
                if(x + chunkSize > image.getWidth()) {
                    return matrix;
                }
                else if(y + chunkSize > image.getHeight()) {
                    break;
                }

                int averageR = 0;
                int averageG = 0;
                int averageB = 0;
                for(int p = 0; p < chunkSize; p++) {
                    for(int u = 0; u < chunkSize; u++) {
                        Color color = new Color(image.getRGB(x + p, y + u));
                        averageB += color.getBlue();
                        averageG += color.getGreen();
                        averageR += color.getRed();
                    }
                }
                matrix[x * image.getHeight() * 3 + y * 3] = (float)(averageR / chunkSize / chunkSize);
                matrix[x * image.getHeight() * 3 + y * 3 + 1] =(float)(averageG / chunkSize / chunkSize);
                matrix[x * image.getHeight() * 3 + y * 3 + 2] =(float)(averageB / chunkSize / chunkSize);
            }
        }
        return matrix;
    }

    //some preprocessing saved in reader object
    public float[] preprocessExample (float[] in) {
        float[] out = new float[in.length];
        for(int i = 0; i < in.length; i++) {
            out[i] = (in[i] - mean) / standardDeviation;
        }
        return out;
    }

    public float[][] preprocessTrainingSet(float[][] in) {
        float[][] out = new float[in.length][];
        for(int p = 0; p < in.length; p++) {
            out[p] = new float[in[p].length];
            for (int u = 0; u < in[p].length; u++) {
                out[p][u] = (in[p][u] - mean) / standardDeviation;
            }
        }
        return out;
    }

    public float[] unpreprocessExample (float[] in) {
        float[] out = new float[in.length];
        for(int i = 0; i < in.length; i++) {
            out[i] = in[i] * standardDeviation + mean;
        }
        return out;
    }

    public void setPreprocessParameters (float[][] in) {
        mean = 0;
        for(int i = 0; i < in.length; i++) {
            for(int u = 0; u < in[i].length; u++) {
                mean += in[i][u];
            }
        }
        mean = mean / in.length / in[0].length;

        float variance = 0;

        //calculate variance
        for(int i = 0; i < in.length; i++) {
            for (int u = 0; u < in[i].length; u++) {
                variance += Math.pow(in[i][u] - mean, 2);
            }
        }

        variance = variance / in.length / in[0].length;
        standardDeviation = (float) Math.sqrt(variance);
    }

}
