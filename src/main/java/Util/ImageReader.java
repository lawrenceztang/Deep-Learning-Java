package Util;

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
    public ArrayList<String> classes;
    File file;
    public ArrayList<ArrayList<Double>> oneHotOutputs;
    Random rand;
    ArrayList<Integer> newImageReferenceForEachClass;
    File[][] allFiles;
    double mean;
    double standardDeviation;

    public ImageReader(String path) throws Exception{
        this.path = path;
        classes = new ArrayList<String>();
        file = new File(path);
        oneHotOutputs = new ArrayList<ArrayList<Double>>();
        rand = new Random();

        File[] files = file.listFiles();


        for (int i = 0; i < files.length; i++) {

            //get class names from subfolders
            String temp = files[i].toString();
            for (int u = 0; u < temp.length(); u++) {
                if (temp.charAt(temp.length() - 1 - u) == '\\') {
                    classes.add(temp.substring(temp.length() - u, temp.length()));
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



    public ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> get2dColorMatrices (int batchSize) throws Exception{
        ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> imagesAsMatrices = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();
        oneHotOutputs = new ArrayList<ArrayList<Double>>();

        //get images and save as matrix

        for(int i = 0; i < batchSize; i++) {
            int random = rand.nextInt(allFiles.length);

            if(newImageReferenceForEachClass.get(random) > allFiles[random].length - 1) {
                break;
            }
            imagesAsMatrices.add(getImageAs2DMatrix(allFiles[random][i].getPath()));
            newImageReferenceForEachClass.set(random, newImageReferenceForEachClass.get(random) + 1);

            oneHotOutputs.add(new ArrayList<Double>());

            for(int t = 0; t < classes.size(); t++) {
                if(t == i) {
                    oneHotOutputs.get(oneHotOutputs.size() - 1).add(0d);
                }
                else {
                    oneHotOutputs.get(oneHotOutputs.size() - 1).add(1d);
                }
            }
        }

        return imagesAsMatrices;
    }

    public ArrayList<ArrayList<Double>> get1dColorMatricesFromImages(int batchSize, int newWidth) throws Exception{
        ArrayList<ArrayList<Double>> imagesAsMatrices = new ArrayList<ArrayList<Double>>();
        oneHotOutputs = new ArrayList<ArrayList<Double>>();

        //get images and save as matrix
        //random chance for each class to be picked, evenly distributed
        for(int i = 0; i < batchSize; i++) {

            int random = rand.nextInt(allFiles.length);

            //if reference is over number of images in class, exit out of loop
            if(newImageReferenceForEachClass.get(random) > allFiles[random].length - 1) {
                break;
            }

            imagesAsMatrices.add(getImageAs1DMatrix(allFiles[random][newImageReferenceForEachClass.get(random)].getPath(), newWidth));
            newImageReferenceForEachClass.set(random, newImageReferenceForEachClass.get(random) + 1);

            oneHotOutputs.add(new ArrayList<Double>());

            for(int t = 0; t < classes.size(); t++) {
                if (t == random) {
                    oneHotOutputs.get(oneHotOutputs.size() - 1).add(1d);
                } else {
                    oneHotOutputs.get(oneHotOutputs.size() - 1).add(0d);
                }
            }

        }
        return imagesAsMatrices;
    }

    //return image as 2d matrix with color
    public ArrayList<ArrayList<ArrayList<Double>>> getImageAs2DMatrix(String filePath) throws Exception {
        BufferedImage image = ImageIO.read(new File(filePath));
        ArrayList<ArrayList<ArrayList<Double>>> matrix = new ArrayList<ArrayList<ArrayList<Double>>>();
        for (int x = 0; x < image.getWidth(); x++) {
            matrix.add(new ArrayList<ArrayList<Double>>());
            for (int y = 0; y < image.getHeight(); y++) {
                Color color = new Color(image.getRGB(x, y));
                matrix.get(x).add(new ArrayList<Double>());
                matrix.get(x).get(y).add((double)color.getRed());
                matrix.get(x).get(y).add((double)color.getGreen());
                matrix.get(x).get(y).add((double)color.getBlue());
            }
        }
        return matrix;
    }

    //return image as 1d matrix with color
    public ArrayList<Double> getImageAs1DMatrix(String filePath, int newWidth) throws Exception {
        BufferedImage image = ImageIO.read(new File(filePath));
        ArrayList<Double> matrix = new ArrayList<Double>();
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
                matrix.add((double)averageR);
                matrix.add((double)averageG);
                matrix.add((double)averageB);
            }
        }
        return matrix;
    }

    //some preprocessing saved in reader object
    public ArrayList<Double> preprocessExample (ArrayList<Double> in) {
        ArrayList<Double> out = new ArrayList<Double>();
        for(int i = 0; i < in.size(); i++) {
            out.add((in.get(i) - mean) / standardDeviation);
        }
        return out;
    }

    public ArrayList<ArrayList<Double>> preprocessTrainingSet(ArrayList<ArrayList<Double>> in) {
        ArrayList<ArrayList<Double>> out = new ArrayList<ArrayList<Double>>();
        for(int p = 0; p < in.size(); p++) {
            out.add(new ArrayList<Double>());
            for (int u = 0; u < in.get(p).size(); u++) {
                out.get(p).add((in.get(p).get(u) - mean) / standardDeviation);
            }
        }
        return out;
    }

    public ArrayList<Double> unpreprocessExample (ArrayList<Double> in) {
        ArrayList<Double> out = new ArrayList<Double>();
        for(int i = 0; i < in.size(); i++) {
            out.add(in.get(i) * standardDeviation + mean);
        }
        return out;
    }

    public void setPreprocessParameters (ArrayList<ArrayList<Double>> in) {
        mean = 0;
        for(int i = 0; i < in.size(); i++) {
            for(int u = 0; u < in.get(i).size(); u++) {
                mean += in.get(i).get(u);
            }
        }
        mean = mean / in.size() / in.get(0).size();

        double variance = 0;

        //calculate variance
        for(int i = 0; i < in.size(); i++) {
            for (int u = 0; u < in.get(i).size(); u++) {
                variance += Math.pow(in.get(i).get(u) - mean, 2);
            }
        }

        variance = variance / in.size() / in.get(0).size();
        standardDeviation = Math.sqrt(variance);
    }

}
