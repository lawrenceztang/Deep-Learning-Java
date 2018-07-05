import org.junit.Assert;
import org.junit.Test;
import Util.*;

import java.util.ArrayList;
import java.util.Random;


public class AparapiTest {

    public static void main(String[] args) {

    }

    @Test
    public void arrayVsArrayList () throws Exception {
        Random rand = new Random();
        int arrSize = 9;
        ArrayList<Double> arList = new ArrayList<Double>();
        double[] arr = new double[arrSize];
        for(int i = 0; i < arr.length; i++) {
            double num = rand.nextInt(1000);
            arr[i] = num;
        }
        double[] out = Sorter.mergeSort(arr);
        System.out.println("hi");
    }

    @Test
    public void testDotProduct () throws Exception {


        //test dot product
        ArrayList<Double> vector1 = new ArrayList<Double>();
        ArrayList<Double> vector2 = new ArrayList<Double>();
        for(int i = 0; i < 60000; i++) {
            vector1.add(3d);
            vector2.add(5d);
        }

        double startTime3 = System.currentTimeMillis();
        double dotProduct3 = Util.dotProductNoGPU(vector1, vector2);
        System.out.println("Without GPU: " + (System.currentTimeMillis() - startTime3));

        double startTime = System.currentTimeMillis();
        double dotProduct = Util.dotProduct(vector1, vector2);
        System.out.println("With GPU: " + (System.currentTimeMillis() - startTime));

        double startTime2 = System.currentTimeMillis();
        double dotProduct2 = Util.dotProduct(vector1, vector2);
        System.out.println("With GPU second time: " + (System.currentTimeMillis() - startTime2));

        Assert.assertEquals(dotProduct3, dotProduct, .02);

        //test matrix vector
        int vectorSize = 1000;
        int numberVectors = 1;
        double[][] matrix = new double[numberVectors][vectorSize];
        double[] vector = new double[vectorSize];
        for(int i = 0; i < vector.length; i++) {
            for(int u = 0; u < matrix.length; u++) {
                matrix[u][i] = 5;
            }
            vector[i] = 3;
        }


        double[] out = new double[numberVectors];
        startTime = System.currentTimeMillis();
        for(int i = 0; i < numberVectors; i++) {
            out[i] = Util.dotProductNoGPU(vector, matrix[i]);
        }
        System.out.println("Without GPU: " + (System.currentTimeMillis() - startTime));

        startTime = System.currentTimeMillis();
        double[] out2 = Util.matrixVectorProduct(vector, matrix);
        System.out.println("With GPU: " + (System.currentTimeMillis() - startTime));

        Assert.assertEquals(out[0], out2[0], .02);

    }

}
