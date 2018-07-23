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
        ArrayList<Float> arList = new ArrayList<Float>();
        float[] arr = new float[arrSize];
        for(int i = 0; i < arr.length; i++) {
            float num = rand.nextInt(1000);
            arr[i] = num;
        }
        float[] out = Sorter.mergeSort(arr);
        System.out.println("hi");
    }

    @Test
    public void testDotProduct () throws Exception {


        //test dot product
        float[] vector1 = new float[60000];
        float[] vector2 = new float[60000];
        for(int i = 0; i < 60000; i++) {
            vector1[i] = 3f;
            vector2[i] = 5f;
        }

        float startTime3 = System.currentTimeMillis();
        float dotProduct3 = ArrOperations.dotProductNoGPU(vector1, vector2);
        System.out.println("Without GPU: " + (System.currentTimeMillis() - startTime3));

        float startTime = System.currentTimeMillis();
        float dotProduct = GPUOperations.dotProduct(vector1, vector2);
        System.out.println("With GPU: " + (System.currentTimeMillis() - startTime));

        float startTime2 = System.currentTimeMillis();
        float dotProduct2 = GPUOperations.dotProduct(vector1, vector2);
        System.out.println("With GPU second time: " + (System.currentTimeMillis() - startTime2));

        Assert.assertEquals(dotProduct3, dotProduct, .02);

        //test matrix vector
        int vectorSize = 1000;
        int numberVectors = 1;
        float[][] matrix = new float[numberVectors][vectorSize];
        float[] vector = new float[vectorSize];
        for(int i = 0; i < vector.length; i++) {
            for(int u = 0; u < matrix.length; u++) {
                matrix[u][i] = 5;
            }
            vector[i] = 3;
        }


        float[] out = new float[numberVectors];
        startTime = System.currentTimeMillis();
        for(int i = 0; i < numberVectors; i++) {
            out[i] = ArrOperations.dotProductNoGPU(vector, matrix[i]);
        }
        System.out.println("Without GPU: " + (System.currentTimeMillis() - startTime));

        startTime = System.currentTimeMillis();
        float[] out2 = ArrOperations.matrixVectorProduct(vector, matrix);
        System.out.println("With GPU: " + (System.currentTimeMillis() - startTime));

        Assert.assertEquals(out[0], out2[0], .02);

    }

}
