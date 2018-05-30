import org.junit.Assert;
import org.junit.Test;
import Util.Util;

import java.util.ArrayList;


public class AparapiTest {

    public static void main(String[] args) {


    }

    @Test
    public void testDotProduct () {

        ArrayList<Double> vector1 = new ArrayList<Double>();
        ArrayList<Double> vector2 = new ArrayList<Double>();
        for(int i = 0; i < 2000; i++) {
            vector1.add(3d);
            vector2.add(7d);
        }

        double startTime = System.currentTimeMillis();
        double dotProduct = Util.dotProduct(vector1, vector2, 2);
        System.out.println("With GPU: " + (System.currentTimeMillis() - startTime));

        double startTime2 = System.currentTimeMillis();
        double dotProduct2 = Util.dotProduct(vector1, vector2, 2);
        System.out.println("With GPU second time: " + (System.currentTimeMillis() - startTime2));

        double startTime3 = System.currentTimeMillis();
        double dotProduct3 = Util.dotProductNoGPU(vector1, vector2);
        System.out.println("Without GPU: " + (System.currentTimeMillis() - startTime3));

        Assert.assertEquals(dotProduct3, dotProduct, .02);
    }

}
