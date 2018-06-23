package Kernels;

import com.aparapi.Kernel;

public class TestKernel extends Kernel {


    double[] in2;


    public TestKernel(double[] in2) {
        this.in2 = in2;

    }

    @Override
    public void run() {

        @Local
        double[] localIn2;
        localIn2 = new double[in2.length];

        int i = getLocalId();
        localIn2[i] += in2[i];
        localBarrier();
        localBarrier();
    }


}
