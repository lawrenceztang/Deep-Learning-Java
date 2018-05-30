package Kernels;


import com.aparapi.Kernel;

public class MultiplicationKernel extends Kernel {
    @Local
    double in1[];
    @Local
    double in2[];
    double result[];
    double in1Temp[];
    double in2Temp[];

    public MultiplicationKernel(double[] in1Temp, double[] in2Temp, double[] result) {
        this.in1Temp = in1Temp;
        this.in2Temp = in2Temp;
        this.result = result;
        in1 = new double[in1Temp.length];
        in2 = new double[in1Temp.length];
    }

    @Override
    public void run() {
        int i = getGlobalId();

        in1[i] = in1Temp[i];
        in2[i] = in2Temp[i];


        result[i] = in1[i] * in2[i];
    }

}
