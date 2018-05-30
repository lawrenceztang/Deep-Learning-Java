package Kernels;

import com.aparapi.Kernel;
import com.aparapi.opencl.OpenCL;

public class AdditionKernel extends Kernel {
    double[] in;

    double[] resultLocal;

    int numPasses;

    int arraySize;

    int chunkSize = 2;

    public AdditionKernel(double[] in, int chunkSize) {
        this.in = in;
        resultLocal = new double[in.length];
        arraySize = resultLocal.length;
    }

    @Override
    public void run() {
        if (getCurrentPass() == 0) {
            for (int p = 0; p < in.length; p++) {
                resultLocal[p] = in[p];
            }
        }

        int gap = (int) Math.pow(chunkSize, numPasses);

        int i = getGlobalId() * gap * chunkSize;


        for (int a = 1; a < chunkSize; a++) {
            if (i + a < arraySize) {
                resultLocal[i] += resultLocal[i + a * gap];
                System.out.println("hi");
            } else {
                break;
            }
        }


        if (getGlobalId() == resultLocal.length - 1) {
            arraySize = (int) Math.ceil((double) arraySize / (double) chunkSize);
            numPasses++;
            if (arraySize == 1) {
                //should exit
            }
        }
    }
    public double getResult() {return resultLocal[0];}


}

