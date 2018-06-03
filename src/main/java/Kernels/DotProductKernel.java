package Kernels;

import com.aparapi.Kernel;
import com.aparapi.opencl.OpenCL;

public class DotProductKernel extends Kernel {

    double[] in2;
    final int chunkSize;
    double[] result;
    int[] arraySize;
    int[] numPasses;
    int[] gap;

    public DotProductKernel(double[] result, double[] in2, int chunkSize) {

        this.result = result;
        this.in2 = in2;
        this.chunkSize = chunkSize;

        arraySize = new int[1];
        arraySize[0] = result.length;

        numPasses = new int[1];

        gap = new int[1];
        gap[0] = 1;
    }

    @Override
    public void run() {
        int i = getGlobalId();
        result[i] = result[i] * in2[i];
        globalBarrier();


        boolean exit = false;
        for(;!exit;){

            if (arraySize[0] == 1) {
                return;
            }

            i = getGlobalId() * gap[0] * chunkSize;

            if(i >= arraySize[0] * gap[0]) {
                return;
            }

            for (int a = 1; a < chunkSize; a++) {
                if (i + a * gap[0] < arraySize[0] * gap[0]) {
                    result[i] += result[i + a * gap[0]];
                } else {
                    break;
                }
            }

            globalBarrier();
            if(getGlobalId() == 0) {
                numPasses[0]++;
                arraySize[0] = (int) Math.ceil((double) arraySize[0] / (double) chunkSize);
                gap[0] = (int) Math.pow(chunkSize, numPasses[0]);
            }
            globalBarrier();

        }
    }

    public double getResult() {
        return result[0];
    }

}

