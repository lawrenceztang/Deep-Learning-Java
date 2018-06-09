package Kernels;

import com.aparapi.Kernel;

public class DotProductKernel extends Kernel {


    double[] in2;
    final int chunkSize;
    double[] in1;
    int[] arraySize;
    int[] numPasses;
    int[] gap;

    @Local
    double[] localResult;
    @Local
    double[] localIn2;

    public DotProductKernel(double[] in1, double[] in2, int chunkSize) {

        this.in1 = in1;
        this.in2 = in2;
        this.chunkSize = chunkSize;

        arraySize = new int[1];
        arraySize[0] = in1.length;

        numPasses = new int[1];

        gap = new int[1];
        gap[0] = 1;

        localIn2 = new double[in2.length];
        localResult = new double[in2.length];
    }

    @Override
    public void run() {
        int i = getGlobalId();

        //copy to local
        int position = i * chunkSize;
        if(position + chunkSize < arraySize[0]) {
            for(int s = 0; s < chunkSize; s++) {
                localResult[position + s] = in1[position + s];
                localIn2[position + s] = in2[position + s];
            }
        }
        else {
            for (int s = 0; s < chunkSize; s++) {
                if(position + s < arraySize[0]) {
                    localResult[position + s] = in1[position + s];
                    localIn2[position + s] = in2[position + s];
                }
                else {
                    break;
                }
            }
        }
        globalBarrier();

        //multiply
        position = i * chunkSize;

        //if i + chunkSize <= arraySize then no need to check for out of bounds
        if (position + chunkSize > arraySize[0]) {
            for (int d = 0; d < chunkSize; d++) {
                if (position + d < arraySize[0]) {
                    localResult[position + d] = localResult[position + d] * localIn2[position + d];
                }
            }
        } else {
            for (int d = 0; d < chunkSize; d++) {
                localResult[position + d] = localResult[position + d] * localIn2[position + d];
            }
        }

        globalBarrier();

        //add
        boolean exit = false;
        for (; !exit; ) {

            if (arraySize[0] == 1) {
                if(i == 0) {
                    in1[0] = localResult[0];
                }
                return;
            }

            if (position >= arraySize[0] * gap[0]) {
                globalBarrier();
                globalBarrier();
                continue;
            }

            position = i * gap[0] * chunkSize;

            //if position + chunkSize <= arraySize then no need to check for out of bounds
            if (position + chunkSize > arraySize[0]) {

                //add groups of chunkSize with stride size gap[0]
                for (int a = 1; a < chunkSize; a++) {
                    if (position + a * gap[0] < arraySize[0] * gap[0]) {
                        localResult[position] += localResult[position + a * gap[0]];
                    } else {
                        break;
                    }
                }
            } else {
                for (int a = 1; a < chunkSize; a++) {
                    localResult[position] += localResult[position + a * gap[0]];
                }
            }

            globalBarrier();
            if (getGlobalId() == 0) {
                numPasses[0]++;
                arraySize[0] = (int) Math.ceil((double) arraySize[0] / (double) chunkSize);
                gap[0] = (int) Math.pow(chunkSize, numPasses[0]);
            }
            globalBarrier();
        }

    }

    public double getResult() {
        return in1[0];
    }

    public int getRange() {
        return (int) Math.ceil((double) in1.length / (double) chunkSize);
    }

}

