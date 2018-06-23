package Kernels;

import com.aparapi.Kernel;


//one local group for each vector because they dont have dependency on each other
public class MatrixVectorKernel extends Kernel {

    public double[] in2;
    final int chunkSize;
    public double[][] in1;

    @Local
    int[] arraySize;
    @Local
    int[] numPasses;
    @Local
    int[] gap;

    @Local
    double[] localResult;
    @Local
    double[] localIn2;

    public MatrixVectorKernel(double[][] in1, double[] in2, int chunkSize) {

        this.in1 = in1;
        this.in2 = in2;
        this.chunkSize = chunkSize;

        arraySize = new int[1];
        numPasses = new int[1];
        gap = new int[1];

        localResult = new double[in1[0].length];
        localIn2 = new double[in2.length];
    }

    @Override
    public void run() {
        int i = getLocalId();
        int e = getGroupId();

        gap[0] = 1;
        numPasses[0] = 0;
        arraySize[0] = in2.length;

        //copy to local
        int position = i * chunkSize;
        if (position + chunkSize < arraySize[0]) {
            for (int s = 0; s < chunkSize; s++) {
                localResult[position + s] = in1[e][position + s];

                localIn2[position + s] = in2[position + s];
            }
        } else {
            for (int s = 0; s < chunkSize; s++) {
                if (position + s < arraySize[0]) {
                    localResult[position + s] = in1[e][position + s];
                } else {
                    break;
                }
            }
        }

        localBarrier();

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

        localBarrier();

        //add
        boolean exit = false;
        for (; !exit; ) {

            if (arraySize[0] == 1) {
                in1[e][0] = localResult[0];
                return;
            }

            if (position >= arraySize[0] * gap[0]) {
                localBarrier();
                localBarrier();
                continue;
            }

            position = i * gap[0] * chunkSize;

            //if position + chunkSize <= arraySize then no need to check for out of bounds
            if (position + chunkSize * gap[0] >= arraySize[0] * gap[0]) {

                //add groups of chunkSize with stride size gap
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

            localBarrier();
            if (getLocalId() == 0) {
                numPasses[0]++;
                arraySize[0] = (int) Math.ceil((double) arraySize[0] / (double) chunkSize);
                gap[0] = (int) Math.pow(chunkSize, numPasses[0]);
                if (gap[0] > localIn2.length) {
                    gap[0] = localIn2.length;
                }
            }
            localBarrier();
        }

    }


}
