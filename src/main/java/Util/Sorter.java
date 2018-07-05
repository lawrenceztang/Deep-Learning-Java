package Util;

import java.util.Arrays;

public class Sorter {

    public static double[] mergeSort (double[] in) {
        int groupSize = 1;
        while (true) {
            if (groupSize > in.length) {
                break;
            }
            for (int i = 0; i < in.length; i += groupSize * 2) {
                double[] array;
                if (i + groupSize > in.length) {

                }
                else if(i + groupSize * 2 > in.length) {
                    array = merge(Arrays.copyOfRange(in, i, i + groupSize), Arrays.copyOfRange(in, i + groupSize, in.length));
                    for(int r = i; r < in.length; r++) {
                        in[r] = array[r - i];
                    }
                }
                else {
                    array = merge(Arrays.copyOfRange(in, i, i + groupSize), Arrays.copyOfRange(in, i + groupSize, i + groupSize * 2));
                    for (int r = i; r < i + groupSize * 2; r++) {
                        in[r] = array[r - i];
                    }

                }
            }
            groupSize *= 2;
        }
        return in;
    }

    public static double[] merge (double[] in, double[] in2) {
        double[] out = new double[in.length + in2.length];
        int inIndex = 0;
        int in2Index = 0;
        for (int i = 0; i < out.length; i++) {
            if (inIndex >= in.length) {
                out[i] = in2[in2Index];
            }
            else if (in2Index >= in2.length) {
                out[i] = in[inIndex];
            }
            else if(in[inIndex] < in2[in2Index]) {
                out[i] = in[inIndex];
            }
            else {
                out[i] = in2[in2Index];
            }

        }
        return out;
    }

}
