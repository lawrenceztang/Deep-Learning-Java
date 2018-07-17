package Util;

import Kernels.DotProductKernel;
import com.aparapi.device.Device;
import com.aparapi.device.JavaDevice;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.CountDownLatch;

public class gpuOperations {

    public static double dotProduct(double[] in1Copy, double[] in2Copy) throws Exception {


        //for running in Java mode:
//       KernelManager.setKernelManager(new JTPKernelManager());

        OpenCLDevice device = (OpenCLDevice) KernelManager.instance().bestDevice();
        device.setSharedMemory(false);

        //max 4000 doubles / 2 arrays = 2000 doubles per array - 200 for extra space
        final int maxDoubles = (int) device.getLocalMemSize() / 16 - 200;
        int chunkSize = (int) Math.ceil((double) maxDoubles / (double) device.getMaxWorkGroupSize());
        if(chunkSize == 1) {
            chunkSize = 2;
        }
        int numIterations = (int) Math.ceil((double) in1Copy.length / (double) maxDoubles);
        CountDownLatch latch = new CountDownLatch(numIterations);
        double[] out = new double[numIterations];

        for (int u = 0; u < numIterations; u++) {

            //create subarrays that are less than max local size
            //todo: don't use copy just pass bounds as parameter
            double[] subarray1;
            double[] subarray2;
            if (u < numIterations - 1) {
                subarray1 = Arrays.copyOfRange(in1Copy, u * maxDoubles, (u + 1) * maxDoubles);
                subarray2 = Arrays.copyOfRange(in2Copy, u * maxDoubles, (u + 1) * maxDoubles);
            } else {
                subarray1 = Arrays.copyOfRange(in1Copy, u * maxDoubles, in1Copy.length);
                subarray2 = Arrays.copyOfRange(in2Copy, u * maxDoubles, in1Copy.length);
            }

            RunKernelThread thread = new RunKernelThread(subarray1, subarray2, chunkSize, device, out, u, latch);
            thread.start();
        }

        //wait until all threads complete
        latch.await();

        //now safe to retrieve output
        for (int i = 1; i < out.length; i++) {
            out[0] += out[i];
        }
        return out[0];
    }

    public static class RunKernelThread extends Thread {
        double[] in1;
        double[] in2;
        int chunkSize;
        Device device;
        int iteration;
        public final double[] out;
        CountDownLatch latch;

        public RunKernelThread(double[] in1, double[] in2, int chunkSize, Device device, double[] out, int iteration, CountDownLatch latch) {
            this.in1 = in1;
            this.in2 = in2;
            this.chunkSize = chunkSize;
            this.device = device;
            this.out = out;
            this.iteration = iteration;
            this.latch = latch;
        }

        @Override
        public synchronized void run() {
            super.run();
            DotProductKernel kernel = new DotProductKernel(in1, in2, chunkSize);
            kernel.execute(device.createRange(kernel.getRange(), kernel.getRange()));
            out[iteration] += kernel.getResult();
            latch.countDown();
        }
    }

    public static class JTPKernelManager extends KernelManager {
        public JTPKernelManager() {
            LinkedHashSet<Device> preferredDevices = new LinkedHashSet<Device>(1);
            preferredDevices.add(JavaDevice.THREAD_POOL);
            setDefaultPreferredDevices(preferredDevices);
        }

        @Override
        protected List<Device.TYPE> getPreferredDeviceTypes() {
            return Arrays.asList(Device.TYPE.JTP);
        }
    }
}
