package Run;

import Network.DenseNetwork;
import Reader.ImageReader;
import Runner.DenseRunner;
import Util.*;


import java.text.DecimalFormat;
import java.util.Scanner;

/*TODO list:
ResNet
maxout
Clustering
image transformations
efficient GPU
RNN
LSTM
L2 Norm
*/

public class RunFullyConnected {

    public static void main(String[] args) throws Exception {

        String trainingDataPath = "C:\\Users\\Anonymous\\Pictures\\Numbers\\mnist_png\\training";
        String testingDataPath = "C:\\Users\\Anonymous\\Pictures\\Numbers\\mnist_png\\testing";
        //"C:\\Users\\Anonymous\\Pictures\\Fortnite\\1\\0.jpg"

        DenseRunner runner = new DenseRunner()
                .setBatchSize(1)
                .setEpochs(10)
                .setIsImage(true)
                .setIterations(10000)
                .setMomentum(.9f)
                .setImageHeight(28)
                .setImageWidth(28)
                .setLayers(new int[]{2352, 200, 10})
                .setTestingIterations(3000)
                .setLearningRate(new float[]{0, .6f, .3f})
//                .setSchedule(new float[][]{{0, 10, 70000}})
                .setDropout(.5f)
                .setUpdateType(DenseNetwork.UPDATE_NESTEROV)
                .setTrainingDataPath(trainingDataPath)
                .setTestingDataPath(testingDataPath)
                .initialize();

//        runner.preTrain(new float[]{0, .15f, 30f},10000, 1, .9f);
        runner.train();
        runner.test();
    }
}
