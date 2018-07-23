package Run;

import Network.DenseNetwork;
import Reader.ImageReader;
import Util.NetworkSaver;

import java.util.Scanner;

public class LoadNetwork {

    public String path;

    public LoadNetwork(String path, DenseNetwork network, ImageReader reader) throws Exception {
        this.path = path;
        Scanner scan = new Scanner(System.in);
        NetworkSaver save = new NetworkSaver();
        network = save.deserializeNetwork(path + "\\network");
        reader = save.deserializeReader(path + "C:\\Users\\Anonymous\\Documents\\DeepLearningSaves\\Good Logistic Sigmoid\\reader");
    }


}
