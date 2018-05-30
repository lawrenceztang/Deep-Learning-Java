package Run;

import Network.FullyConnectedNetwork;
import Util.ImageReader;
import Util.NetworkSaver;
import Util.NetworkSaver;

import java.util.Scanner;

public class LoadNetwork {

    public static void main(String[] args) throws Exception{
        Scanner scan = new Scanner(System.in);
        NetworkSaver save = new NetworkSaver();
        FullyConnectedNetwork network = save.deserializeNetwork("C:\\Users\\Anonymous\\Documents\\DeepLearningSaves\\Good Logistic Sigmoid\\mnistMyOwnNetwork");
        ImageReader reader = save.deserializeReader("C:\\Users\\Anonymous\\Documents\\DeepLearningSaves\\Good Logistic Sigmoid\\reader");

        //specific example
        while(true) {
            System.out.println("Test?");
            if(!scan.nextLine().equals("y")) {
                break;
            }
            System.out.println(network.test(reader));
        }
    }
}
