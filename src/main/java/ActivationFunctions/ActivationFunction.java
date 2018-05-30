package ActivationFunctions;

import java.util.ArrayList;

public interface ActivationFunction {

    ArrayList<Double> getDerivative(ArrayList<Double> activationOutputs);

    ArrayList<Double> computeOutputs(ArrayList<Double> inputs);
}
