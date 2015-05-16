package vub.tien.neutralnetwork;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.ejml.data.DenseMatrix64F;

public class NeutralNetwork {


	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("****************************** Neutral Network Illustration ***************************");
		InputLoader loader=new InputLoader("dataset/iris.arff");
		ImmutablePair<DenseMatrix64F, DenseMatrix64F> pair=loader.loadDataset();
		System.out.println("Number of examples: "+pair.getLeft().getNumRows());
		System.out.println("Number of attributes: "+pair.getLeft().getNumCols());
		System.out.println("classes: "+loader.getClassNames());
		
		ParamOptimizer.runGradientDescent(Constants.epochNum, pair);
	}
}
