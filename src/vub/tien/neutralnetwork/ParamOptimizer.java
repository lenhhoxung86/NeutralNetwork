package vub.tien.neutralnetwork;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

public class ParamOptimizer {
	private static final int MAX=1000;
	private static final int MIN=0;
	private static final double EPSILON_INIT = 0.1;
	/**
	 * Returns a psuedo-random number between min and max, inclusive.
	 * The difference between min and max can be at most
	 * <code>Integer.MAX_VALUE - 1</code>.
	 *
	 * @param min Minimim value
	 * @param max Maximim value.  Must be greater than min.
	 * @return Integer between min and max, inclusive.
	 * @see java.util.Random#nextInt(int)
	 */
	public static int randInt(int min, int max) {

	    // Usually this can be a field rather than a method variable
	    Random rand = new Random();

	    // nextInt is normally exclusive of the top value,
	    // so add 1 to make it inclusive
	    int randomNum = rand.nextInt((max - min) + 1) + min;

	    return randomNum;
	}
	
	/**
	 * Generate a random value between 0 and 1
	 * @return
	 */
	public static double rand() {
		return (double)randInt(MIN, MAX)/(MAX-MIN);
	}
	
	/**
	 * Create initial matrices for theta
	 * @param L_in
	 * @param L_out
	 * @return
	 */
	public static DenseMatrix64F randInitializeWeights(int L_in, int L_out) {
		DenseMatrix64F randM=new DenseMatrix64F(L_out, L_in+1);
		for (int i = 0; i < randM.getNumRows(); i++) {
			for (int j = 0; j < randM.getNumCols(); j++) {
				randM.set(i, j, (rand()*2*EPSILON_INIT-EPSILON_INIT));
			}
		}
		
		return randM;
	}
	
	/**
	 * Run gradient descent algorithm
	 * @param epochNum
	 * @param trainingSet
	 */
	public static void runGradientDescent(int epochNum, ImmutablePair<DenseMatrix64F, DenseMatrix64F> trainingSet) {
		DenseMatrix64F theta1=ParamOptimizer.randInitializeWeights(4, 3);
		DenseMatrix64F theta2=ParamOptimizer.randInitializeWeights(3, 3);
		for (int i = 0; i < epochNum; i++) {
			System.out.println("***** epoch "+(i+1)+"*****");
			ImmutableTriple<Double, DenseMatrix64F, DenseMatrix64F> result=CostCalculator.calculateCostAndDerivatives(
					theta1, 
					theta2, 
					Constants.numberOfInputUnits, 
					Constants.numberOfHiddedUnits, 
					Constants.numberOfOutputUnits, 
					trainingSet.getLeft(), 
					trainingSet.getRight(), 
					Constants.lambda);
			System.out.println("epoch "+(i+1)+" - J="+result.getLeft());
			
			//-update theta values
			CommonOps op=new CommonOps();
			
			DenseMatrix64F theta1_grad=result.getMiddle();
			op.scale(Constants.alpha, theta1_grad);
			op.subtract(theta1, theta1_grad, theta1);
			
			DenseMatrix64F theta2_grad=result.getRight();
			op.scale(Constants.alpha, theta2_grad);
			op.subtract(theta2, theta2_grad, theta2);
		}
		saveWeightsToFile("netparams.txt",theta1,theta2);
		System.out.println("Network parameters saved!");
	}
	
	public static void saveWeightsToFile(String fileName,DenseMatrix64F theta1,DenseMatrix64F theta2) {
		//-delete the old file
		
		BufferedWriter writer = null;
        try {
            //create a temporary file
            File logFile = new File(fileName);
            if(logFile.exists()) {
            	logFile.delete();
            }
            NumberFormat formatter = new DecimalFormat("#0.00000000");
            writer = new BufferedWriter(new FileWriter(logFile));
            writer.write("=== Weights for first layer ===\n");
            for (int i = 0; i < theta1.getNumRows(); i++) {
				for (int j = 0; j < theta1.getNumCols(); j++) {
					writer.write(formatter.format(theta1.get(i, j)));
					if(j!=theta1.getNumCols()-1) {
						writer.write("    ");
					}
				}
				writer.write("\n");
			}
            writer.write("=== Weights for second layer ===\n");
            for (int i = 0; i < theta2.getNumRows(); i++) {
				for (int j = 0; j < theta2.getNumCols(); j++) {
					writer.write(formatter.format(theta1.get(i, j)));
					if(j!=theta2.getNumCols()-1) {
						writer.write("    ");
					}
				}
				if(i!=theta2.getNumRows()-1) {
					writer.write("\n");
				}
			}
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                // Close the writer regardless of what happens...
                writer.close();
            } catch (Exception e) {
            }
        }
	}
}
