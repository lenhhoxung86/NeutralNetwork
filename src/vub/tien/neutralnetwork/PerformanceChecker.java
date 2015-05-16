package vub.tien.neutralnetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.ArrayList;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.ejml.data.DenseMatrix64F;
import org.ejml.equation.Equation;
import org.ejml.ops.CommonOps;

//Specific for a neutral network configuration
public class PerformanceChecker {
	public static void main(String[] args) throws ParseException {
		// TODO Auto-generated method stub
		
		DenseMatrix64F theta1=new DenseMatrix64F(3,5);
		DenseMatrix64F theta2=new DenseMatrix64F(3,4);
		
		//-load parameters into matrices
		BufferedReader reader=null;
		try {
		    File file = new File("netparams.txt");
		    reader = new BufferedReader(new FileReader(file));
		    String line;
		    int lineIndex1=0;
		    int lineIndex2=0;
		    while ((line = reader.readLine()) != null) {
		        if((line.length()>0) && !line.startsWith("=")) {
		        	String[] numArray=line.split("\\s+");
		        	if(numArray.length==5) {
		        		//-for theta1
		        		theta1.set(lineIndex1, 0, Double.valueOf(numArray[0]));
		        		theta1.set(lineIndex1, 1, Double.valueOf(numArray[1]));
		        		theta1.set(lineIndex1, 2, Double.valueOf(numArray[2]));
		        		theta1.set(lineIndex1, 3, Double.valueOf(numArray[3]));
		        		theta1.set(lineIndex1, 4, Double.valueOf(numArray[4]));
		        		lineIndex1++;
		        	} else if(numArray.length==4) {
		        		//-for theta2
		        		theta2.set(lineIndex2, 0, Double.valueOf(numArray[0]));
		        		theta2.set(lineIndex2, 1, Double.valueOf(numArray[1]));
		        		theta2.set(lineIndex2, 2, Double.valueOf(numArray[2]));
		        		theta2.set(lineIndex2, 3, Double.valueOf(numArray[3]));
		        		lineIndex2++;
		        	}
		        }
		    }
		    System.out.println("theta1"+theta1);
		    System.out.println("theta2"+theta2);
		} catch (IOException e) {
		    e.printStackTrace();
		} finally {
		    try {
		        reader.close();
		    } catch (IOException e) {
		        e.printStackTrace();
		    }
		}
		
		//-calculates output values
		InputLoader loader=new InputLoader("dataset/iris.arff");
		ImmutablePair<DenseMatrix64F, DenseMatrix64F> pair=loader.loadDataset();
		DenseMatrix64F X=pair.getLeft();
		DenseMatrix64F y=pair.getRight();
		int m=y.getNumRows();
		int[] outputLabels=new int[m];
		int[] labels=new int[m];
		for (int i = 0; i < y.getNumRows(); i++) {
			labels[i]=(int)y.get(i);
		}
		
		CommonOps op=new CommonOps();
		Equation eq=new Equation();
		eq.alias(theta1, "Theta1");
		eq.alias(theta2, "Theta2");
		for (int i = 0; i < m; i++) {
			/***************** Forward propagation *********************/
			//-extract example at line row i
			DenseMatrix64F X1=op.extractRow(X, i, null);
			System.out.println("example: "+X1);
			System.out.println("label: "+y.get(i));
			eq.alias(X1, "X1");

			//-activation at the input layer
			eq.process("A1=[1 X1]'");

			//-calculate activation vector for the hidden layer
			eq.process("z2=Theta1*A1");
			DenseMatrix64F a2=CostCalculator.sigmoid(eq.lookupMatrix("z2"));
			eq.alias(a2, "a2");
			eq.process("A2=[1;a2]");

			//-calculate activation vector for the output layer
			eq.process("z3=Theta2*A2");
			DenseMatrix64F a3=CostCalculator.sigmoid(eq.lookupMatrix("z3"));
			System.out.println("output:"+a3);
			//-make output labels
			int labelVal=-1;
			double max=-1;
			for (int j = 0; j < a3.getNumRows(); j++) {
				if(max<a3.get(j)) {
					max=a3.get(j);
					labelVal=j+1;
				}
			}
			outputLabels[i]=labelVal;
		}
		
		//-print result
//		for (int i = 0; i < outputLabels.length; i++) {
//			System.out.println(outputLabels[i]);
//		}
		
	}
}
