package vub.tien.neutralnetwork;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.ejml.data.DenseMatrix64F;
import org.ejml.equation.Equation;
import org.ejml.ops.*;

public class CostCalculator {
	/**
	 * This methods calculates cost value and partial derivatives
	 * @return
	 */
	public static ImmutableTriple<Double, DenseMatrix64F, DenseMatrix64F> calculateCostAndDerivatives(
			DenseMatrix64F theta1,
			DenseMatrix64F theta2,
			int inputNum,
			int hiddenNum,
			int classNum,
			DenseMatrix64F X,
			DenseMatrix64F y,
			double lambda) {
		
		ImmutableTriple<Double, DenseMatrix64F, DenseMatrix64F> retVal=null;
		Equation eq = new Equation();
		//-number of examples
		int m=X.getNumRows();
		double J=0;
		//-make aliases 
		eq.alias(m, "m");
		eq.alias(theta1, "Theta1");
		eq.alias(theta2, "Theta2");
		eq.alias(inputNum, "inputNum");
		eq.alias(hiddenNum, "hiddenNum");
		eq.alias(classNum, "classNum");
		eq.alias(X, "X");
		eq.alias(y, "y");
		eq.alias(lambda, "lambda");
		
		//-init Theta1_grad
		eq.process("Theta1_grad=zeros("+theta1.getNumRows()+","+theta1.getNumCols()+")");
		//-init Theta2_grad
		eq.process("Theta2_grad=zeros("+theta2.getNumRows()+","+theta2.getNumCols()+")");

		//-common operator to handle basic matrix operations
		CommonOps op=new CommonOps();
		
		for (int i = 0; i < m; i++) {
			/***************** Forward propagation *********************/
			//-extract example at line row i
			DenseMatrix64F X1=op.extractRow(X, i, null);
			eq.alias(X1, "X1");
			
			//-activation at the input layer
			eq.process("A1=[1 X1]'");
			
			//-calculate activation vector for the hidden layer
			eq.process("z2=Theta1*A1");
			DenseMatrix64F a2=sigmoid(eq.lookupMatrix("z2"));
			eq.alias(a2, "a2");
			eq.process("A2=[1;a2]");
			
			//-calculate activation vector for the output layer
			eq.process("z3=Theta2*A2");
			DenseMatrix64F a3=sigmoid(eq.lookupMatrix("z3"));
			eq.alias(a3, "a3");
			
			//-build output vector
			eq.process("yvec=zeros(classNum,1)");
			DenseMatrix64F yvec=eq.lookupMatrix("yvec");
			yvec.set((int)Math.round(y.get(i)-1), 0, 1);
			
			//-calculate cost J
			eq.process("h=-yvec .* log(a3) - (1 - yvec) .* log(1-a3)");
			DenseMatrix64F h=eq.lookupMatrix("h");
			J+=op.elementSum(h);
			
			/********************* Start back propagation here ********************/
			eq.process("delta3=a3-yvec");
			DenseMatrix64F sgZ2=sigmoidGradient(eq.lookupMatrix("z2"));
			eq.alias(sgZ2, "sgZ2");
			eq.process("delta2=((Theta2')*delta3) .* [1;sgZ2]");
			eq.process("delta2=delta2(1:,0:)");
			
			//-calculate theta gradient
			eq.process("Theta2_grad=Theta2_grad + (delta3 * (A2'))");
			eq.process("Theta1_grad=Theta1_grad + (delta2 * (A1'))");
		}
		
		//-update cost function again
		J = J/m;
		
		//-update theta gradient
		eq.process("Theta1_grad=Theta1_grad / m");
		eq.process("Theta2_grad=Theta2_grad / m");
		
		//-regularize cost value
		DenseMatrix64F mainTheta1=op.extract(theta1, 0, theta1.getNumRows(), 1, theta1.getNumCols());
		DenseMatrix64F mainTheta2=op.extract(theta2, 0, theta2.getNumRows(), 1, theta2.getNumCols());
		op.elementPower(mainTheta1, 2, mainTheta1);
		double term1=op.elementSum(mainTheta1);
		
		op.elementPower(mainTheta2, 2, mainTheta2);
		double term2=op.elementSum(mainTheta2);
		
		double regTerm=(lambda*(term1+term2))/(2*m);
		J+=regTerm;
		
		//-regularize theta gradient
		eq.process("regTerm1=(Theta1 * lambda) / m");
		eq.process("regTerm1(0:,0)=zeros(hiddenNum,1)");
		eq.process("Theta1_grad=Theta1_grad+regTerm1");
		
		eq.process("regTerm2=(Theta2 * lambda) / m");
		eq.process("regTerm2(0:,0)=zeros(classNum,1)");
		eq.process("Theta2_grad=Theta2_grad+regTerm2");
		
		//-build return value here
		Double costJ=new Double(J);
		retVal=ImmutableTriple.of(costJ, eq.lookupMatrix("Theta1_grad"), eq.lookupMatrix("Theta2_grad"));
		
		return retVal;
	}
	
	/**
	 * Sigmoid function - logistics function
	 * @param inputMatrix
	 * @return
	 */
	public static DenseMatrix64F sigmoid(DenseMatrix64F inputMatrix) {
		Equation eq=new Equation();
		eq.alias(inputMatrix, "inM");
		eq.process("outM=(1 / (1 + exp(-inM)))");
		
		return eq.lookupMatrix("outM");
	}
	
	/**
	 * Sigmoid gradient function
	 * @param inputMatrix
	 * @return
	 */
	public static DenseMatrix64F sigmoidGradient(DenseMatrix64F inputMatrix) {
		Equation eq=new Equation();
		DenseMatrix64F s1=sigmoid(inputMatrix);
		eq.alias(s1, "s1");
		eq.process("s2=1-s1");
		eq.process("g=s1 .* s2");
		
		return eq.lookupMatrix("g");
	}
}
