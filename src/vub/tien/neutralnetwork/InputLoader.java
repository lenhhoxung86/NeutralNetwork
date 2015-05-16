package vub.tien.neutralnetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.ejml.data.DenseMatrix64F;

public class InputLoader {
	static final String INVALID_CLASS="INVALID_CLASS";
	private String fileLocation;
	private ArrayList<String>classes;
	private ArrayList<String>attributes;
	
	/**
	 * Constructor
	 * @param fileName
	 */
	public InputLoader(String fileName) {
		fileLocation=fileName;
		classes=new ArrayList<String>();
		attributes=new ArrayList<String>();
	}
	
	/**
	 * Translate a class name to class code
	 * Class code starts from index 1
	 * @param classCode
	 * @return
	 */
	public String translateToClassName(int classCode) {
		if((classCode>classes.size()) || (classCode<=0)) {
			return INVALID_CLASS;
		} else {
			return classes.get(classCode-1);
		}
	}
	
	/**
	 * Translate from class name to class code
	 * If the class name doesn't exist, it returns 0
	 * @param className
	 * @return integer value of class name starting from 1
	 */
	public int translateToClassCode(String className) {
		return classes.indexOf(className)+1;
	}
	
	/**
	 * Get all classes
	 * @return
	 */
	public ArrayList<String> getClassNames() {
		return classes;
	}
	
	/**
	 * Only care about lines starting with @ or text data
	 * parameter names start with @
	 * @param fileName
	 * @return
	 */
	public ImmutablePair<DenseMatrix64F, DenseMatrix64F> loadDataset() {
		//return values
		DenseMatrix64F X=null;
		DenseMatrix64F y=null;
		
		//-temporary lists
		ArrayList<ArrayList<Double>> examples=new ArrayList<ArrayList<Double>>();
		ArrayList<String> labels=new ArrayList<String>();

		File inputFile = new File(this.fileLocation);
		try {
			BufferedReader input = new BufferedReader(new FileReader(inputFile));
			try {
				String line = null;
				while ((line = input.readLine()) != null) {
					if (!line.startsWith("%")) {
						if(line.startsWith("@")) {
							String[] args=line.split("\\s+");
							if(args.length==3) {
								if(args[1].equals("class")) {
									String classNames=args[2];
									classNames=classNames.substring(1, classNames.length()-1);
									String[] classArray=classNames.split(",");
									for (int i = 0; i < classArray.length; i++) {
										this.classes.add(classArray[i]);
									}
								} else {
									this.attributes.add(args[1]);
								}
							}
						} else if(line.length()>0) {
							String[] args=line.split(",");
							ArrayList<Double>data=new ArrayList<Double>();
							for (int i = 0; i < args.length; i++) {
								if(i!=(args.length-1)) {
									data.add(Double.valueOf(args[i]));
								} else {
									labels.add(args[i]);
								}
							}
							examples.add(data);
						}
					} 
				}
				X=new DenseMatrix64F(examples.size(), examples.get(0).size());
				for (int i = 0; i < examples.size(); i++) {
					ArrayList<Double> doubles=examples.get(i);
					for (int j = 0; j < doubles.size(); j++) {
						X.set(i, j, doubles.get(j).doubleValue());
					}
				}
				
				y=new DenseMatrix64F(labels.size(), 1);
				for (int i = 0; i < labels.size(); i++) {
					y.set(i, 0, this.translateToClassCode(labels.get(i)));
				}
			} finally {
				input.close();
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		
		return ImmutablePair.of(X, y);
	}
}
