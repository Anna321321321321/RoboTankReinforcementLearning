package a2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;

import robocode.RobocodeFileOutputStream;

public class NeuralNet implements NeuralNetInterface{
	int argNumInputs;
	int argNumHidden;
	double argLearningRate; 
	double argMomentumTerm;
	double argA;
	double argB;
	double[][][]weight;
	double [][][] weightChange;
	double[] input;
	double []h;
	double []delta;

	public NeuralNet ( 
			int argNumInputs,
			int argNumHidden,
			double argLearningRate, 
			double argMomentumTerm, 
			double argA,
			double argB ) {
		this.argNumInputs=argNumInputs;
		this.argNumHidden=argNumHidden;
		this.argLearningRate=argLearningRate;
		this.argMomentumTerm=argMomentumTerm; 
		this.argA=argA;
		this.argB=argB;
		this.weight=new double[2][][];
		this.weight[0]=new double[argNumInputs+1][argNumHidden+1];
		this.weight[1]=new double[argNumHidden+1][1];
		this.weightChange=new double[2][][];
		this.weightChange[0]=new double[argNumInputs+1][argNumHidden+1];
		this.weightChange[1]=new double[argNumHidden+1][1];

		this.h =new double[argNumHidden+1];
		this.delta=new double[argNumHidden+1];
	}

	public double outputFor(double[] X) {		
		return 0;
	}

	public double train(double[] X, double argValue,String type) {
		double yhat=forward( X,type);
		if(type=="binary") {
			//yhat=sigmoid(sum);
			for(int i=0;i<argNumHidden+1;i++) {
				delta[i]=yhat*(1-yhat)*(argValue-yhat);				
				weight[1][i][0]+=argLearningRate*delta[i]*h[i]+argMomentumTerm*weightChange[1][i][0];
				weightChange[1][i][0]=argLearningRate*delta[i]*h[i]+argMomentumTerm*weightChange[1][i][0];
				System.out.println(weightChange[1][i][0]);
			}
			for(int i=0;i<argNumInputs+1;i++) {
				for(int j=0;j<argNumHidden+1;j++) {
					weight[0][i][j]+=argLearningRate*h[j]*(1-h[j])*delta[j]*weight[1][j][0]*input[i]+argMomentumTerm*weightChange[0][i][j];
					weightChange[0][i][j]=argLearningRate*h[j]*(1-h[j])*delta[j]*weight[1][j][0]*input[i]+argMomentumTerm*weightChange[0][i][j];
				}
			}
		}
		else {
			//yhat=customSigmoid(sum);//customSigmoid
			for(int i=0;i<argNumHidden+1;i++) {
				delta[i]=(argB-argA)*0.25*(1-Math.pow(yhat,2))*(argValue-yhat);   //0.5*(1-Math.pow(yhat,2))*(argValue-yhat);
				//System.out.println("delta[ "+i+"]:"+delta[i]);
				weight[1][i][0]+=argLearningRate*delta[i]*h[i]+argMomentumTerm*weightChange[1][i][0];
				weightChange[1][i][0]=argLearningRate*delta[i]*h[i]+argMomentumTerm*weightChange[1][i][0];
			}
			for(int i=0;i<argNumInputs+1;i++) {
				for(int j=0;j<argNumHidden;j++) {					
					weight[0][i][j]+=argLearningRate*(argB-argA)*0.25*(1-Math.pow(h[j],2))*delta[j]*weight[1][j][0]*input[i]+argMomentumTerm*weightChange[0][i][j];
					//System.out.println("weight[0]["+i+"]["+j+"]:"+argLearningRate*0.5*(1-Math.pow(h[j],2))*delta[j]*weight[1][j][0]*input[i]);
					weightChange[0][i][j]=argLearningRate*(argB-argA)*0.25*(1-Math.pow(h[j],2))*delta[j]*weight[1][j][0]*input[i]+argMomentumTerm*weightChange[0][i][j];
					//System.out.println("weightChange[0]["+i+"]["+j+"]:"+argLearningRate*0.5*(1-Math.pow(h[j],2))*delta[j]*weight[1][j][0]*input[i]);
				}
			}
		}
		/*
		for(int i=0;i<argNumInputs;i++) {
			System.out.println(Arrays.toString(weightChange[0][i])); 
		}
		for(int i=0;i<argNumHidden;i++) {
			System.out.println(Arrays.toString(weightChange[1][i])); 
		}*/
		return Math.pow(argValue-yhat, 2);//0.5*
	}

	public double forward(double[] X, String type){
		//double []h =new double[argNumHidden+1],delta=new double[argNumHidden+1];

		double yhat=0,error=0;
		input = Arrays.copyOf(X, X.length+1);//数组扩容
		input[input.length-1]=bias;  
		for(int j=0;j<this.argNumHidden;j++) {
			for(int i=0;i<this.argNumInputs+1;i++) {
				h[j]+=input[i]*weight[0][i][j];				
			}
			if(type=="binary")
				h[j]=sigmoid(h[j]);
			else
				h[j]=customSigmoid(h[j]);//customSigmoid
		}		
		h[argNumHidden]=bias;
		double sum=0;
		for(int i=0;i<argNumHidden+1;i++) {			
			sum+=h[i]*weight[1][i][0];			
		}
		return customSigmoid(sum);
	}

	public void save(File argFile) {
		// TODO Auto-generated method stub		
	}


	public void load(String argFileName) throws IOException {
		// TODO Auto-generated method stub		
	}


	public double sigmoidBipolar(double x) {		
		return 2 / (1 + Math.exp(-x))-1;
	}
	public double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	public double relu(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public double customSigmoid(double x) {
		return (argB-argA) / (1 + Math.exp(-x))+argA;
	}

	public void initializeWeights() {
		for(int i=0;i<=this.argNumInputs;i++) {
			for(int j=0;j<=this.argNumHidden;j++) {
				weight[0][i][j]=Math.random()*2-1;
			}
		}
		for(int i=0;i<=this.argNumHidden;i++) {
			weight[1][i][0]=Math.random()*2-1;//Math.random()-0.5;
		}
		System.out.println("INNNNNNNNNNNNNNNNNitalizing weights"); 
	}
	public void loadWeights(File file) {
		if(!file.exists())
			initializeWeights(); 
		
		BufferedReader read = null; 
		try{ 
			read = new BufferedReader(new FileReader(file)); 
			for(int i=0;i<weight.length;i++)
				for(int j=0;j<weight[i].length;j++) 
					for(int k=0;k<weight[i][j].length;k++){
						weight[i][j][k] = Double.parseDouble(read.readLine());
					}
		}
		catch (IOException e){ 
			System.out.println("loadWeights. IOException trying to open reader: " + e);			
		} 
		catch (NumberFormatException e){ 
		} 
		finally{ 
			try{ 
				if (read != null) 
					read.close(); 
			} 
			catch (IOException e) 
			{ 
				  System.out.println("IOException trying to close reader: " + e); 
			} 
		}		
	}
	public void saveWeights(File file) { 
		int i,j=0,count=0;
		PrintStream write = null; 
		try{ 
			write = new PrintStream(new RobocodeFileOutputStream(file)); 
			count++;
			System.out.println("count is: "+count);
			for ( i = 0; i < weight.length; i++) 
				for (j = 0; j < weight[i].length; j++) 
					for(int k=0;k<weight[i][j].length;k++){
						write.println(weight[i][j][k]); //write.println(new Double(table[i][j]));				
				}
			System.out.println("i is"+i+"j is "+j);
			if (write.checkError()) 
				System.out.println("Could not save the data!"); 
			write.close(); 
		} 
		catch (IOException e){ 
			   System.out.println("IOException trying to write: " + e); 
		} 
		finally{ 
			try{ 
				if (write != null) 
					write.close(); 
			} 
			catch (Exception e){ 
				     System.out.println("Exception trying to close witer: " + e); 
			} 
		} 
	}

	public void zeroWeights() {
		// TODO Auto-generated method stub		
	}

	double forward_with_action(double[] state, double i, String string) {
		double[] input = Arrays.copyOf(state, state.length+1);//数组扩容
		input[input.length-1]=(i-3)/3; 
		return forward(input,string);
	}
	double train_with_action(double[] state, double i,double argY, String string) {
		double[] input = Arrays.copyOf(state, state.length+1);//数组扩容
		input[input.length-1]=i; 
		return train(input,argY,string);
	}
	//getHeading()/180-1,target.distance/500-1,target.bearing/180-1, reward);		
}
