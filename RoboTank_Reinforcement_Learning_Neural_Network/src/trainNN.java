package a2;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
//import org.jfree.chart.JFreeChart;

public class trainNN {
	private  static LUT table=new LUT();
	public static void main(String[]args) throws IOException{
		double ErrorBound=0.05,totalError,RMS=0.0;		
//		int table[][]=new int [4][2];
//		int[] yBinary= {0,1,1,0};
//		int[] yBipolar= {-1,1,1,-1};
		double []saveTotalError = null;
//		double[][] xBinary= {{0,0},{0,1},{1,0},{1,1}};
//		double[][] xBipolar= {{-1,-1},{-1,1},{1,-1},{1,1}};
		double[][] x= {{0,1,2,3},{0,1,2,3,4,5,6,7,8,9},{0,1,2,3},{0,1,2,3,4,5,6,7},{0,1,2,3,4,5},{0,1,2,3,4,5,6}};//{-1,1},{-1,1},{-1,1},{-1,1},{-1,1},{-1,1},{-1,1}
		double [][] x_input= getInput(x);
		double [][] X_input_NN=normalize(x);
		loadData("/Users/anna321321/E/CPEN502/hw/backup/updateRobot.data/movement1.dat");
		double mean=85;
		double max=150;
		double min=-20;
				
		
		
		int epochmin=10000;
		int minEpoch=10000,maxEpoch=0;
		
		File writename=new File("output.txt");
		boolean createNewFile =writename.createNewFile();
		BufferedWriter write=new BufferedWriter(new FileWriter(writename));
	
		
		int avgEpoch=0;
		for(int trials=0;trials<1;trials++) {		
			write.write("Start trial="+String.valueOf(trials)+":\r\n");
			NeuralNet NN=new NeuralNet(6, 4, 0.01, 0.1, -1, 1);	//, 23, 0.01, 0.3, -1, 1	
			NN.initializeWeights();	
			int epoch=0;
			for(epoch=0;;epoch++) {				
				totalError=0;
				for(int i=0;i<x_input.length;i++) {
					for(int j=0;j<NN.argNumInputs;j++) {
						//totalError+=NN.train(xBinary[i],yBinary[i],"binary");
						int state =State.Mapping[(int) x_input[i][0]][(int)x_input[i][1]][(int)x_input[i][2]][(int)x_input[i][3]][(int)x_input[i][4]]; // x_input[i][1] x_input[i][2] ;//;						
						int action=(int)x_input[i][x.length-1];
						totalError+=NN.train(X_input_NN[i],(table.table[state][action]-mean)*2/(max-min),"bipolar");
					}
					RMS=Math.sqrt(totalError/x_input.length);
				}
				System.out.println("epoch="+epoch+", error="+RMS);
				File weightname=new File("weight.txt");
				boolean createNewWeightFile =weightname.createNewFile();
				BufferedWriter weight=new BufferedWriter(new FileWriter(weightname));
				for(int i=0;i<NN.weight.length;i++)
					for(int j=0;j<NN.weight[i].length;j++) 
						for(int k=0;k<NN.weight[i][j].length;k++){
							weight.write(String.valueOf(NN.weight[i][j][k])+"\r\n");					
					}
				write.write(String.valueOf(RMS)+"\r\n");					
				if(totalError<ErrorBound||epoch>20000) {
					System.out.println("error="+totalError);					
					break;
				}
				write.flush();
				weight.flush();
				weight.close();
			}
			write.write("End trail="+trials+".\r\n");
			avgEpoch+=epoch;
		}
		avgEpoch=avgEpoch/1000;
		System.out.println("avgEpoch="+avgEpoch);
		write.write("avgEpoch="+avgEpoch);
		//totalError=0;
		/*
		if (epoch >= 10000) {
            ++counter;
        } else {
            counter2 += epoch;
            if (epoch < countermin) {
                countermin = epoch;
            }
            if (epoch > countermax) {
                countermax = epoch;
            }
        }*/
		write.flush();
		write.close();	
	}
//	private int getState()   
//	{   
//		int heading = State.getHeading(getHeading());  
//		int XPosition = State.getXPosition(getX());
//		int YPosition = State.getYPosition(getY());		
//		int targetDistance = State.getTargetDistance(target.distance);   
//		int targetBearing = State.getTargetBearing(target.bearing);  
//		int energy = State.getEnergy(getEnergy()); 
//		 
//		int state = State.Mapping[heading][targetDistance][targetBearing][XPosition][YPosition];   //[isHitWall]
//		return state;   
//	} 
	private static double [][] getInput(double [][] x){
		double [][] table=new double[53760][6]; //=
		int i=0;
		for (int a=0;a<x[0].length;a++) {
			for(int b=0;b<x[1].length;b++) {
				for(int c=0;c<x[2].length;c++) {
					for(int d=0;d<x[3].length;d++) {
						for(int e=0;e<x[4].length;e++) {
							for(int f=0;f<x[5].length;f++) {
								table[i]= new double[] {x[0][a],x[1][b],x[2][c],x[3][d],x[4][e],x[5][f]};
								i++;
							}
						}
					}
				}
			}
		}
		return table;		
	}
	private static double [][] normalize(double [][] x){
		double [][] table=new double[53760][6]; //=
		int i=0;
		for (int a=0;a<x[0].length;a++) {
			for(int b=0;b<x[1].length;b++) {
				for(int c=0;c<x[2].length;c++) {
					for(int d=0;d<x[3].length;d++) {
						for(int e=0;e<x[4].length;e++) {
							for(int f=0;f<x[5].length;f++) {
								table[i]= new double[] {(x[0][a]-1.5)/1.5,(x[1][b]-4.5)/4.5,(x[2][c]-1.5)/1.5,(x[3][d]-3.5)/3.5,(x[4][e]-2.5)/2.5,(x[5][f]-3)};
								i++;
							}
						}
					}
				}
			}
		}
		return table;		
	}
	
	public static void loadData(String path)   
	{   
		File file=new File(path);
//        if(!file.exists()||file.isDirectory())
//            throw new FileNotFoundException();
//        BufferedReader br=new BufferedReader(new FileReader(file));
		try   
		{   
			table.load(file);   
		}   
		catch (Exception e)   
		{   
		}   
	}
}
