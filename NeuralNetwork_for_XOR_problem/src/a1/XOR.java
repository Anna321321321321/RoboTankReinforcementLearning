package a1;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
//import org.jfree.chart.JFreeChart;

public class XOR {
	public static void main(String[]args) throws IOException{
		double ErrorBound=0.05,totalError;		
		int table[][]=new int [4][2];
		int[] yBinary= {0,1,1,0};
		int[] yBipolar= {-1,1,1,-1};
		double []saveTotalError = null;
		double[][] xBinary= {{0,0},{0,1},{1,0},{1,1}};
		double[][] xBipolar= {{-1,-1},{-1,1},{1,-1},{1,1}};
		int epochmin=10000;
		int minEpoch=10000,maxEpoch=0;
		
		File writename=new File("output.txt");
		boolean createNewFile =writename.createNewFile();
		BufferedWriter write=new BufferedWriter(new FileWriter(writename));
		int avgEpoch=0;
		for(int trials=0;trials<1000;trials++) {		
			write.write("Start trial="+String.valueOf(trials)+":\r\n");
			NeuralNet NN=new NeuralNet(2, 4, 0.2, 0.9, 1, 1);		
			NN.initializeWeights();	
			int epoch=0;
			for(epoch=0;;epoch++) {
				totalError=0;
				for(int i=0;i<4;i++) {
					//totalError+=NN.train(xBinary[i],yBinary[i],"binary");					
					totalError+=NN.train(xBipolar[i],yBipolar[i],"bipolar");
				}
				System.out.println("epoch="+epoch+", error="+totalError);
				write.write(String.valueOf(totalError)+"\r\n");
				if(totalError<ErrorBound||epoch>10000) {
					System.out.println("error="+totalError);					
					break;
				}
				write.flush();
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
}
