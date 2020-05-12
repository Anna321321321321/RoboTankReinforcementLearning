package a2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import robocode.*;
import robocode.Robot;

public class LUT implements LUTInterface{
	public double[][] table; 
	 
	public LUT(){ 
		table = new double[State.numStates][Action.numActions]; 
		initialiseLUT(); // set all function values to be zero
	} 
	public double max_Q(int state) //return the maximum value in the LUT
	  { 
	    double maxinum = Double.NEGATIVE_INFINITY; 
	    for (int i = 0; i < table[state].length; i++){ 
	    	if (table[state][i] > maxinum) 
	        maxinum = table[state][i]; 
	    } 
	    return maxinum; 
	  }
	@Override
	public double outputFor(double[] X) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double train(double[] X, double argValue, String type) {
		// TODO Auto-generated method stub
		return 0;
	}
	public int getBestAction(int state){ 
	    double maxinum = Double.NEGATIVE_INFINITY; 
	    int bestAction = 0; 
	    for (int i = 0; i < table[state].length; i++) 
	    {  
	      if (table[state][i] > maxinum){ 
	        maxinum = table[state][i]; 
	        bestAction = i; 
	      } 
	    } 
	    return bestAction; 
	} 
	public double getQValue(int state, int action){ 
	    return table[state][action]; 
	}	 
	public void setQValue(int state, int action, double value){ 
	    table[state][action] = value; 
	} 
	public void load(File file){ 
	    BufferedReader read = null; 
	    try{ 
	      read = new BufferedReader(new FileReader(file)); 
	      for (int i = 0; i < State.numStates; i++) 
	        for (int j = 0; j < Action.numActions; j++) {
	          table[i][j] = Double.parseDouble(read.readLine()); 
	        }
	    } 
	    catch (IOException e){ 
	      System.out.println("initialiseLUT. IOException trying to open reader: " + e); 
	      initialiseLUT(); 
	    } 
	    catch (NumberFormatException e){ 
	      initialiseLUT(); 
	    } 
	    finally{ 
	    	try{ 
	    		if (read != null) 
	    			read.close(); 
	    	} 
	    	catch (IOException e) 
	    	{ 
	    		//   System.out.println("IOException trying to close reader: " + e); 
	    	} 
	    } 
	 }   
 
	@Override
	public void save(File file) { 
		int i,j=0,count=0;
		PrintStream write = null; 
		try{ 
		write = new PrintStream(new RobocodeFileOutputStream(file)); 
		count++;
		System.out.println("count is: "+count);
		for ( i = 0; i < State.numStates; i++) 
			for (j = 0; j < Action.numActions; j++) {
				//System.out.println("i:"+i+"j"+j);
				write.println(new Double(table[i][j])); //write.println(new Double(table[i][j]));				
			}
		System.out.println("i is"+i+"j is "+j);
		if (write.checkError()) 
			System.out.println("Could not save the data!"); 
		write.close(); 
		} 
		catch (IOException e){ 
			//   System.out.println("IOException trying to write: " + e); 
		} 
		finally{ 
			try{ 
				if (write != null) 
					write.close(); 
			} 
			catch (Exception e){ 
				//      System.out.println("Exception trying to close witer: " + e); 
			} 
		} 
	}  

	@Override
	public void initialiseLUT(){
		for (int i = 0; i < State.numStates; i++) 
			for (int j = 0; j < Action.numActions; j++) 
				table[i][j] = 0.0; 
		
	}

	@Override
	public int indexFor(double[] X) {
		// TODO Auto-generated method stub
		return 2;
	}
	
}
