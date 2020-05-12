package a2;
import java.awt.*;   
import java.awt.geom.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;


import robocode.*;   

public class updateRobotNN extends AdvancedRobot   
{   
	public static final double PI = Math.PI;   
	private Target target;   
	private LUT table;   
	private NeuralNet NN; 
	private Learner learner;  
	private boolean found=false;  
	private double reward = 0.0;   
	private double firePower;   
	private int direction = 1;   
	private int isHitWall = 0;   
	private int isHitByBullet = 0;  
	private int winFlag = 0; 
	private static int countForWin=0;
	double rewardForWin=100;//100
	double rewardForDeath=-20;//-20
	double accumuReward=0.0;
	private static int count=0;	
	private int lastAction=4;	
	private double[] NN_last_states = new double[5]; 
	private boolean NNFlag=true;
	private int action;
	private static final double NN_alpha = 0.3; 
	private double NN_last_action;
	private double NN_lambda=0.9;
	private double NN_epsilon=0;
	double [] NN_current_states;

	public void run()   
	{   
		target = new Target();   
		target.distance = 1000;  
		if(NNFlag==false){
			table = new LUT();   
			loadData();   
			learner = new Learner(table);   
		} 
		else{
			NN=new NeuralNet(6, 23, 0.02, 0.2, -1, 1);		
			System.out.println("NNNNNNNNNNNNNNNNNN6, 12, 0.001, 0.8, -1, 1"); 
			//NN.initializeWeights();	
			loadWeights(); 
			action=selectActionNN(getState_numerical(), reward);
		}

		setColors(Color.green, Color.white, Color.green);   
		setAdjustGunForRobotTurn(true);   
		setAdjustRadarForGunTurn(true); 
		setAdjustRadarForRobotTurn(true);  
		turnRadarRightRadians(2 * PI);   
		while (true)   
		{ 
			if(getRoundNum()<200)//
				learner.explorationRate=0.0;
			else
				learner.explorationRate=0.0;
			robotMovement(); 
			execute();   
		}   
	}   


	private void robotMovement()   
	{   
		int state = getState();
		reward = 0.0;
		if(NNFlag==false){			   
			action = learner.selectAction(state);
			learner.learn(state, action, reward);   
		}
		else {
			action=selectActionNN(getState_numerical(), reward);
		}		 
		
		switch (action)   
		{   
		case Action.ahead:   
			setAhead(Action.aheadDistance);   
			break;   
		case Action.back:   
			setBack(Action.aheadDistance);   
			break;   
		case Action.aheadLeft:   
			setAhead(Action.aheadDistance);
			setTurnLeft(Action.turnDegree);      
			break;   
		case Action.aheadRight: 
			setAhead(Action.aheadDistance);
			setTurnRight(Action.turnDegree);    
			break; 
		case Action.fireOne:  
			findTargetFire();	
			if (getGunHeat() == 0) {   
				setFire(1);   
			}
			break;		
		case Action.fireTwo:  
			findTargetFire();	
			if (getGunHeat() == 0) {   
				setFire(2);   
			}
			break;	
		case Action.fireThree:  
			findTargetFire();	
			if (getGunHeat() == 0) {   
				setFire(3);   
			}
			break;	
		}   
}    
	private void findTargetFire() {
		found=false;
		while(!found) {
			setTurnRadarLeft(360);
			execute();
		}
		double gunOffset=(getGunHeading()-getHeading())/360*2*PI-target.bearing;
		setTurnGunLeftRadians(NormaliseBearing(gunOffset));
		execute();		
	}
	
	
	private int getState()   
	{   
		int heading = State.getHeading(getHeading());  
		int XPosition = State.getXPosition(getX());
		int YPosition = State.getYPosition(getY());		
		int targetDistance = State.getTargetDistance(target.distance);   
		int targetBearing = State.getTargetBearing(target.bearing);  
		int energy = State.getEnergy(getEnergy()); 
		 
		int state = State.Mapping[heading][targetDistance][targetBearing][XPosition][YPosition];   //[isHitWall]
		return state;   
	}   
	private double[] getState_numerical()   
	{   
		double heading = getHeading();  
		double XPosition = getX();
		double YPosition = getY();		
		double targetDistance = target.distance;   
		double targetBearing = target.bearing;  
		double[] state= {heading,targetDistance,targetBearing,XPosition,YPosition,};		
		return state;   
	}
	 
	double NormaliseBearing(double ang){

		if (ang > PI)   
			ang -= 2*PI;   
		if (ang < -PI)    
			ang += 2*PI;   
		return ang;   
	}   

	//heading within the 0 to 2pi range   
	double NormaliseHeading(double ang) {   
		if (ang > 2*PI)   
			ang -= 2*PI;   
		if (ang < 0)   
			ang += 2*PI;   
		return ang;   
	}   

	//returns the distance between two x,y coordinates   
	public double getrange( double x1,double y1, double x2,double y2 )   
	{   
		double xo = x2-x1;   
		double yo = y2-y1;   
		double h = Math.sqrt( xo*xo + yo*yo );   
		return h;   
	}   

	//gets the absolute bearing between to x,y coordinates   
	public double absbearing( double x1,double y1, double x2,double y2 )   
	{  
		double xo = x2-x1;   
		double yo = y2-y1;   
		double h = getrange( x1,y1, x2,y2 );   
		if( xo > 0 && yo > 0 )   
		{   
			return Math.asin( xo / h );   
		}   
		if( xo > 0 && yo < 0 )   
		{   
			return Math.PI - Math.asin( xo / h );   
		}   
		if( xo < 0 && yo < 0 )   
		{   
			return Math.PI + Math.asin( -xo / h );   
		}   
		if( xo < 0 && yo > 0 )   
		{   
			return 2.0*Math.PI - Math.asin( -xo / h );   
		}   
		return 0;   
	}   



	public void onBulletHit(BulletHitEvent e)   
	{  

		if (target.name == e.getName())   
		{     
			double change = e.getBullet().getPower() * 9;   
			out.println("Bullet Hit: " + change);  
			if(NNFlag==false) {
				int state = getState();   
				int action = learner.selectAction(state);  
				learner.learn(state, action, change);  
			}
			else {
				action=selectActionNN(getState_numerical(), reward);
			}
		}   
	}   


	public void onBulletMissed(BulletMissedEvent e)   
	{   
		double change = -e.getBullet().getPower();   
		out.println("Bullet Missed: " + change);  
		if(NNFlag==false) {
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, change);  
		}
		else {
			action=selectActionNN(getState_numerical(), reward);
		} 
	}   

	public void onHitByBullet(HitByBulletEvent e)   
	{   
		double power = e.getBullet().getPower();   
		double change = -(4 * power + 2 * (power - 1));   
		out.println("Hit By Bullet: " + change);   
		if(NNFlag==false) {
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, change);  
		}
		else {
			action=selectActionNN(getState_numerical(), reward);
		}   
	}   

	public void onHitRobot(HitRobotEvent e)   
	{   		
		double change = -6.0;   
		out.println("Hit Robot: " + change);   
		if(NNFlag==false) {
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, change);  
		}
		else {
			action=selectActionNN(getState_numerical(), reward);
		} 
	}   

	public void onHitWall(HitWallEvent e)   
	{   
		double change = -(Math.abs(getVelocity()) * 0.5 );   
		out.println("Hit Wall: " + change);   
		if(NNFlag==false) {
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, change);  
		}
		else {
			action=selectActionNN(getState_numerical(), reward);
		} 
	}   

	public void onScannedRobot(ScannedRobotEvent e)   
	{   
		if ((e.getDistance() < target.distance)||(target.name == e.getName()))   
		{   
			found=true;
			//the next line gets the absolute bearing to the point where the bot is   
			double absbearing_rad = (getHeadingRadians()+e.getBearingRadians())%(2*PI);   
			//this section sets all the information about our target   
			target.name = e.getName();   
			double h = NormaliseBearing(e.getHeadingRadians() - target.head);   
			h = h/(getTime() - target.ctime);   
			target.changehead = h;   
			target.x = getX()+Math.sin(absbearing_rad)*e.getDistance(); //works out the x coordinate of where the target is   
			target.y = getY()+Math.cos(absbearing_rad)*e.getDistance(); //works out the y coordinate of where the target is   
			target.bearing = e.getBearingRadians();   
			target.head = e.getHeadingRadians();   
			target.ctime = getTime();             //game time at which this scan was produced   
			target.speed = e.getVelocity();   
			target.distance = e.getDistance();   
			target.energy = e.getEnergy(); 
		}   
	}   

	public void onRobotDeath(RobotDeathEvent e)   
	{   

		if (e.getName() == target.name)   
		{
			target.distance = 1000; 
		}

	}   

	public void onWin(WinEvent event)   
	{   
		winFlag=1;
		File file = getDataFile("accumReward1.dat");
		if(NNFlag==false) {
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, rewardForWin);  
			saveData();
		}
		else {
			action=selectActionNN(getState_numerical(), rewardForWin);
			saveWeights();
		}   		 
		saveResult2(winFlag);
	}   
	
	public void saveResult2(int winFlag) {
		if(true) {
			File file = getDataFile("saveResult1.dat");
			PrintStream w = null; 
			try 
			{ 
				w = new PrintStream(new RobocodeFileOutputStream(file.getAbsolutePath(), true)); 
				w.println(winFlag); 
				if (w.checkError()) 
					System.out.println("Could not save the data!");  //setTurnLeft(180 - (target.bearing + 90 - 30));
				w.close(); 
			}

			catch (IOException e1) 
			{ 
				System.out.println("IOException trying to write: " + e1); 
			} 
			finally 
			{ 
				try 
				{ 
					if (w != null) 
						w.close(); 
				} 
				catch (Exception e2) 
				{ 
					System.out.println("Exception trying to close witer: " + e2); 
				}
			} 
		}
	}
	public void onDeath(DeathEvent event)   
	{   
		winFlag=0;
		accumuReward+=rewardForDeath;
		count++;
		if(NNFlag==false) {
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, rewardForDeath); 
			saveData(); 
		}
		else {
			action=selectActionNN(getState_numerical(), rewardForDeath);
			saveWeights();
		}   
		saveResult2(winFlag);
	}   

	public void loadData()   
	{   
		try   
		{   
			table.load(getDataFile("movement1.dat"));   
		}   
		catch (Exception e)   
		{   
			out.println("Exception trying to load: " + e); 
		}   
	}   

	public void saveData()   
	{   
		try   
		{   
			table.save(getDataFile("movement1.dat"));   
		}   
		catch (Exception e)   
		{   
			out.println("Exception trying to write: " + e);   
		}   
	} 
	public void loadWeights()   
	{   
		try   
		{   
			NN.loadWeights(getDataFile("weight.dat"));   
		}   
		catch (Exception e)   
		{   
			out.println("Exception trying to load: " + e); 
			NN.initializeWeights(); 
		}   
	}   

	public void saveWeights()   
	{   
		try   
		{   
			NN.saveWeights(getDataFile("weight.dat")); 
			out.println("saaaaaaaaaaaaaaaaaaavingWeights"); 
			
		}   
		catch (Exception e)   
		{   
			out.println("Exception trying to write: " + e);   
		}   
	} 
	private void initialNN(){
		double ErrorBound=0.05,totalError,RMS=0.0;		
		double []saveTotalError = null;
		double[][] x= {{0,1,2,3},{0,1,2,3,4,5,6,7,8,9},{0,1,2,3},{0,1,2,3,4,5,6,7},{0,1,2,3,4,5},{0,1,2,3,4,5,6}};
		double [][] x_input= getInput(x);
		double [][] X_input_NN=normalize(x);
		//loadData("/Users/anna321321/E/CPEN502/hw/a2/bin/a2/updateRobot.data/movement1.dat");
		double mean=85;
		double max=150;
		double min=-20;
				
		
		//File writename=new File("output.bat");
		//boolean createNewFile =writename.createNewFile();
		//BufferedWriter write=new BufferedWriter(new FileWriter(writename));
		int avgEpoch=0;
		for(int trials=0;trials<1;trials++) {		
			//write.write("Start trial="+String.valueOf(trials)+":\r\n");
			NN=new NeuralNet(6, 12, 0.001, 0.8, -1, 1);		
			NN.initializeWeights();	
			int epoch=0;
			for(epoch=0;epoch<1;epoch++) {
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
				//System.out.println("epoch="+epoch+", error="+RMS);
				//write.write(String.valueOf(RMS)+"\r\n");
//				if(totalError<ErrorBound||epoch>40000) {
//					System.out.println("error="+totalError);					
//					break;
//				}
				//write.flush();
			}
			//write.write("End trail="+trials+".\r\n");
			avgEpoch+=epoch;
		}
		avgEpoch=avgEpoch/1000;
		System.out.println("avgEpoch="+avgEpoch);
//		write.write("avgEpoch="+avgEpoch);
//		write.flush();
//		write.close();		
	}  
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
								table[i]= new double[] {(x[0][a]-1.5)/1.5,(x[1][b]-4.5)/4.5,(x[2][c]-1.5)/1.5,(x[3][d]-3.5)/3.5,(x[4][e]-2.5)/2.5,(x[5][f]-3)/3};
								i++;
							}
						}
					}
				}
			}
		}
		return table;		
	}
	public void loadData(String path)   
	{   
		File file=new File(path);
		try   
		{   
			table.load(file);   
		}   
		catch (Exception e)   
		{   
		} 
	}
	public int selectActionNN(double [] state,double reward){
		state=normalizeState(state);
		int action=4;
		double qmax=Double.NEGATIVE_INFINITY;
		for(int i=0;i<7;i++){			 
			double q=NN.forward_with_action(state,(double)i,"bipolar");
			if (q>qmax){
				qmax=q;
				action=i;			
			}
		}
		double NN_Q_new=NN.forward_with_action(state,(double)action,"bipolar");//myNet[action].outputFor(NN_current_states);
    	double error_signal = 0;
    	double old_Q=NN.forward_with_action(NN_last_states,(double)NN_last_action,"bipolar");
    	error_signal = NN_alpha*((reward-85)/170 + NN_lambda * NN_Q_new - old_Q); //myNet[NN_last_action].outputFor(NN_last_states));
    	  
    	  
		//newRobot1.total_error_in_one_round += error_signal*error_signal/2;
		double correct_old_Q = old_Q + error_signal;
		if((action==6)&&(state[1]<0.1&&state[1]>-0.1)&&(state[0]<0.1&&state[0]>-0.1)&&(state[2]<0.1&&state[2]>-0.1)) {
			saveActionStatePairError(error_signal);
		}
		NN.train_with_action(NN_last_states,(double)NN_last_action,(correct_old_Q-85)/170,"bipolar");//myNet[NN_last_action].train(NN_last_states, correct_old_Q); 
		if((Math.random() < NN_epsilon) && (getRoundNum()<1000) )
		{
		  action = new Random().nextInt(Action.numActions);
		}
		for(int i=0; i<5; i++)
		{
		  NN_last_states[i] = state[i];
		}
		NN_last_action=action;
		//out.println("action is SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSs"+action);
		return action;
	}
	public double [] normalizeState(double[] state){
		state= new double[] {(state[0]-180)/180,(state[1]-500)/500,(state[2])/180,(state[3]-400)/400,(state[4]-300)/300};							
		return state;
	}
	public void saveActionStatePairError(double error) {
		if(true) {
			File file = getDataFile("ActionStatePairError.dat");
			PrintStream w = null; 
			try 
			{ 
				w = new PrintStream(new RobocodeFileOutputStream(file.getAbsolutePath(), true)); 
				w.println(error); 
				if (w.checkError()) 
					System.out.println("Could not save the data!");  //setTurnLeft(180 - (target.bearing + 90 - 30));
				w.close(); 
			}

			catch (IOException e1) 
			{ 
				System.out.println("IOException trying to write: " + e1); 
			} 
			finally 
			{ 
				try 
				{ 
					if (w != null) 
						w.close(); 
				} 
				catch (Exception e2) 
				{ 
					System.out.println("Exception trying to close witer: " + e2); 
				}
			} 
		}		
	}
}
