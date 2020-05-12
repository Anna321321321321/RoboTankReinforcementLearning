package a2;
import java.awt.*;   
import java.awt.geom.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import robocode.*;   

public class newRobot extends AdvancedRobot   
{   
	public static final double PI = Math.PI;   
	private Target target;   
	private LUT table;   
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

	public void run()   
	{   
		
		table = new LUT();   
		loadData();   
		learner = new Learner(table);   
		target = new Target();   
		target.distance = 1000;   


		setColors(Color.green, Color.white, Color.green);   
		setAdjustGunForRobotTurn(true);   
		setAdjustRadarForGunTurn(true); 
		setAdjustRadarForRobotTurn(true);  
		turnRadarRightRadians(2 * PI);   
		while (true)   
		{ 
			if(getRoundNum()<200)//
			learner.explorationRate=0.9;//0.5
//			else if(getRoundNum()<getNumRounds()/5)///3
//				learner.explorationRate=0.0;//0.2
			else
				learner.explorationRate=0.9;
			robotMovement(); 
			execute();   

		}   
	}   


	private void robotMovement()   
	{   
		int state = getState();   
		int action = learner.selectAction(state); 

		learner.learn(state, action, reward);  
		//accumuReward+=reward;
		reward = 0.0;   

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
			accumuReward += change;  
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, change);  
		}   
	}   


	public void onBulletMissed(BulletMissedEvent e)   
	{   
		double change = -e.getBullet().getPower();   
		out.println("Bullet Missed: " + change);   

		accumuReward += change;  
		int state = getState();   
		int action = learner.selectAction(state);  
		learner.learn(state, action, change);  
	}   

	public void onHitByBullet(HitByBulletEvent e)   
	{   
		if (target.name == e.getName())   
		{   
			double power = e.getBullet().getPower();   
			double change = -(4 * power + 2 * (power - 1));   
			out.println("Hit By Bullet: " + change);   
			accumuReward += change; 
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, change);  
		}   
		isHitByBullet = 1;   
	}   

	public void onHitRobot(HitRobotEvent e)   
	{   
		if (target.name == e.getName())   
		{   
			double change = -6.0;   
			out.println("Hit Robot: " + change);   
			accumuReward += change;   
			int state = getState();   
			int action = learner.selectAction(state);  
			learner.learn(state, action, change);  
		}   
	}   

	public void onHitWall(HitWallEvent e)   
	{   

		double change = -(Math.abs(getVelocity()) * 0.5 );   
		out.println("Hit Wall: " + change);   
		accumuReward += change;   
		isHitWall = 1;   
		int state = getState();   
		int action = learner.selectAction(state);  
		learner.learn(state, action, change);  
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
		accumuReward+=rewardForWin;
		int state = getState();   
		int action = learner.selectAction(state);  
		learner.learn(state, action, rewardForWin);  
		saveData(); 
		countForWin++;
		count++;
		PrintStream w = null; 
		try 
		{ 

			w = new PrintStream(new RobocodeFileOutputStream(file.getAbsolutePath(), true)); 
			if(count==20){
				count=0;
				w.println(Math.abs(accumuReward)+" "+countForWin*5+" "+" "+learner.explorationRate); 
				accumuReward=0; 
				countForWin=0;
				if (w.checkError()) 
					System.out.println("Could not save the data!");  //setTurnLeft(180 - (target.bearing + 90 - 30));
				w.close(); 
			}
		} 
		catch (IOException e) 
		{ 
			System.out.println("IOException trying to write: " + e); 
		} 
		finally 
		{ 
			try 
			{ 
				if (w != null) 
					w.close(); 
			} 
			catch (Exception e) 
			{ 
				System.out.println("Exception trying to close witer: " + e); 
			} 

		} 
		saveResult2(winFlag);
	}   
	public void saveResult(BattleEndedEvent e) {
		if(getRoundNum()%50==0) {
			File file = getDataFile("saveResult1.dat");
			int a=e.getResults().getFirsts();
			PrintStream w = null; 
			try 
			{ 
				w = new PrintStream(new RobocodeFileOutputStream(file.getAbsolutePath(), true)); 
				w.println(a); 
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
		int state = getState();   
		int action = learner.selectAction(state);  
		learner.learn(state, action, rewardForDeath);  

		saveData();   
		File file = getDataFile("accumReward1.dat"); 
		PrintStream w = null; 
		try 
		{ 

			w = new PrintStream(new RobocodeFileOutputStream(file.getAbsolutePath(), true)); 
			if(count==20){
				count=0;
				w.println(Math.abs(accumuReward)+" "+countForWin*5+" "+" "+learner.explorationRate); 
				accumuReward=0;
				countForWin=0;
				if (w.checkError()) 
					System.out.println("Could not save the data!"); 
				w.close(); 
			}
		} 
		catch (IOException e) 
		{ 
			System.out.println("IOException trying to write: " + e); 
		} 
		finally 
		{ 
			try 
			{ 
				if (w != null) 
					w.close(); 
			} 
			catch (Exception e) 
			{ 
				System.out.println("Exception trying to close witer: " + e); 
			} 

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
}  
