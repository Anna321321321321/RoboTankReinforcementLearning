package a2;

public class State  
{  
  public static final int numHeading = 4;  
  public static final int numTargetDistance = 10;  
  public static final int numTargetBearing = 4;  
  public static final int XPosition = 8; 
  public static final int YPosition = 6; 
  //public static final int numEnergy =3 ; 
  public static final int numStates;  
  public static final int Mapping[][][][][];  
  
  static  
  {  
    Mapping = new int[numHeading][numTargetDistance][numTargetBearing][XPosition][YPosition];//[numHitWall] 
    int count = 0;  
    for (int a = 0; a < numHeading; a++)  
      for (int b = 0; b < numTargetDistance; b++)  
        for (int c = 0; c < numTargetBearing; c++)  
          for (int d = 0; d < XPosition; d++)              
            for (int e = 0; e < YPosition; e++) 
            	//for (int f = 0; f < numEnergy; f++)
          Mapping[a][b][c][d][e] = count++;  
  
    numStates = count;  
  }  
  
  public static int getHeading(double heading)  
  {  
	  double unit=(360/numHeading);
	  
	  return (int)((heading-1)/unit);
	  /*
    double angle = 360 / numHeading;  
    double newHeading = heading + angle / 2;  
    if (newHeading > 360.0)  
      newHeading -= 360.0;  
    return (int)(newHeading / angle);  */
  }  
  
  public static int getTargetDistance(double value)  
  {  
    int distance = (int)(value / 100.0); //30 
    if (distance > numTargetDistance - 1)  
      distance = numTargetDistance - 1;  
    return distance;  
  }  
  public static int getDistanceToWall(double x,double y)  
  {
	double dis=Math.min(Math.min(x, y),Math.min(800-x, 600-y));
    if (dis > 10.0)  
      return 1;  
    else
    	return 0;  
  }  
  
  public static int getTargetBearing(double bearing)  
  {  double unit=(360/numTargetBearing);
  
  	return (int)((bearing+180-1)/unit);
	  /*
    double PIx2 = Math.PI * 2;  
    if (bearing < 0)  
      bearing = PIx2 + bearing;  
    double angle = PIx2 / numTargetBearing;  
    double newBearing = bearing + angle / 2;  
    if (newBearing > PIx2)  
      newBearing -= PIx2;  
    return (int)(newBearing / angle);  */
  }  
  public static int getEnergy(double energy)  
  {  
    if(energy>80)
    	return 2;
    else if(energy>30)
    	return 1;
    else 
    return 0;  
  }
  public static int getXPosition(double XPosition)  
  {  
    return (int)((XPosition-1)/100);  
  }
  public static int getYPosition(double YPosition)  
  {  
    return (int)((YPosition-1)/100);  
  }    
} 
