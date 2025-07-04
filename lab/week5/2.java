import java.util.*;
class rad{
  static double circleArea( double r){
    double p = Math.PI;
    return (p*(Math.pow(r,2)));
  }
  static double circlePerimeter(double r){
    double p = Math.PI;
    return (2*p*r);
  }
  public static void main(String []adsa){
    System.out.println("Enter radius of circle");
    Scanner s = new Scanner(System.in);
    double r = s.nextDouble();
    double ar = circleArea(r);
    double pe = circlePerimeter(r);
    System.out.println("The area = "+ar+" & perimeter is: "+pe);
    
  }
}
