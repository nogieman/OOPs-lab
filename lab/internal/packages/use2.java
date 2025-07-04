import calculator.calc;
import java.util.*;
class calcu{
  public static void main(String []adsa){
    calc c = new calc();
    Scanner s = new Scanner(System.in);
    System.out.println("Enter two decimal numbers: ");
    double a = s.nextDouble();
    double b = s.nextDouble();
    c.add(a,b);
    c.diff(a,b);
    c.prod(a,b);
    c.div(a,b);
    c.pow(a,b);
  }
}
