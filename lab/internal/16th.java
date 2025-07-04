import java.util.*;
interface Fare{
  double getAmount(double a, double b);
}
class Travel implements Fare{
  public static void main(String ards[]){
    Travel o = new Travel();
    Scanner sc = new Scanner(System.in);
    p("Enter distance in km");
    double d = sc.nextDouble();
    p("Enter cost per km");
    double c = sc.nextDouble();
    o.train(c,d);
    o.bus(c,d);
  }
  public double getAmount(double c, double d){
    return c*d;
  }
  void train(double c, double d){
    double t = (getAmount(c,d))*0.7;
    System.out.println("Total cost by train is: "+t);
  }
  void bus(double c, double d){
    double t = getAmount(c,d);
    System.out.println("Total cost by bus is: "+t);
  }
  static void p(String g){
    System.out.println(g);
  }
  
}
