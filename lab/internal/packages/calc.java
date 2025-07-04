package calculator;
public class calc{
  public void add(double a, double b){
    System.out.println("The sum is: "+(a+b));
  }
  public void diff(double a, double b){
    System.out.println("The difference is: "+(a-b));
  }
  public void prod(double a, double b){
    System.out.println("The product is: "+(a*b));
  }
  public void div(double a, double b){
    System.out.println("First number/Second number is: "+(a/b));
  }
  public void pow(double a, double b){
    System.out.println("a^b = "+Math.pow(a,b));
  }
}
