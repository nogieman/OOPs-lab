//  Write a Java program to find Area and Circle of different shapes using polymorphism concept
import java.util.*;
class shape{
  public static void main(String []args){
  //shape caller = new shape();
  call();
  }
  static void call(){
   // shape c = new shape();
    p("Enter radius of circle, l & b of rectangle and side of square");
    Scanner sc = new Scanner(System.in);
    double r = sc.nextDouble();
    float l = sc.nextFloat(), b = sc.nextFloat();
    int a = sc.nextInt();
    area(r);
    area(l,b);
    area(a);
  }
  static void area(double r){
  double ar = Math.PI*r*r;
  p("Area of circle is: "+ar);
  }
  static void area(float l, float b){
  float ar = l*b;
  p("Area of rectangle is: "+ar);
  }
  static void area(int l){
  int ar = l*l;
  p("Area of square is: "+ar);
  }
  static void p(String a){
    System.out.println(a);
  }
  
  
}
