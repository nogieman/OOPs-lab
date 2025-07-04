import java.util.*;
class shape{
  public static void main(String []args){
  shape caller = new shape();
  caller.call();
  }
  void call(){
    shape c = new shape();
    c.p("Enter radius of circle, l & b of rectangle and side of square");
    Scanner sc = new Scanner(System.in);
    double r = sc.nextDouble();
    float l = sc.nextFloat(), b = sc.nextFloat();
    int a = sc.nextInt();
    c.area(r);
    c.area(l,b);
    c.area(a);
  }
  void area(double r){
  double ar = Math.PI*r*r;
  p("Area of circle is: "+ar);
  }
  void area(float l, float b){
  float ar = l*b;
  p("Area of rectangle is: "+ar);
  }
  void area(int l){
  int ar = l*l;
  p("Area of square is: "+ar);
  }
  void p(String a){
    System.out.println(a);
  }
  
  
}
