import java.util.*;
interface Pay{
 void getAmount(double r);
  }
class payment implements Pay{
  public static void main(String args[]){
    p("Enter invoice number: ");
    Scanner sc = new Scanner(System.in);
    String inv = sc.nextLine();
    p("Enter employee Pay: ");
    double e = sc.nextDouble();
    p("The pay for invoice ID "+inv+" is: ");
    payment po = new payment();
    po.getAmount(e);
  }
 public void getAmount(double r){
    p(" "+r);
  }
  static void p(String a){
    System.out.println(a);
  }
}









/*
class payment implements Pay{
  static String inv;
  static double e;
  payment(String i, double d){
    this.inv = i;
    this.e = d;
  }
  public static void main(String []args){
    p("Enter invoice number: ");
    Scanner sc = new Scanner(System.in);
    inv = sc.nextLine();
    p("Enter employee payment");
    e = sc.nextDouble();
    payment u = new payment(inv,e);
    double a = u.getAmount(e); 
    p("Invoice number: "+inv+"\nPay: "+e);
  }
  public double getAmount(double r){
    return r;
  }
  static void p(String a){
    System.out.println(a);
  }
}
*/
