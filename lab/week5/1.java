import java.util.*;
class wrap{
  public static void main(String []args){
     int a=3; boolean b = true; double c=6.9; 
     p("The primitive datatypes are: a(int): "+a+" b(boolean): "+b+" c(double): "+c);

     Integer in = Integer.valueOf(a);
     Boolean bol =  Boolean.valueOf(b);
     Double dou = Double.valueOf(c);
     
  }
  static void p(String a){
    System.out.println(a);
  }
}
