import java.util.*;
class calc{
  public static void main(String arg[]){
    p("Enter 1 for sum, 2 for diff, 3 for prod, 4 for division, 5 for modulus, 6 for a^b, 7 for factorial \n");
    Scanner sc = new Scanner(System.in);
    int n = sc.nextInt();
    p("Enter a & b\n");
    int a,b;
    a = sc.nextInt();
    b = sc.nextInt();
    switch(n){
      case 1: p("the sum is:\t"); 
                i(a+b);
               break;
      case 2: p("The difference is \t");
                i(a-b);
                break;
      case 3: p("The product is\t");
              i(a*b);
              break;
      case 4: p(" a/b is:\t");
              i(a/b);
              break;
      case 5: p("a%b is:\t");
              i(a%b);
              break;
      case 6: p("a raised to b is:\t");
              dou(Math.pow(a,b));
              break;
      case 7: p("Factorial of a is:\t"); int fac=1;
              for(int r = 1; r<=a;r++){ 
              fac = r*fac;
              }
              i(fac);
              p("Factorial of b is:\t"); fac=1;
              for(int r = 1; r<=b;r++){ 
              fac = r*fac;
              }
              i(fac);
              break;
      default: p("Enter a valid num");
  }
  }
  static void p(String a){
    System.out.print(a);
  }
  static void i(int a){
  System.out.println(a);
  }
  static void dou(double a){
  System.out.println(a);
  }
}
