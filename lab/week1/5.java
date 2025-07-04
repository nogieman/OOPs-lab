import java.util.*;
class pali{
  public static void main(String []aaa){
      p("Enter a number");
      Scanner c = new Scanner(System.in);
      int n = c.nextInt();
      int a = n,r,re=0;
      while(a != 0){
        r = a%10;
        re = re*10 + r;
        a = a/10;
      }
      if(re == n){
        p("The number is a palindrome");
      }
      else p("The number isn't a palindrome");
  }
  static void p(String a){
  System.out.println(a);
  }
}

