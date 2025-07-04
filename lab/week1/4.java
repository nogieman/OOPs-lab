import java.util.*;
class pri{
  public static void main(String arg[]){
    System.out.println("Enter an integer");
    Scanner j = new Scanner(System.in);
    int n = j.nextInt();
    System.out.println("The prime factors are:\t");
    for(int i = 1; i <= n; i++){
    if(n%i == 0){
      int c = 0;
      for(int k = 1; k <= i; k++){
        if(i%k == 0) { c++;}
        }
        if(c == 2) { System.out.print(i+","); }
      }
    } 
    System.out.println("\n");
  }
}

