import java.util.*;
class sorti{
  public static void main(String []argss){
    Scanner x = new Scanner(System.in);
    p("Enter number of integers");
    int n = x.nextInt();
    int t;
    int a[] = new int[n];
    p("Enter the numbers");
    for(int i = 0; i<n ; i++){
      a[i] = x.nextInt();
      }
    for(int i = 0; i<n ; i++){
      for(int j = 0; j<n-1 ; j++){
        if(a[j] > a[j+1]){
          t = a[j];
          a[j] = a[j+1];
          a[j+1] = t;
          }
        }
      }
    p("The sorted array is:\t");
    System.out.println(Arrays.toString(a));
    }
  static void p(String a){
    System.out.println(a);
  } 
}
