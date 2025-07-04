import java.util.*;
class line{
  public static void main(String []af){
    p("Enter number of elements\n");
    Scanner v = new Scanner(System.in);
    int n = v.nextInt();
    int a[] = new int[n];
    p("Enter elements\n");
    for(int i = 0; i<n ; i++){  //Storing the numbers
      a[i] = v.nextInt();
    }
    p("Enter element to search\n");
    int s = v.nextInt();
    for(int j =0; j < n; j++){    //searching the number
    if(a[j] == s){
      p("The required number is in the ");
      if(j==1){ System.out.print(j+"st index of the array.");}
      else if(j==2){ System.out.print(j+"nd index of the array.");}
      else if(j==3){ System.out.print(j+"rd index of the array");}
      else{ System.out.print(j+"th index of the array");}
      }
    }
  }
  
  static void p(String a){
    System.out.print(a);
  } 
}
