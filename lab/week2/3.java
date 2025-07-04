import java.util.*;
class bin{
  public static void main(String ar[]){
  System.out.println("Enter number of elements");
  Scanner s = new Scanner(System.in);
  int n = s.nextInt();
  //System.out.println("Enter an array of n elements:  ");
  System.out.println("Enter a sorted array of n elements:  ");
  int a[] = new int[n];
  for(int i =0; i<n ; i++){
    a[i] = s.nextInt();
  }
  //Arrays.sort(a);
  //System.out.println("hhhhhhhhhh\n"+Arrays.toString(a));
  int l=0,h=n-1;
  System.out.println("Enter the number to search");
  int sea = s.nextInt();
  int m = (l+h)/2;
  while(a[m] != sea){
    if(a[m] > sea){
      h = m;
      m = (m+l)/2;
      }
    else{
      l = m;
      m = (m+h)/2;
      }
    }
    if(m == 1){
      System.out.println("The number is in the "+m+"st position in the array");
        }
    else if(m == 2){
        System.out.println("The number is in the "+m+"nd position in the array");
        }
    else if(m == 3){
        System.out.println("The number is in the "+m+"rd position in the array");
        }
    else{
        System.out.println("The number is in the "+m+"th position in the array");
        }
  }
}

