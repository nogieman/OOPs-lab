import java.util.*;
class NameSort{
  public static void main(String arg[]){
  Scanner sc = new Scanner(System.in);
  System.out.println("Enter number of names & names to sort");
  String t;
  int n = sc.nextInt();
  String names[] = new String[n+1];
  for(int i = 0; i<=n ; i++){
    names[i] = sc.nextLine();
    
    }
  Arrays.sort(names);
 /* for(int i = 0; i<n; i++){
    for(int j =0; j<n; j++){
      if(names[j].compareTo(names[j+1])>0){
        t= names[j];
        names[j]=names[j+1];
        names[j+1]=t;
      }
    }
  }*/
  /*for(int i =0; i<=n; i++){
    System.out.print(names[i]+",");
    }*/
    System.out.print(Arrays.toString(names));

  }
}

