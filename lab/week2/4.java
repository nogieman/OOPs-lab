import java.util.*;
class mat1{
 public static void main(String anj[]){
   p("Enter rows & columns of both matrices");
   Scanner s = new Scanner(System.in);
   int r = s.nextInt(); int c = s.nextInt();
   int ma[][] = new int[r][c];
   p("Enter elements of first matrix");
   for(int i =0; i<r; i++){
    for(int j = 0; j<c ; j++){
      ma[i][j] = s.nextInt();
    }
   }
   p("Enter elements of second matrix");
   int mb[][] = new int[r][c];
   for (int i =0; i<r; i++){
    for (int j =0; j<c; j++){
      mb[i][j] = s.nextInt();
    }
   }
   p("\n");
   int cm[][] = new int[r][c];
   for (int i=0; i<r; i++){
    for (int j=0; j<c; j++){
      cm[i][j] = ma[i][j] + mb[i][j];
    }
   }
   for(int i =0; i<r; i++){
    System.out.println(Arrays.toString(cm[i]));
    }
    p("\n");
   }
  
 static void p(String a){
  System.out.println(a);
 }
}
