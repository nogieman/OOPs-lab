import java.util.*;
class mult{
    public static void main(String argg[]){
        System.out.println("Enter number of rows and columns of first matrix: ");
        Scanner s = new Scanner(System.in);
        int r1 = s.nextInt();
        int c1 = s.nextInt();
        System.out.println("Enter the number of rows and columns of second matrix");
        int r2 = s.nextInt();
        int c2 = s.nextInt();
        if(c1 == r2){
            System.out.println("Enter elements of First matrix: ");
           int m1[][] = new int[r1][c1];
            int m2[][] = new int[r2][c2];
            for(int i = 0; i<r1; i++){
                for(int j = 0; j<c1; j++){
                    m1[i][j] = s.nextInt();
            }
        }
        
        System.out.println("Enter the elements of second matrix:  ");
        for(int i = 0; i<r2; i++){
            for(int j = 0; j<c2; j++){
                m2[i][j] = s.nextInt();
                }
           }
        
        int m3[][] = new int[r1][c2];
        for(int i = 0; i< r1; i++){
            
            for(int j = 0; j< c2; j++){
                int s1 = 0;
                for(int k =0; k< c2; k++){
                s1 = s1 + m1[i][k]*m2[k][j];
                }
            m3[i][j] = s1;
            }
        
        }
        System.out.println("The product matrix is:  ");
        for(int i = 0; i<r1;i++){
        System.out.println(Arrays.toString(m3[i]));
        
        }
      }
      else System.out.println("The operation isn't valid");
    }
}
