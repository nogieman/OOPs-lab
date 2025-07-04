import java.util.*;
class vecc{
    public static void main(String []arer){
        Vector v = new Vector();
        Scanner s = new Scanner(System.in);
        p("Enter 5 unique numbers between 10 and 100: ");
        for(int i =0; i<=5; i++){
            int n = s.nextInt();
            if(n >= 10 && n<= 100){
                if(!v.contains(n)){
                    v.add(n);
                    System.out.println("The unique nos. are:  "+v);
                }
                else System.out.println("already exists");
            }
            else{
                p("number isn't between 10 & 100 ");
                i--;
            }
        }
    }
    static void p(String a){
        System.out.println(a);
    }
}
