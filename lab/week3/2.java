import java.util.*;
class prod{
    public static void main(String []asad){
        double pr1 = 99.90, pr2 = 20.20, pr3 = 6.87, pr4 = 45.50, pr5 = 40.49;
        Scanner s = new Scanner(System.in);
        p("Enter number of products you want");
        int n = s.nextInt();
        double cost = 0, tot = 0;
        for(int i = 0; i<n; i++){
        p("Enter the product ID");
        int prid = s.nextInt();
        if(prid < 1 || prid>5){
        p("Enter a valid ID");
        break;}
        p("Enter number of Items");
        int prno = s.nextInt();
        switch(prid){
            case 1: 
                cost = prno*pr1;
                break;
            case 2:
                cost = prno*pr2;
                break;
            case 3:
                 cost = prno*pr3;
                break;
            case 4:
                 cost = prno*pr4;
                break;
            case 5: 
                 cost = prno*pr5;
                break;
            default: 
                p("Enter a valid ID");
        }
        tot = tot + cost;
        System.out.println("The total cost upto now is:  "+tot);
        }
        System.out.println("*******************\nThe total cost is:  "+tot+"\n*******************");
    }
    static void p(String a){
        System.out.println(a);
    }
}
