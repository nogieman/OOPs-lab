import java.util.*;
class dice{
    public static void main(String []asd){
        Random r = new Random();
        int n,p,c=0;
        for(int i =1; i<=10;i++){
            System.out.println("Rolling");
            for(int k=0;k<6;k++){
                System.out.print(".");
                try{Thread.sleep(500);} //Original delay 1000
                catch(InterruptedException e){e.printStackTrace();}
            }
            n  = r.nextInt(6)+1;
            p = r.nextInt(6)+1;
            System.out.println("\n");
            if(n == p){
                System.out.print("Successfull roll!");
                for(int g=1;g<=4;g++){
                System.out.print("!-)");
                try{Thread.sleep(500);}    //original delay 1000
                catch (InterruptedException e){ e.printStackTrace();
                }
                }
                System.out.println("The obtained numbers are: "+n+" & "+p);
                c++;
            }
            else {
                System.out.print("Unsuccessful roll  :");
                for(int h=1;h<=4;h++){              
                    System.out.print("'");
                    try{Thread.sleep(500);}    //Original delay 1000
                    catch(InterruptedException e){e.printStackTrace();}
                    }
                System.out.println("'(");
                System.out.println("The obtained numbers are: "+n+" & "+p);
                }
            }
        System.out.println("\n **************************************\n\tYour success count is:\t"+c+"\n**************************************");
        }  
    }
