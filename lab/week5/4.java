import java.util.*;
class Account{
  double balance;
  String Acc_Name;
  int Acct_no;
  String Acct_Address;
  Account(){
    this.balance = 1000;
    this.Acc_Name = "Nibb";
    this.Acct_no = 82323923;
    this.Acct_Address = "Basar tation";
  }
   void credit(double am){
    this.balance = balance+am;
    getBalance();
  }
   void debit(double de){
    if(de <= balance){
      this.balance = balance-de;
      getBalance();
    }
    else{
      System.out.println("************************************************\nThe debit amount exceeds the account balance\n************************************************");
      getBalance();}
  }
   void getBalance(){
    System.out.println("************************************************\nAccount details:\nAccount number: "+Acct_no+"\nAccount name: "+Acc_Name+"\nAddress: "+Acct_Address);
    System.out.println("The account balance is: "+this.balance+"\n************************************************");
  }
  public static void main(String []args){
    double de,am;
    Account caller = new Account();
    caller.getBalance();
    System.out.println("\nContinue transaction? yes or no");
    Scanner s = new Scanner(System.in);
    String st = s.nextLine();
    if(st.equalsIgnoreCase("yes")){
      System.out.println("Do you want to debit? yes or no");
      String ce = s.nextLine();
      if(ce.equalsIgnoreCase("yes")){
        System.out.println("Enter amount to debit: \n*****");
        de = s.nextDouble();
        System.out.println("\n*****\n");
        caller.debit(de);
      }
      else {
        System.out.println("Enter amount to add: \n****");
        am = s.nextDouble();
        System.out.println("\n*****\n");
        caller.credit(am); 
      }
     } 
    }  
  }

