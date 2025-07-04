import java.util.*;
class empl{
  public static void main(String []aad){
    //  Stored Data
    String det[][] = {
      {"010","Indira","Female","Systems Engineer","90,000","Chandigarh"},
      {"020","Narendar","Male","Team Lead","50,000","bhavnagar"},
      {"030","Rahul","Male","Testing Engineer","30,000","Delhi"},
      {"040","M Banarjee","Female","Testing  designer","60,000","Kolkata"}
    };
    System.out.println("Enter employee ID, the available ids are:");
    //Printing Available IDs
    for(int i =0; i<det.length;i++){
      System.out.println(det[i][0]);
    }
    int a = det.length;
    Scanner sc = new Scanner(System.in);
    String sea = sc.nextLine(); // Taking which ID to search
    Boolean t = false;

    for(int i = 0; i< a; i++){
      if(sea.equals(det[i][0])){  //  checking if the searching id is equal to the first element of the data, which is the ID.
        System.out.println("\nID:\t"+det[i][0]);
        System.out.println("Name:\t"+det[i][1]);
        System.out.println("Gender:\t"+det[i][2]);
        System.out.println("Designation:\t"+det[i][3]);
        System.out.println("Salary:\t"+det[i][4]);
        System.out.println("Address:\t"+det[i][5]);
        t = true; // Changing the boolean value into true
      }
    }
    if(t == false){ //printing if the ID isn't found
      System.out.println("The details are unavailable");
    }
  }
}

