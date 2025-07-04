import java.util.*;
class emps{
  public static void main(String []arfs){
    System.out.println("Enter number of hours for hourly employee and wage per hour: ");
    Scanner s = new Scanner(System.in);
    int n = s.nextInt();
    float wa = s.nextFloat();
  
    System.out.println("Enter number of weeks for weekly employee and wage per week");
    int n1 = s.nextInt();
    float wa2 = s.nextFloat();
    HourlyEmployee h = new HourlyEmployee();
    WeeklyEmployee w = new WeeklyEmployee();
    float s1 = h.getAmount(n,wa);
    float s2 = w.getAmount(n1,wa2);
    System.out.println(" The salary of hourly employee is: "+s1+" \nThe salary of weekly employee is: "+s2);
  }
}
abstract class Employee{
 float getAmount(int n , float wa){
      return wa*n;
      }
}
class HourlyEmployee extends Employee{

}
class WeeklyEmployee extends Employee{
}

