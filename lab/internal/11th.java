//11) Derive sub-classes of ContractEmployee namely HourlyEmployee & WeeklyEmployee with information number of hours & wages per hour, number of weeks & wages per week respectively & method calculateWages() to calculate their monthly salary. Also override getDesig () method depending on the type of contract employee.

class ContractEmployee{
  int hrno = 200, waph = 350, weno = 4, wapw = 20000;
  int calculateWages(int a, int b){
    return a*b;
  }
  void getDesig(){
    System.out.println("Employee designation & salary");
  }
}
class HourlyEmployee extends ContractEmployee{
  int c = calculateWages(hrno, waph);
  void getDesig(){
    System.out.println("The employee is hourly. The monthly salary is: "+c);
  }
}
class WeeklyEmployee extends ContractEmployee{
  int c = calculateWages(weno, wapw);
  void getDesig(){
    System.out.println("The employee is weekly. The monthly salary is: "+c);
  }
}
class emplo{
  public static void main(String []ads){
    HourlyEmployee a = new HourlyEmployee();
    WeeklyEmployee b = new WeeklyEmployee();
    a.getDesig();
    b.getDesig();
  }
}
