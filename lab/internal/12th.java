//12)Write an application to create a super class Employee with information first Name & last name and methods getFirstName(), getLastName() derive the Subclasses ContractEmployee and RegularEmployee with the information About department, designation & method displayFullName() ,getDepartment(), getDesig() to print the salary and to set department name & designation of the corresponding sub-class objects respectively
class Employee{
  String firstname, lastname;	
  Employee(String firstname,String lastname){
    this.firstname = firstname;
    this.lastname = lastname;
  }
}	////
class ContractEmployee extends Employee{
  String department, designation;
  ContractEmployee(String firstname,String lastname){
  super(firstname,lastname);
  }
  void displayFullName(){
    System.out.println("Name: "+firstname+" "+lastname);
  }
  void getDepartment(String department){
    this.department = department;
    System.out.println("Department: "+department);
  }
  void getDesig(String designation,int a){
    this.designation = designation;
    System.out.println("Designation: "+designation+"Salary: "+a);
    }
}////
class RegularEmployee extends Employee{
  RegularEmployee(String firstname,String lastname){
  super(firstname,lastname);
  }
  String department, designation;
  void displayFullName(){
    System.out.println("Name: "+firstname+" "+lastname);
  }
  void getDepartment(String department){
    this.department = department;
    System.out.println("Department: "+department);
  }
  void getDesig(String designation,int a){
    this.designation = designation;
    System.out.println("Designation: "+designation+"Salary: "+a);
  }
}/////
class employees{
  public static void main(String []args){
    RegularEmployee r = new RegularEmployee("lalith", "Nibba");
    ContractEmployee c = new ContractEmployee("PM","G");
    r.displayFullName();
    r.getDepartment("Testing Enginner");
    r.getDesig("Regular",50000);
    c.displayFullName();
    c.getDepartment("CEO of nipples");
    c.getDesig("Contract prostitute(male)",100000);
  }
}/////


