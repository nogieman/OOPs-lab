class example{
  static int x=5;
  int y;
  example(int x, int y){
    this.x = x;
    this.y = y;
    System.out.println("Constructor: "+x+" "+y);    
  }
  static void stat(){
    System.out.println("The static variable is: "+x+"\n");
  }
  void inst(){
    System.out.println("\nThe static variable is: "+x+"\nInstance variable is: "+y);
  }
}
class examp extends example{
  examp(){
    super(11,12);
    System.out.println("Setting above variables using super keyword by creating an object of subclass: \n");
  }
  public static void main(String []arsd){
    examp a = new examp();
    System.out.println("Now creating object of superclass and invoking constructor");
    example e = new example(13,14);
    System.out.println("Accessing static variable through classname.varName");
    example.x = 10;
    System.out.println("Calling static method using classname.MethodName");
    example.stat();
    System.out.println("Accessing instance variable using object");
    e.y = 20;
    System.out.println("Calling instance method using superclass object");
    e.inst();
  }
}
