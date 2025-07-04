class anoth{
  public static void main(String ar[]){
  String s = "123j"; String l = "123456";
    try{
      int h = Integer.parseInt(l);
      System.out.println("Now formatting string "+s+" into an integer");
      int g = Integer.parseInt(s);
    }
    catch(Exception u){
      System.out.println("Catch block: "+u);
      System.out.println(u.getMessage());
    }
    finally{
      System.out.println("Finally at finally block");
    }
  }
}
