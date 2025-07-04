class anot{
  public static void main(String []sadas){
    try{
      Class.forName("excep");
      System.out.println("Loaded an existing class, now loading a non existant class: \n");
      Class.forName("nibbs");
    }
    catch(Exception d){
      System.out.println("Catch block: "+d);
      System.out.println("Message: "+d.getMessage());
    }
    finally{
      System.out.println("Finally at finally block");
    }
  }
}
