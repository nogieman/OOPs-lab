//import java.util.IOException;
class excep{
  public static void main(String []args) throws Exception{ 
    try{
      System.out.println("Hi");
      throw new Exception("Here's an exception");
     
    }
    catch(Exception w){
      System.out.println("Catch block: "+w);
      System.out.println(w.getMessage());
    }
    finally{
      System.out.println("This is the finally block");
    }
  }
 
}
