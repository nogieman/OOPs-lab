import java.sql.*;
class jdbc{
  public static void main(String ar[]){
    try {
      Class.forName("com.mysql.cj.jdbc.Driver"); //loading jdbc
      System.out.println("Enjoyy life"); 
      Connection con = DriverManager.getConnection("jdbc:mysql://localhost/Trial","lalli","bot99");
      System.out.println("Malli enjoyy life");
      con.close();
      System.out.println("Connection closed");
    }
    catch(Exception e){
      System.out.println("\nmg\n "+e);
    }
  }
}
/*
class thre extends Thread{
  public static void main(String []aaa){

  }
}*/