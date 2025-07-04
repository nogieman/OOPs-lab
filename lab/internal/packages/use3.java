import java.util.*;
import dept.cse;
import dept.ece;
import dept.mech;
import dept.civil;
class branch{
  public static void main(String[] args){
    Scanner s=new Scanner(System.in);
    System.out.println("Enter year: ");
    int n=s.nextInt();
    cse c=new cse();
    ece e=new ece();
    mech m=new mech();
    civil ci=new civil();
    c.cs(n);
    e.cs(n);
    m.cs(n);
    ci.cs(n);
}
}
