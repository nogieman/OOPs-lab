import org.shapes.square;
import org.shapes.triangle;
import org.shapes.circle;
import java.util.*;
class use{
  public static void main(String []args){
    square s = new square();
    triangle t = new triangle();
    circle c = new circle();
    Scanner sc = new Scanner(System.in);
    System.out.println("Enter radius, side of square, length of triangle and it's base");
    float r = sc.nextFloat();
    float a = sc.nextFloat();
    float l = sc.nextFloat();
    float b = sc.nextFloat();
    s.area(a);
    t.area(l,b);
    c.area(r);
  }
}
