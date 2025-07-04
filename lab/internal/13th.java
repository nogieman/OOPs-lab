import java.util.*;
abstract class shape{
  void getArea(){}
  void getVolume(){}
  }
class square extends shape{
  double getArea(double a){
    double area = a*a;
    return area;
  }
  double getVolume(double a){
    return 0;
  }
}
class cube extends shape{
  double getArea(double a){
    return a*a*6;
  }
  double getVolume(double a){
    return a*a*a;
  }
}
class circle extends shape{
  double getArea(double a){
    return  3.1415*a*a;
    }
  double getVolume(double a){
    return  0; 
    }
}
class sphere extends shape{
  double getVolume(double a){
    return (4.0*3.1415*a*a*a)/3;
  }
  double getArea(double a){
    return 4*3.1415*a*a;
  }
}
class shapes{
  public static void main(String []arfsfs){
    Scanner s = new Scanner(System.in);
    square sq = new square();
    cube cu = new cube();
    circle c= new circle();
    sphere sp = new sphere();
    System.out.println("Enter radius of circle/sphere: ");
    float r = s.nextFloat();
    System.out.println("The area of circle is: "+c.getArea(r)+"\nThe area of sphere is:  "+sp.getArea(r)+"\nThe volume of sphere is: "+sp.getVolume(r));
    System.out.println("Enter the length(side) of side of square/cube");
    float a = s.nextFloat();
    System.out.println("The area of square is: "+sq.getArea(a)+"\nThe area of cube is:  "+cu.getArea(a)+"\nThe volume of cube is: "+cu.getVolume(a));
  }
}
