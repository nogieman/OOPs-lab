import java.util.*;
abstract class shape{
  abstract double getArea();
  abstract double getVolume();
}
class Cube extends shape{
  double s;
  Cube(double s){
    this.s = s;
  }
  double getArea(){
    return 6*s*s;
  }
  double getVolume(){
    return s*s*s;
  }
}
class areaVolume{
  public static void main(String []args){
    Scanner in = new Scanner(System.in);
    System.out.println("Enter side of cube");
    Cube c = new Cube(in.nextDouble());

    System.out.println("area of cube = "+c.getArea()+"\nVolume ="+ c.getVolume());
  }
}
