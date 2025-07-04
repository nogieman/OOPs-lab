import java.util.*;
 
class Quad {
public static void main(String ar[]){
float a,b,c;
Scanner sc = new Scanner(System.in);
System.out.println("Enter a,b,c");
//System.out.println(ri);
a = sc.nextInt();
b = sc.nextInt();
c = sc.nextInt();
float d = b*b - 4*a*c;
if (d>0){
  double r1 = (-b+ Math.sqrt(d))/(2*a);
    double r2 = (-b- Math.sqrt(d))/(2*a);
    System.out.println("The roots are\t"+r1+" ,\t"+r2);
}
else if(d == 0){
 float r1 = (-b)/(2*a);
  System.out.println("The roots are\t"+r1+" ,\t"+r1);
  }
else {
  float rea = -b/(2*a);
    double imag = (Math.sqrt(-d))/(2*a);
    System.out.println("The roots are: "+rea+" +  i"+imag+" ,\t"+rea+" -  i"+imag);
  }
}/*
static void re( float a ,float b, float d){
    

}
static void eq(float a , float b){
  
  }
static void im(float a ,float b, float d){
    
}*/
}
