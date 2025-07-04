import java.util.*;
class ra{
  public static void main(String arggg[]){
  int r[] = new int[5];
  Random ra = new Random();
  for(int i = 0; i < 5; i++){
    r[i] = ra.nextInt();
  }
  System.out.println("The generated numbers are:\t"+Arrays.toString(r));
  int l;
  int s = l = 0;
  for(int i = 0; i < 5; i++){
    if(s > r[i]) { s = r[i];}
    if(l < r[i]) { l = r[i]; }
  }
  System.out.println("The largest value is:\t"+l+"\t& smallest is:\t"+s);
}
}
