class pa{
  void me(int a){
  a = 3;
    System.out.println("The number is: "+a);
  }
  void me(int a, int b){
    a = 3; b = a*2;
    System.out.println("The numbers are: "+a+"\t"+b);
  }
  void mea(){
    System.out.println("hi");
  }
}
class ch extends pa{
  public static void main(String []args){
    pa n = new pa();
    int a=0,b=0;
    n.me(a);
    n.me(a,b);
    n.mea();
    ch v = new ch();
    v.mea();
  }
  void mea(){
  System.out.println("modified stuff");  
}
}
