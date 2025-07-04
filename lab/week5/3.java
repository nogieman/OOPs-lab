class calling{
  int a;
   void modify(int b){    
   b = b+1;
    System.out.println("The passed variable is now of value: " + b);
    calling bn = new calling();   //creating object to pass to method obj
    obj(bn);
  }
  void man(){

    int c =3;
    System.out.println("The initial value of vaiable: "+c);
    modify(c);    //passing value
    
  }
   void obj(calling bn){   //receiving object and accessing value through it 
    bn.a = 5;
    System.out.println("The variable called through object holds the value: "+bn.a);
    System.out.println("The new value is: "+(bn.a+1));
  }
  public static void main(String []afsa){
    calling call = new calling();
    call.man();
  }
  
}
