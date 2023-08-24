import javax.swing.*;
class ButtonA extends JFrame{
  public static void main(String []asd){
    JFrame f = new JFrame();
    JButton b = new JButton("Click");
    b.setBounds(50,100,80,30);
    f.add(b);
    f.setSize(400,400);
    f.setLayout(null);
    f.setVisible(true);
    f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    
  }
}
