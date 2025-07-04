import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
class fra extends JFrame{
  fra(){
    JButton butt = new JButton("close");
    butt.setBounds(100,250,50,50);
    butt.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e){
          System.exit(0);
        }
    });
    add(butt);
    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    
    setTitle("Menu");
    setLayout(null);
    setBounds(50,200,30,400);
    setVisible(true);
    
  }
}
class Menu{
  public static void main(String []ar){
    new fra();
  }
}
