import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

class login extends JFrame implements ActionListener{
  JTextField IDf;
  JTextField psswdf;
  JTextField capf;
  public login(){
  //    Creating a frame & setting properties like size, layout n stuff    //
    setTitle("Login page");
    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    setSize(500,400);
    //setLocationRelativeTo(null);
    setLayout(new GridLayout(5,2));
    //setLayout(new FlowLayout(FlowLayout.LEFT,30,30));
  //    Create labels   //
    JLabel ID = new JLabel("ID kottu");
    JLabel psswd = new JLabel("Password kottu (nenu_nibb)");
    JLabel cap = new JLabel("Captcha : ImAbOt");
    JLabel l = new JLabel(" First login page...");
    l.setFont(new Font(" ",Font.ITALIC,28));
    
  //    Create text fields   //
    IDf = new JTextField();
    psswdf = new JTextField();
    capf = new JTextField();
    
  //    create button   //
    JButton butt = new JButton("Login");
    butt.addActionListener(this);
  //    Add components that we created    //
    add(l);
    add(new JLabel());
    add(ID);
    add(IDf);
    add(psswd);
    add(psswdf);
    add(cap);
    add(capf);
    //add(new JLabel());
    add(butt);
  //    Making it visible   //
    setVisible(true);
  }
  public void actionPerformed(ActionEvent w){
  //    if button is clicked    //
    if(w.getActionCommand().equals("Login")){
      if(psswdf.getText().equals("nenu_nibb") && capf.getText().equals("ImAbOt")){
        String un = IDf.getText();
        JOptionPane.showMessageDialog(this,"Successful ga login aiyyaru.\nId :   "+un);
      }
      else if(!psswdf.getText().equals("nenu_nibb") && capf.getText().equals("ImAbOt")){
        JOptionPane.showMessageDialog(this,"Password tappu kottaru babu");
      }
      else if(!capf.getText().equals("ImAbOt") && psswdf.getText().equals("nenu_nibb")){
        JOptionPane.showMessageDialog(this,"captcha tappu kottaru babu");
      }
      else{
        JOptionPane.showMessageDialog(this,"Password inka captcha rendu tappu kottaru babu");
      }
    }
  }
  public static void main(String []sad){
    SwingUtilities.invokeLater(() -> new login());
  }
}
