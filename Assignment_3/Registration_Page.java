import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

class fram extends  Frame implements ActionListener{
    JButton but;
     JFrame frame;
    JButton butt;
    fram(){
        // Label Components //
        Label name = new Label("Name: ");
        name.setBounds(20,30,150,30);

        Label dept = new Label("Select Department: ");
        dept.setBounds(20,100,150,40);

        Label id = new Label("ID : ");
        id.setBounds(20,60,50,30);

        JLabel gender = new JLabel();
        gender.setText("Gender: ");
        gender.setBounds(20,150,80,30);
      //    TextFields    //
        TextField nam = new TextField();
        nam.setBounds(70,30,150,30);

        TextField ag = new TextField();
        ag.setBounds(70,60,150,30);
        
        //    Checkbox    //
        Checkbox b = new Checkbox("I'm not a bot");
        b.setBounds(40,200,200,20);
        b.setFont(new Font(null,Font.PLAIN,20));
      //    Radiobutton   //
        JRadioButton ge = new JRadioButton("Male");
        JRadioButton gee = new JRadioButton("Female");
        ge.setBounds(110,150,60,30);
        gee.setBounds(180,150,100,30);

        ButtonGroup gro = new ButtonGroup();
        gro.add(ge);
        gro.add(gee);
        //    Submit button   //
        but = new JButton("Submit");
        but.setBounds(155,280,90,20);
        but.addActionListener(this);  // Adding action listener to see if it's pressed.
        
      //    Options | Dropdown list   //
        Choice cour = new Choice();
        cour.setBounds(180,100,60,30);
        cour.add("ECE");
        cour.add("EEE");
        cour.add("ME");
        //  Frame stuff   //
  
        frame = new  JFrame();
        frame.setLayout(null);
        frame.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE);
        frame.setSize(500,500);
        //  Adding components one by one    //

        butt = new JButton("Sudddddit");
        butt.setBounds(255,280,90,20);
        butt.addActionListener(this);

        frame.add(id);
        frame.add(nam);
        frame.add(name);
        frame.add(dept);
        frame.add(but);
        frame.add(b);
        frame.add(ge);
        frame.add(gee);
        frame.add(ag);
        frame.add(cour);
        frame.add(gender);
        frame.add(butt);
        
        frame.setVisible(true);
    }
    public void actionPerformed(ActionEvent e){
        if(e.getSource() == but){
            System.out.println("Stuff");
            frame.dispose();
            new sec();
        }
        else if(e.getSource() == butt){
            System.out.println("bottt");
        }
    }
}

class sec{
     JFrame frame = new  JFrame();
    Label lab = new Label("Submission successful!!");
    sec(){
        frame.setLayout(null);
        frame.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE);
        frame.setSize(400,200);

        lab.setBounds(100,50,200,60);

        frame.add(lab);
        frame.setVisible(true);
    }
}

class tryi{
    public static void main(String asd[]){
        new fram();
    }
}
