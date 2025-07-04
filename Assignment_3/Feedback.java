import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

class feedback extends JFrame implements ActionListener{
	JTextField nam;
	JTextField  mai;
	JTextField  feedb;
	JButton sub;
	feedback(){

		 setSize(700,800);
		 setLayout(null);
		 setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		//	Declaring components	//
		//	Labels	//
		Label na = new Label("Name: ");
		Label mail = new Label("E-mail: ");
		Label rat = new Label("Rating out of 5 stars :");
		Label feed = new Label("Feedback(optional) :");
		//	JTextFields	//
		nam = new JTextField();
		mai = new JTextField();
		feedb = new JTextField();
		//	RadioButton	//
		JRadioButton one = new JRadioButton("1 *");
		JRadioButton two = new JRadioButton("2 *");
		JRadioButton three = new JRadioButton("3 *");
		JRadioButton four = new JRadioButton("4 *");
		JRadioButton five = new JRadioButton("5 *");
		//	Grouping the radiobuttons	//
		ButtonGroup gr = new ButtonGroup();
		gr.add(one);
		gr.add(two);
		gr.add(three);
		gr.add(four);
		gr.add(five);

		//	Button	//
		sub = new JButton("Submit");
		sub.setFocusable(true);
		sub.addActionListener(this);
		sub.setBackground(Color.blue);

		//	setting bounds 	//
		na.setBounds(80,105,70,30);
		nam.setBounds(160,105,120,30);
		mail.setBounds(80,200,80,30);
		mai.setBounds(160,200,110,30);
		rat.setBounds(80,300,110,30);
		one.setBounds(250,300,50,30);
		two.setBounds(320,300,50,30);
		three.setBounds(390,300,50,30);
		four.setBounds(460,300,50,30);
		five.setBounds(530,300,50,30);
		feed.setBounds(100,550,140,30);
		feedb.setBounds(240,500,340,130);
		sub.setBounds(250,650,200,70);
		//	Adding them to the frame	//
		 add(na);
		 add(nam);
		 add(mai);
		 add(mail);
		 add(rat);
		 add(one);
		 add(two);
		 add(three);
		 add(four);
		 add(five);
		 add(feed);
		 add(feedb);
		 add(sub);
		 getContentPane().setBackground(Color.cyan);
		 setVisible(true);

	}
	public static void main(String []asd){
		new feedback();
	}

	public void actionPerformed(ActionEvent e){
		if(e.getSource() == sub){
			String n = nam.getText();
			String em = mai.getText();
			if(n.length() == 0){
				System.out.println("name");
				JOptionPane.showMessageDialog(this,"Name Can't be empty");
			}
			else if(!em.endsWith("gmail.com")){
				JOptionPane.showMessageDialog(this,"enter a valid e mail ending with gmail");
			}
			else {
				JOptionPane.showMessageDialog(this,"Thankyou for your feedback!");
			}
		}

	}
}

