import java.applet.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

class Smile extends JFrame{
	Smile(){
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public void paint(Graphics g){
		g.setColor(Color.yellow);
		g.fillOval(45,95,80,80);
		g.setColor(Color.red);
		g.fillOval(110,95,5,5);
		g.fillOval(145,95,5,5);
		g.setColor(Color.green);
		g.fillArc(113,115,35,20,0,-180);


	}
	public static void main(String []ss){
		Smile s = new Smile();
		s.setSize(400,400);
		s.setTitle("dd");
		s.setVisible(true);

	}

}