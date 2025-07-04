import javax.swing.*;
import java.awt.*;

class BatmanLogo extends JPanel {

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        drawBatmanLogo(g);
    }

    private void drawBatmanLogo(Graphics g) {
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, getWidth(), getHeight());

        g.setColor(Color.YELLOW);
        int centerX = getWidth() / 2;
        int centerY = getHeight() / 2;

        // Draw the oval background
        g.fillOval(centerX - 70, centerY - 140, 140, 280);

        // Draw the bat symbol
        g.setColor(Color.BLACK);
        int[] xPoints = {centerX, centerX - 70, centerX - 25, centerX, centerX + 25, centerX + 70};
        int[] yPoints = {centerY - 140, centerY - 70, centerY - 70, centerY - 40, centerY - 70, centerY - 70};
        g.fillPolygon(xPoints, yPoints, xPoints.length);

        int[] xWing = {centerX, centerX - 40, centerX - 10, centerX, centerX + 10, centerX + 40};
        int[] yWing = {centerY - 100, centerY - 40, centerY - 70, centerY - 40, centerY - 70, centerY - 40};
        g.fillPolygon(xWing, yWing, xWing.length);

        // Draw the eyes
        g.fillOval(centerX - 30, centerY - 110, 20, 40);
        g.fillOval(centerX + 10, centerY - 110, 20, 40);
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Batman Logo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300, 400);
        frame.setLocationRelativeTo(null);

        BatmanLogo batmanLogo = new BatmanLogo();
        frame.add(batmanLogo);

        frame.setVisible(true);
    }
}

