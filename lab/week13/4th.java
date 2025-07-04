import java.sql.*;
import java.io.*;

class RetrieveImageFromDB {
    public static void main(String[] args) {
        try (Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/Trial", "lalli", "bot99")) {
            String sql = "SELECT image_data FROM images WHERE name = ?";
            try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
                pstmt.setString(1, "batmans.jpg");
                try (ResultSet rs = pstmt.executeQuery()) {
                    if (rs.next()) {
                        Blob blob = rs.getBlob("image_data");
                        InputStream inputStream = blob.getBinaryStream();
                        File imageFile = new File("/home/lalli/Pictures/retreived.png");
                        FileOutputStream fos = new FileOutputStream(imageFile);
                        byte[] buffer = new byte[1024];
                        int bytesRead;
                        while ((bytesRead = inputStream.read(buffer)) != -1) {
                            fos.write(buffer, 0, bytesRead);
                        }
                        fos.close();
                        inputStream.close();

                        System.out.println("Image retrieved and saved successfully.");
                    } else {
                        System.out.println("Image not found in the database.");
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

