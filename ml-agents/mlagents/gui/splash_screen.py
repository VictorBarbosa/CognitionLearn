"""
Script to create and show the initial splash screen with the logo
"""
import sys
from PyQt6.QtWidgets import (QApplication, QSplashScreen, QMainWindow, 
                             QVBoxLayout, QWidget, QLabel)
from PyQt6.QtGui import QPixmap, QIcon, QPainter, QFont, QColor
from PyQt6.QtCore import Qt, QTimer
import os

def create_logo_pixmap(width=200, height=200):
    """Create a simple logo programmatically"""
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(30, 30, 30))  # Dark background
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    # Draw brain-like shape
    painter.setBrush(QColor(74, 111, 165))  # Blue color
    painter.setPen(QColor(100, 181, 246))  # Light blue pen
    
    # Main brain shape
    painter.drawEllipse(50, 40, 100, 80)  # Left hemisphere
    painter.drawEllipse(100, 40, 100, 80)  # Right hemisphere
    painter.drawEllipse(80, 80, 40, 40)    # Connection
    
    # Draw details
    painter.setBrush(QColor(224, 224, 224))  # Light color for details
    painter.drawEllipse(90, 90, 10, 10)     # Neural connection
    
    painter.end()
    
    return pixmap

def show_splash_screen():
    """Show the splash screen with the logo"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    splash = QSplashScreen()
    
    # Try to load the logo image
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logo_path = os.path.join(base_dir, "cognitionlearn", "ico", "logo.png")
    
    if os.path.exists(logo_path):
        logo_pixmap = QPixmap(logo_path)
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(800, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            logo_pixmap = create_logo_pixmap(300, 300)
    else:
        logo_pixmap = create_logo_pixmap(300, 300)
        
    splash.setPixmap(logo_pixmap)
    splash.show()
    
    # Close splash after 5 seconds
    QTimer.singleShot(5000, splash.close)
    
    return app, splash

if __name__ == "__main__":
    app, splash = show_splash_screen()
    app.exec()