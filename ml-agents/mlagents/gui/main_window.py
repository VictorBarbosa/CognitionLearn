"""
Simple GUI for CognitionLearn - ML-Agents with GUI
This serves as the main window that replaces the CLI interface
"""
import sys
import os
from typing import Optional

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QPushButton, QTextEdit, QLabel,
                                 QTabWidget, QFileDialog, QComboBox, QGroupBox,
                                 QSplashScreen)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QIcon, QColor, QPixmap, QPainter, QPen, QBrush
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class TrainingConfigWidget(QWidget):
    """Widget for configuring training parameters"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Configuration file selection
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        
        config_h_layout = QHBoxLayout()
        config_h_layout.addWidget(QLabel("Config File:"))
        self.config_path_edit = QTextEdit()
        self.config_path_edit.setMaximumHeight(30)
        config_h_layout.addWidget(self.config_path_edit)
        
        self.browse_config_btn = QPushButton("Browse...")
        self.browse_config_btn.clicked.connect(self.browse_config)
        config_h_layout.addWidget(self.browse_config_btn)
        
        config_layout.addLayout(config_h_layout)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QHBoxLayout()
        
        env_layout.addWidget(QLabel("Environment Path:"))
        self.env_path_edit = QTextEdit()
        self.env_path_edit.setMaximumHeight(30)
        env_layout.addWidget(self.env_path_edit)
        
        self.browse_env_btn = QPushButton("Browse...")
        self.browse_env_btn.clicked.connect(self.browse_env)
        env_layout.addWidget(self.browse_env_btn)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        # Run options
        options_group = QGroupBox("Run Options")
        options_layout = QVBoxLayout()
        
        options_h_layout = QHBoxLayout()
        options_h_layout.addWidget(QLabel("Run ID:"))
        self.run_id_edit = QTextEdit()
        self.run_id_edit.setMaximumHeight(30)
        self.run_id_edit.setText("ppo")
        options_h_layout.addWidget(self.run_id_edit)
        
        options_h_layout.addWidget(QLabel("Seed:"))
        self.seed_edit = QTextEdit()
        self.seed_edit.setMaximumHeight(30)
        self.seed_edit.setText("12345")
        options_h_layout.addWidget(self.seed_edit)
        
        options_layout.addLayout(options_h_layout)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def browse_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Configuration File", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
    
    def browse_env(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Unity Environment", "", "All Files (*)"
        )
        if file_path:
            self.env_path_edit.setText(file_path)


class TrainingControlWidget(QWidget):
    """Widget for controlling training process"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Training controls
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.clicked.connect(self.pause_training)
        control_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_training)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout)
        
        # Progress and status
        status_group = QGroupBox("Training Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to start training...")
        self.status_label.setStyleSheet("font-weight: bold; color: #64B5F6; padding: 5px;")
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def start_training(self):
        self.status_label.setText("Training in progress...")
        self.status_label.setStyleSheet("font-weight: bold; color: #81C784; padding: 5px;")  # Light green
        print("Starting training...")

    def pause_training(self):
        self.status_label.setText("Training paused")
        self.status_label.setStyleSheet("font-weight: bold; color: #FFB74D; padding: 5px;")  # Light orange
        print("Pausing training...")

    def stop_training(self):
        self.status_label.setText("Training stopped")
        self.status_label.setStyleSheet("font-weight: bold; color: #E57373; padding: 5px;")  # Light red
        print("Stopping training...")


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CognitionLearn - Advanced ML-Agents Interface")
        self.setGeometry(100, 100, 800, 600)

        # Enable dark mode
        self.set_dark_theme()

        # Set application font
        font = QFont("Arial", 10)
        self.setFont(font)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title label
        title_label = QLabel("CognitionLearn - Advanced ML-Agents Interface")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #64B5F6; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Tabs
        tab_widget = QTabWidget()
        
        # Config tab
        self.config_tab = TrainingConfigWidget()
        tab_widget.addTab(self.config_tab, "Configuration")
        
        # Control tab
        self.control_tab = TrainingControlWidget()
        tab_widget.addTab(self.control_tab, "Training Control")
        
        # Console output tab
        self.console_tab = QWidget()
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            background-color: #1E1E1E;
            color: #DCDCDC;
            border: 1px solid #505050;
            border-radius: 3px;
            padding: 5px;
        """)
        console_layout.addWidget(self.console_output)
        self.console_tab.setLayout(console_layout)
        tab_widget.addTab(self.console_tab, "Console Output")
        
        main_layout.addWidget(tab_widget)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Initialize console with welcome message
        self.console_output.append("Welcome to CognitionLearn!")
        self.console_output.append("Advanced interface for ML-Agents training.")
        self.console_output.append("Use this GUI to configure and control your training sessions.")
        self.console_output.append("")

    def set_dark_theme(self):
        """Configure dark theme for the application"""
        # Set dark palette
        app = QApplication.instance()
        if app is not None:
            # Create a dark color scheme
            dark_palette = app.style().standardPalette()
            dark_palette.setColor(dark_palette.ColorRole.Window, QColor(30, 30, 30))
            dark_palette.setColor(dark_palette.ColorRole.WindowText, QColor(220, 220, 220))
            dark_palette.setColor(dark_palette.ColorRole.Base, QColor(45, 45, 45))
            dark_palette.setColor(dark_palette.ColorRole.AlternateBase, QColor(60, 60, 60))
            dark_palette.setColor(dark_palette.ColorRole.ToolTipBase, QColor(30, 30, 30))
            dark_palette.setColor(dark_palette.ColorRole.ToolTipText, QColor(220, 220, 220))
            dark_palette.setColor(dark_palette.ColorRole.Text, QColor(220, 220, 220))
            dark_palette.setColor(dark_palette.ColorRole.Button, QColor(50, 50, 50))
            dark_palette.setColor(dark_palette.ColorRole.ButtonText, QColor(220, 220, 220))
            dark_palette.setColor(dark_palette.ColorRole.BrightText, QColor(240, 240, 240))
            dark_palette.setColor(dark_palette.ColorRole.Highlight, QColor(61, 125, 189))
            dark_palette.setColor(dark_palette.ColorRole.HighlightedText, QColor(0, 0, 0))

            app.setPalette(dark_palette)

            # Set application style sheet for dark theme
            app.setStyleSheet("""
                QMainWindow, QWidget, QTabWidget, QGroupBox {
                    background-color: #1E1E1E;
                    color: #DCDCDC;
                }
                QLabel {
                    color: #DCDCDC;
                }
                QPushButton {
                    background-color: #323232;
                    color: #DCDCDC;
                    border: 1px solid #505050;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #3C3C3C;
                    border: 1px solid #606060;
                }
                QPushButton:pressed {
                    background-color: #2A2A2A;
                }
                QPushButton#start_btn {
                    background-color: #2E7D32;
                    color: white;
                    font-weight: bold;
                }
                QPushButton#start_btn:hover {
                    background-color: #388E3C;
                }
                QPushButton#pause_btn {
                    background-color: #FF8F00;
                    color: black;
                    font-weight: bold;
                }
                QPushButton#pause_btn:hover {
                    background-color: #FFA000;
                }
                QPushButton#stop_btn {
                    background-color: #C62828;
                    color: white;
                    font-weight: bold;
                }
                QPushButton#stop_btn:hover {
                    background-color: #D32F2F;
                }
                QTextEdit {
                    background-color: #2D2D2D;
                    color: #DCDCDC;
                    border: 1px solid #505050;
                    border-radius: 3px;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #505050;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #64B5F6;
                }
                QTabWidget::pane {
                    border: 1px solid #505050;
                    border-radius: 5px;
                }
                QTabBar::tab {
                    background-color: #2D2D2D;
                    color: #A0A0A0;
                    padding: 8px;
                    border: 1px solid #505050;
                    border-bottom-color: #505050;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    color: #64B5F6;
                    background-color: #1E1E1E;
                    border-bottom-color: #1E1E1E;
                }
                QStatusBar {
                    background-color: #1E1E1E;
                    color: #DCDCDC;
                }
                QMenuBar {
                    background-color: #2D2D2D;
                    color: #DCDCDC;
                }
                QMenuBar::item {
                    background: transparent;
                }
                QMenuBar::item:selected {
                    background: #3C3C3C;
                }
                QMenuBar::item:pressed {
                    background: #2A2A2A;
                }
            """)


def create_logo_pixmap(width=300, height=300):
    """Create a simple logo programmatically"""
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(30, 30, 30))  # Dark background

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Set pen for drawing
    pen = QPen(QColor(100, 181, 246))  # Light blue
    pen.setWidth(2)
    painter.setPen(pen)

    # Draw brain-like shape
    brush = QBrush(QColor(74, 111, 165))  # Blue color
    painter.setBrush(brush)

    # Main brain shape (simplified)
    painter.drawEllipse(75, 60, 150, 120)  # Main oval
    painter.drawEllipse(100, 70, 50, 60)   # Left bump
    painter.drawEllipse(150, 70, 50, 60)   # Right bump

    # Neural connections
    painter.setBrush(QBrush(QColor(224, 224, 224)))  # Light color for details
    painter.drawEllipse(120, 100, 15, 15)   # Neural connection 1
    painter.drawEllipse(145, 100, 15, 15)   # Neural connection 2
    painter.drawEllipse(132, 120, 10, 10)   # Neural connection 3

    painter.end()

    return pixmap

# Global references to keep objects alive
main_window_instance = None
splash_instance = None

def launch_gui():
    """Launch the main GUI window"""
    if not GUI_AVAILABLE:
        print("PyQt6 is not available. Please install it using: pip install PyQt6")
        return

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

    # Try to load the logo image
    # Path: ml-agents/cognitionlearn/ico/logo.png
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logo_path = os.path.join(base_dir, "cognitionlearn", "ico", "logo.png")
    
    if os.path.exists(logo_path):
        logo_pixmap = QPixmap(logo_path)
        if not logo_pixmap.isNull():
            # Scale image to be prominent
            logo_pixmap = logo_pixmap.scaled(800, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            logo_pixmap = create_logo_pixmap(400, 400)
    else:
        logo_pixmap = create_logo_pixmap(400, 400)

    # Show splash screen
    global splash_instance
    splash_instance = QSplashScreen(logo_pixmap)
    # Ensure it stays on top and is visible
    splash_instance.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
    splash_instance.show()

    # Process events to ensure the splash screen is drawn
    app.processEvents()

    # Create main window after a delay (5 seconds for better visibility)
    QTimer.singleShot(5000, lambda: show_main_window(app, splash_instance))

    # If running as standalone app
    if not QApplication.instance().property('is_subapp'):
        sys.exit(app.exec())

# Global reference to keep main window alive
main_window_instance = None

def show_main_window(app, splash):
    """Show the main window and close splash screen"""
    global main_window_instance
    
    # Create and show main window
    main_window_instance = MainWindow()
    main_window_instance.show()
    
    # Close splash screen when main window is ready
    splash.finish(main_window_instance)


if __name__ == "__main__":
    launch_gui()