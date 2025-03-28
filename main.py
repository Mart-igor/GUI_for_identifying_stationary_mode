import sys
import os
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QButtonGroup, 
                              QFileDialog, QTableWidgetItem, QMessageBox)
from PySide6.QtCore import QPropertyAnimation, Qt, QSize, QEasingCurve
from PySide6.QtGui import (QIcon, QTextDocument, QTextCursor, 
                          QTextTableFormat, QTextCharFormat, 
                          QTextFrameFormat, QPixmap)

from ui.ui_station_interface import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.styles = self.load_styles("json/style.json")#
        self.apply_default_styles()
        self.apply_global_styles()

        self.button_group = QButtonGroup(self)
        for button_name in self.styles["QPushButtonGroup"]["Buttons"]:
            button = getattr(self.ui, button_name, None)
            if button:
                self.button_group.addButton(button)

        self.button_group.buttonClicked.connect(self.on_button_clicked)

        # сдвигам левое меню 
        self.ui.menu_btn.clicked.connect(lambda:self.slide_left_menu())
        # сдвигаем вправо форму
        self.ui.signin.clicked.connect(lambda:self.slide_right_menu())

        # Устанавливаем начальную страницу
        self.ui.stackedWidget.setCurrentIndex(0)

        self.ui.home_btn.setIcon(QIcon("icons/home.png"))
        self.ui.reports_btn.setIcon(QIcon("icons/print.png"))
        self.ui.settings_btn.setIcon(QIcon("icons/settings.png"))
        self.ui.algorithm_btn.setIcon(QIcon("icons/algorithm.png"))
        self.ui.instruction_btn.setIcon(QIcon("icons/instruction.png"))
        self.ui.account_btn.setIcon(QIcon("icons/my_account.png"))
        self.ui.help_btn.setIcon(QIcon("icons/help.png"))
        self.ui.about_btn.setIcon(QIcon("icons/about.png"))
        self.ui.menu_btn.setIcon(QIcon("icons/menu2.png"))
        self.ui.notifications.setIcon(QIcon("icons/notifications.png"))
        self.ui.notifications.setIconSize(QSize(32, 32))
        self.ui.label_2.setPixmap(QPixmap("icons/user.png"))

        self.ui.home_btn.clicked.connect(self.show_home_page)
        self.ui.reports_btn.clicked.connect(self.show_report_page)
        self.ui.settings_btn.clicked.connect(self.show_settings_page)
        self.ui.algorithm_btn.clicked.connect(self.show_algorithm_page)
        self.ui.instruction_btn.clicked.connect(self.show_instruction_page)
        self.ui.account_btn.clicked.connect(self.show_account_page)
        self.ui.help_btn.clicked.connect(self.show_help_page)
        self.ui.about_btn.clicked.connect(self.show_about_page)

    def slide_left_menu(self):
        width = self.ui.left_menu.width()
        if width == 0:
            new_width = 200
        else:
            new_width = 0

        self.animation_min = QPropertyAnimation(self.ui.left_menu, b"minimumWidth")
        self.animation_min.setDuration(250)
        self.animation_min.setStartValue(width)
        self.animation_min.setEndValue(new_width)
        self.animation_min.setEasingCurve(QEasingCurve.InOutQuart)

        self.animation_max = QPropertyAnimation(self.ui.left_menu, b"maximumWidth")
        self.animation_max.setDuration(250)
        self.animation_max.setStartValue(width)
        self.animation_max.setEndValue(new_width)
        self.animation_max.setEasingCurve(QEasingCurve.InOutQuart)

        self.animation_min.start()
        self.animation_max.start()

    def slide_right_menu(self):
        width = self.ui.right_menu.width()
        if width == 0:
            new_width = 150
        else:
            new_width = 0

        self.animation_min = QPropertyAnimation(self.ui.right_menu, b"minimumWidth")
        self.animation_min.setDuration(250)
        self.animation_min.setStartValue(width)
        self.animation_min.setEndValue(new_width)
        self.animation_min.setEasingCurve(QEasingCurve.InOutQuart)

        self.animation_max = QPropertyAnimation(self.ui.right_menu, b"maximumWidth")
        self.animation_max.setDuration(250)
        self.animation_max.setStartValue(width)
        self.animation_max.setEndValue(new_width)
        self.animation_max.setEasingCurve(QEasingCurve.InOutQuart)

        self.animation_min.start()
        self.animation_max.start()

    def load_styles(self, style_file):
        """Загружает стили из JSON файла с проверкой ошибок"""
        try:
            with open(style_file, 'r', encoding='utf-8') as f:
                styles = json.load(f)
                
            if not isinstance(styles, dict):
                raise ValueError("Style file should contain a dictionary")
                
            return styles
        except Exception as e:
            print(f"Error loading styles: {str(e)}")
            return {}  

    def apply_global_styles(self):
        style_string = ""
        
        for selector, properties in self.styles["Global"].items():
            style_string += f"{selector} {{"
            for prop, value in properties.items():
                style_string += f"{prop}: {value}; "
            style_string += "}\n"
        
        for category, items in self.styles.items():
            if category != "Global" and category != "ButtonGroups":
                for selector, properties in items.items():
                    if not isinstance(properties, dict):  
                        continue
                    style_string += f"{selector} {{"
                    for prop, value in properties.items():
                        style_string += f"{prop}: {value}; "
                    style_string += "}\n"
        
        self.setStyleSheet(style_string)

    def apply_default_styles(self):
        """Применяет специальные стили для групп виджетов"""
        if not hasattr(self, 'styles') or 'QPushButtonGroup' not in self.styles:
            return

        button_group = self.styles['QPushButtonGroup']
        not_active_style = button_group.get('Style', {}).get('NotActive', '')
        
        if not not_active_style:
            return

        for button_name in button_group.get('Buttons', []):
            button = getattr(self.ui, button_name, None)
            if button:
                original_name = button.objectName()
                try:
                    button.setStyleSheet(not_active_style)
                except Exception as e:
                    print(f"Error applying style to {original_name}: {str(e)}")

    def on_button_clicked(self, button):
        active_style = self.styles["QPushButtonGroup"]["Style"]["Active"]
        not_active_style = self.styles["QPushButtonGroup"]["Style"]["NotActive"]

        for btn in self.button_group.buttons():
            btn.setStyleSheet(not_active_style)
        button.setStyleSheet(active_style)  

    def show_home_page(self):
        self.ui.stackedWidget.setCurrentIndex(0)  

    def show_algorithm_page(self):
        self.ui.stackedWidget.setCurrentIndex(1)  

    def show_report_page(self):
        self.ui.stackedWidget.setCurrentIndex(4)  

    def show_settings_page(self):
        self.ui.stackedWidget.setCurrentIndex(8)  

    def show_instruction_page(self):
        self.ui.stackedWidget.setCurrentIndex(3)  

    def show_account_page(self):
        self.ui.stackedWidget.setCurrentIndex(6)  

    def show_help_page(self):
        self.ui.stackedWidget.setCurrentIndex(7)  

    def show_about_page(self):
        self.ui.stackedWidget.setCurrentIndex(5)  

    def go_to_page_2(self):
        algorithm = self.ui.alg_box.currentText()
        if algorithm == "Одномерные конрольные карты":
            self.ui.stackedWidget.setCurrentIndex(2) 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
