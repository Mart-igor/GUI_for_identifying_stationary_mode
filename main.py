import sys
import os
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QButtonGroup, 
                              QFileDialog, QTableWidgetItem, QMessageBox)
from PySide6.QtCore import QPropertyAnimation, Qt, QSize, QEasingCurve
from PySide6.QtGui import (QIcon, QTextDocument, QTextCursor, 
                          QTextTableFormat, QTextCharFormat, 
                          QTextFrameFormat, QPixmap)

import pandas as pd
import numpy as np
import matplotlib.dates as mdates

from ui.ui_station_interface import Ui_MainWindow
from mplwidget import MplWidgetOptimize, MplWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.df = None
        self.data_graph = None
        self.y_true = None
        self.data_v3 = None
        self.table_result = None
        self.selection_start = None
        self.selection_end = None
        self.selected_regions = []
        self.all_ratios = None
        
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

        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.progressBar.setVisible(False)

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

        self.ui.load_btn.clicked.connect(self.load_csv)
        self.ui.data_graph_btn.clicked.connect(self.plot_graph)
        self.ui.data_graph_btn.setEnabled(False)
        self.ui.next_page_1.clicked.connect(self.go_to_page_2)

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

    def display_table(self, data: pd.DataFrame, table_name) -> None:
        table_name.setRowCount(data.shape[0])
        table_name.setColumnCount(data.shape[1])
        table_name.setHorizontalHeaderLabels(data.columns)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                table_name.setItem(i, j, QTableWidgetItem(str(data.iat[i, j])))

    def algorith(self, data: pd.DataFrame, range_max_min: int, start_ratio: list[float] = [0.2, 0.1, 0.1]) -> pd.DataFrame:

        l1 = start_ratio[0]
        l2 = start_ratio[1]
        l3 = start_ratio[2]

        x_st = data.iloc[:50, 0].mean()
        xf_st = data.iloc[:50, 0].mean()
        vf_st = data.iloc[:50, 0].var()
        df_st = 2 * data.iloc[:50, 0].var()

        r_list = []
        vf_list = []
        df_list = []
        xf_list = []

        for i in range(51, range_max_min):
            xf = l1 * data.iloc[i, 0] + (1 - l1) * xf_st
            vf = l2 * (data.iloc[i, 0] - xf_st) ** 2 + (1 - l2) * vf_st
            df = l3 * (data.iloc[i, 0] - x_st) ** 2 + (1 - l3) * df_st
            r = round(((2 - l1) * vf) / df, 4)

            vf_list.append(vf)
            df_list.append(df)
            r_list.append(r)
            xf_list.append(xf)

            x_st = data.iloc[i, 0]
            xf_st = xf
            vf_st = vf
            df_st = df

        rr = pd.DataFrame(data=r_list, index=data.iloc[51:range_max_min].index, columns=['R'])
        rr['stationary'] = np.where(rr['R'] > 2.3715370273232828, 0, 1)
        return rr

    def plot(self, widget_name, data_x: pd.Series, data_y: pd.Series, label, label_x, label_y) -> None:

        widget_name.figure.clear()

        ax = widget_name.figure.add_subplot(111)
        ax.plot(data_x, data_y, label=label)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.legend()
        ax.grid(True)

        if pd.api.types.is_datetime64_any_dtype(data_x):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            widget_name.figure.autofmt_xdate()

        widget_name.canvas.draw()

    def load_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if not file_name:
            return

        self.ui.name_file_label.setText(f"Selected file: {file_name}")
        self.df = pd.read_csv(file_name)

        for col in self.df.columns:
            if pd.api.types.is_string_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass

        if len(self.df.columns) > 0: 
            current_first_column = self.df.columns[0]  
            self.df.rename(columns={current_first_column: "index"}, inplace=True)

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].round(4)  

        self.display_table(self.df, self.ui.table_data)

        self.ui.box_x.clear()
        self.ui.box_y.clear()
        self.ui.box_x.addItems(self.df.columns)
        self.ui.box_y.addItems(self.df.columns)

        self.ui.data_graph_btn.setEnabled(True)

    def plot_graph(self):
        try:
            x_column = self.ui.box_x.currentText()
            y_column = self.ui.box_y.currentText()
            self.plot(self.ui.widget_4, self.df[x_column], self.df[y_column], f"{y_column} vs {x_column}", x_column, y_column)
        except Exception as e:
            print(f"Ошибка при построении графика: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
