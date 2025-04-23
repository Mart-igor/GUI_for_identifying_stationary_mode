import sys
import os
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QButtonGroup, 
                              QFileDialog, QTableWidgetItem, QMessageBox)
from PySide6.QtCore import QPropertyAnimation, Qt, QSize, QEasingCurve
from PySide6.QtGui import (QIcon, QTextDocument, QTextCursor, 
                          QTextTableFormat, QTextCharFormat, 
                          QTextFrameFormat, QPixmap)
from PySide6.QtPrintSupport import QPrinter

import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from scipy.stats import f
from sklearn.covariance import MinCovDet

from ui.ui_station_interface import Ui_MainWindow
from mplwidget import MplWidgetOptimize, MplWidget
from report import ReportPrinter

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
        self.ui.correlation_btn.clicked.connect(self.calculate)

        self.ui.optimize.clicked.connect(self.calculate_stationary)
        self.ui.next_page_1.clicked.connect(self.go_to_page_2)
        self.ui.graph_opt.clicked.connect(self.plot_graph_optimize)
        self.ui.add_stationar.clicked.connect(self.on_add_stationary_clicked)
        self.ui.search_k.clicked.connect(self.ratio_calculation)
        self.ui.graph_result.clicked.connect(self.plot_graph_result)

        # печать
        self.printer = ReportPrinter(self)
        self.ui.print_btn.clicked.connect(self.on_print)
        self.ui.export_pdf_btn.clicked.connect(self.export_to_pdf)
        self.ui.export_csv_btn.clicked.connect(self.export_to_csv)
        self.ui.export_json_btn.clicked.connect(self.export_to_json)
        self.ui.pushButton_4.clicked.connect(self.export_to_excel)

        # Подключение событий мыши widget12
        self.ui.widget_12.connect_events(self)

        # settings
        self.ui.alg_box.setCurrentText("Choose the algorithm")
        self.ui.alg_box.addItem("Одномерные конрольные карты")
        self.ui.alg_box.addItem("Многомерные конрольные карты")

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
        algorithm = self.ui.alg_box.currentText()
        if algorithm == "Одномерные конрольные карты":
            self.ui.correlation_btn.setEnabled(False)
            self.ui.next_page_1.setEnabled(True)
            self.ui.stackedWidget.setCurrentIndex(1) 
        elif algorithm == "Многомерные конрольные карты":
            self.ui.next_page_1.setEnabled(False)
            self.ui.correlation_btn.setEnabled(True)
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
        elif algorithm == "Многомерные конрольные карты":
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

    def find_stable_segment(self, data, min_length=30, rel_threshold=0.05):  # rel_threshold = 5%
        value_columns = data.columns[2:]
        
        for i in range(len(data) - min_length):
            segment = data.iloc[i:i+min_length]
            median_vals = segment[value_columns].median()
            mad = np.median(np.abs(segment[value_columns] - median_vals), axis=0)
            rMAD = mad / np.abs(median_vals)  # Относительное отклонение
            
            if (rMAD < rel_threshold).mean() > 0.9 and np.median(rMAD) < rel_threshold * 1.5:
                return segment
        raise ValueError("Стабильный участок не найден")

    def adaptive_monitoring(self, min_length=20, var_threshold=0.1, 
                      l1=0.05, l2=0.01, transition_window=20, 
                      alpha=0.05):
        if self.df is None:
            raise ValueError("Сначала загрузите данные с помощью load_csv()")
        
        # Выбираем только числовые параметры (исключаем 'index' и 'time')
        param_columns = [col for col in self.df.columns 
                        if col not in ['index', 'time'] and 
                        pd.api.types.is_numeric_dtype(self.df[col])]
        
        if not param_columns:
            raise ValueError("Нет числовых параметров для анализа")
        
        data_values = self.df[param_columns].values  # Только числовые значения
        timestamps = self.df['time']  # Сохраняем временные метки
        
        p = len(param_columns)  # количество параметров
        
        # 1. Инициализация на стабильном сегменте
        stable_segment = self.find_stable_segment(self.df, min_length, var_threshold)
        if stable_segment is None or len(stable_segment) < min_length:
            raise ValueError(f"Не удалось найти стабильный сегмент длиной {min_length}")
        
        stable_values = stable_segment[param_columns].values
        
        # 2. Начальные параметры модели
        try:
            robust_cov = MinCovDet(support_fraction=0.75).fit(stable_values).covariance_
        except Exception as e:
            print(f"Ошибка MinCovDet: {e}, используем стандартную ковариацию")
            robust_cov = np.cov(stable_values, rowvar=False)
        
        mean = np.median(stable_values, axis=0)
        cov = robust_cov.copy()
        n_eff = max(int(2/l1), p+1)  # эффективный объем выборки
        
        transition_flag = False
        anomaly_counter = 0
        results = []
        
        # 3. Адаптивный мониторинг
        for t in range(len(stable_segment), len(data_values)):
            x_t = data_values[t]
            
            try:
                # 4. Расчет статистики T²
                diff = x_t - mean
                inv_cov = np.linalg.pinv(cov + 1e-6*np.eye(p))  # регуляризация
                T2 = diff @ inv_cov @ diff
                
                # 5. F-преобразование
                F = (n_eff - p) * T2 / (p * (n_eff - 1)) if n_eff > p else T2
                F_threshold = f.ppf(1-alpha, p, n_eff-p) * 2
                
                # 6. Детектирование аномалий
                is_anomaly = F > F_threshold
                results.append({
                    'timestamp': timestamps.iloc[t],
                    'F': F,
                    'threshold': F_threshold,
                    'anomaly': is_anomaly,
                    'transition': transition_flag,
                    **dict(zip(param_columns, x_t))  # сохраняем значения параметров
                })
                
                # 7. Логика переходных процессов
                if is_anomaly:
                    anomaly_counter += 1
                    if anomaly_counter >= transition_window and not transition_flag:
                        print(f"Обнаружен переходный процесс в {timestamps.iloc[t]}")
                        transition_flag = True
                        anomaly_counter = 0
                else:
                    anomaly_counter = max(0, anomaly_counter - 1)
                
                # 8. Обновление параметров модели
                if not transition_flag:
                    # Экспоненциальное сглаживание
                    mean = l1 * x_t + (1 - l1) * mean
                    
                    if not is_anomaly:
                        residual = x_t - mean
                        cov = l2 * np.outer(residual, residual) + (1 - l2) * cov
                else:
                    # Проверка окончания переходного процесса
                    if t + transition_window <= len(data_values):
                        window_values = data_values[t:t+transition_window]
                        mad = np.median(np.abs(window_values - np.median(window_values, axis=0)), axis=0)
                        if (mad < var_threshold).all():
                            mean = np.median(window_values, axis=0)
                            cov = np.cov(window_values, rowvar=False)
                            transition_flag = False
                            print(f"Новый стабильный режим с {timestamps.iloc[t]}")
                            n_eff = max(int(2/l1), p+1)
            
            except Exception as e:
                print(f"Ошибка на шаге {t}: {str(e)}")
                continue
        
        self.results = pd.DataFrame(results).set_index('timestamp')
        return self.results

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

    def calculate_stationary(self):
        try:
            x_min = int(self.ui.x_min.text())
            x_max = int(self.ui.x_max.text())

            column_name = self.ui.box_y.currentText()

            if column_name not in self.df.columns:
                raise ValueError(f"Столбец '{column_name}' не найден в DataFrame")

            data_PT = self.df[[column_name]]
            range_max_min = x_max - x_min

            indexes_v3 = [i for i in range(x_min, x_max, 1)]
            self.data_v3 = data_PT.iloc[indexes_v3]

            rr = self.algorith(self.data_v3, range_max_min)
            self.data_graph = pd.concat([self.data_v3, rr], axis=1)
            self.data_graph['assessment'] = 0
            self.display_table(self.data_graph, self.ui.table_classification)            

        except Exception as e:
            print(f"Ошибка: {e}")

    def plot_graph_optimize(self):
        try:
            table = self.ui.table_classification 
            row_count = table.rowCount()
            column_count = table.columnCount()

            if row_count == 0 or column_count == 0:
                print("Ошибка: таблица пуста.")
                return
            
            y_data = []
            for i in range(row_count):
                item = table.item(i, 0) 
                if item is not None:
                    try:
                        y_value = float(item.text()) 
                        y_data.append(y_value)
                    except ValueError:
                        print(f"Ошибка: значение в строке {i} не является числом.")
                        return
                    
            x_min = int(self.ui.x_min.text())
            x_max = int(self.ui.x_max.text())

            x_data = range(x_min, x_max)
            self.plot(self.ui.widget_12, x_data, y_data, "График по первому столбцу", "Индекс строки", "Значение первого столбца")
        except Exception as e:
            print(f"Ошибка при построении графика: {e}")
            
    def on_press(self, event):
        """Обработка нажатия мыши"""
        if event.inaxes is not None:
            self.selection_start = event.xdata 
    
    def calculate(self):
        self.table_result = self.adaptive_monitoring()
        self.display_table(self.table_result, self.ui.report_table)
        self.plot_results()

    def plot_results(self):
        if not hasattr(self, 'results'):
            raise ValueError("Сначала выполните adaptive_monitoring()")
        
        # Очищаем предыдущий график
        self.ui.widget_4.figure.clear()
        
        # Создаем оси для нового графика
        ax = self.ui.widget_4.figure.add_subplot(111)
        
        # Строим график
        ax.plot(self.results.index, self.results['F'], label='F-статистика')
        ax.plot(self.results.index, self.results['threshold'], 'r--', label='Порог')
        ax.scatter(self.results[self.results['anomaly']].index,
                self.results[self.results['anomaly']]['F'],
                color='red', label='Аномалии')
        
        # Добавляем легенду и заголовок
        ax.legend()
        ax.set_title('Результаты мониторинга')
        
        # Автоматическое масштабирование осей
        ax.relim()
        ax.autoscale_view()
        
        # Обновляем холст
        self.ui.widget_4.canvas.draw()

    def on_release(self, event):
        """Обработка отпускания мыши"""
        if event.inaxes is not None and self.selection_start is not None:
            self.selection_end = event.xdata  

            self.selected_regions.append((self.selection_start, self.selection_end))

            self.highlight_selected_region(self.selection_start, self.selection_end)

            self.selection_start = None
            self.selection_end = None

    def highlight_selected_region(self, start, end):
        """Перекрашивание выделенного участка на графике"""
        ax = self.ui.widget_12.figure.axes[0]
        height = ax.get_ylim()[1] - ax.get_ylim()[0] 
        rect = Rectangle((start, ax.get_ylim()[0]), end - start, height,
                         color='yellow', alpha=0.3)
        ax.add_patch(rect)
        self.ui.widget_12.canvas.draw()
        
    def on_add_stationary_clicked(self):
        """Обработка нажатия кнопки 'add_stationary'"""
        for start, end in self.selected_regions:
            start_index = int(np.floor(start))
            end_index = int(np.ceil(end))

            start_index = max(0, min(start_index, len(self.df) - 1))
            end_index = max(0, min(end_index, len(self.df) - 1))

            self.data_graph.loc[start_index:end_index, 'assessment'] = 1

        self.display_table(self.data_graph, self.ui.table_classification)

        self.y_true = np.array(self.data_graph['assessment'].iloc[51:])

        self.selected_regions.clear()

        self.clear_highlights()

    def clear_highlights(self):
        """Очистка всех выделений на графике"""
        ax = self.ui.widget_12.figure.axes[0]
        for patch in ax.patches:
            patch.remove()
        self.ui.widget_12.canvas.draw()

    def search_ratio(self, x):
        x_min = int(self.ui.x_min.text())
        x_max = int(self.ui.x_max.text())
        x_range = x_max - x_min

        rr = self.algorith(self.data_v3, x_range, x)
        score = f1_score(self.y_true, rr['stationary'], average='binary')

        return -score

    def ratio_calculation(self):
        x0 = [0.5, 0.5, 0.5]
        bnds = ((0, 1), (0, 1), (0, 1))

        def callback(xk, *args, **kwargs):
            print(f'current ratio: {xk}')

        result = minimize(self.search_ratio, x0, method="Powell", bounds=bnds, callback=callback)
        self.all_ratios = result.x
        self.essential_f = result.fun
        self.ui.k_1.setText(f"K_1: {round(self.all_ratios[0], 4)}")
        self.ui.k_2.setText(f"K_2: {round(self.all_ratios[1], 4)}")
        self.ui.k_3.setText(f"K_3: {round(self.all_ratios[2], 4)}")
        self.ui.F_score.setText(f"F_score: {abs(round(self.essential_f, 4))}")

    def plot_graph_result(self):
        column_name = self.ui.box_y.currentText()
        x_min = 0
        x_max = len(self.df)
        range_max_min = x_max - x_min

        rr = self.algorith(self.df[[column_name]], range_max_min, self.all_ratios)
        self.table_result = pd.concat([self.df, rr], axis=1)

        if 'time' not in self.table_result.columns:
            print("Столбец 'time' не найден в данных")
            return

        self.ui.widget_12.figure.clear()

        ax = self.ui.widget_12.figure.add_subplot(111)

        ax.plot(self.table_result['time'], 
            self.table_result[column_name], 
            'g-', alpha=0.5, label=column_name)

        non_stationary_data = self.table_result[self.table_result['stationary'] == 0]
        ax.plot(non_stationary_data['time'], 
                non_stationary_data[column_name], 
                'bo', markersize=2, label='Нестационарные точки')

        stationary_data = self.table_result[self.table_result['stationary'] == 1]
        ax.plot(stationary_data['time'], 
                stationary_data[column_name], 
                'r*', markersize=2, label='Стационарные точки')

        ax.set_xlabel('Время')
        ax.set_ylabel("Значение")
        ax.legend()
        ax.grid(True)

        self.ui.widget_12.canvas.draw()
        self.display_table(self.table_result, self.ui.report_table)

    def on_print(self):
        if hasattr(self, 'table_result'):
            self.printer.print_data(self.table_result, "DataFrame Report")

    def export_to_pdf(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить PDF", "", "PDF Files (*.pdf)")
        
        if filename:
            try:
                printer = QPrinter(QPrinter.HighResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(filename)
                
                doc = QTextDocument()
                cursor = QTextCursor(doc)
                
                title_format = QTextCharFormat()
                title_format.setFontPointSize(14)
                title_format.setFontWeight(75)
                cursor.insertText("Отчет\n", title_format)
                
                table_format = QTextTableFormat()
                table_format.setAlignment(Qt.AlignHCenter)
                table_format.setBorderStyle(QTextFrameFormat.BorderStyle_Solid)
                
                table = cursor.insertTable(
                    self.table_result.shape[0] + 1, 
                    self.table_result.shape[1], 
                    table_format
                )
                
                for col in range(self.table_result.shape[1]):
                    cursor.insertText(self.table_result.columns[col])
                    cursor.movePosition(QTextCursor.NextCell)
                
                for row in range(self.table_result.shape[0]):
                    for col in range(self.table_result.shape[1]):
                        cursor.insertText(str(self.table_result.iloc[row, col]))
                        cursor.movePosition(QTextCursor.NextCell)
                
                doc.print_(printer)
                QMessageBox.information(self, "Успех", "PDF успешно сохранен!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка: {str(e)}")
  
    def export_to_csv(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV", "", "CSV Files (*.csv)")
        
        if filename:
            try:
                self.table_result.to_csv(filename, index=False, encoding='utf-8-sig', sep=';')
                QMessageBox.information(self, "Успех", "CSV успешно сохранен!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка: {str(e)}")
    
    def export_to_json(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить JSON", "", "JSON Files (*.json)")
        
        if filename:
            try:
                data_dict = self.table_result.to_dict(orient='records')
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data_dict, f, ensure_ascii=False, indent=4)
                
                QMessageBox.information(self, "Успех", "JSON успешно сохранен!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка: {str(e)}")
    
    def export_to_excel(self):
        try:
            import openpyxl  
        except ImportError:
            QMessageBox.critical(
                self, 
                "Ошибка", 
                "Для экспорта в Excel требуется модуль openpyxl.\n"
                "Установите его командой: pip install openpyxl"
            )
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить Excel", "", "Excel Files (*.xlsx)")
        
        if filename:
            try:
                if not filename.endswith('.xlsx'):
                    filename += '.xlsx'
                
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    self.table_result.to_excel(writer, index=False, sheet_name='Данные')
                
                QMessageBox.information(self, "Успех", "Excel файл успешно сохранен!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
