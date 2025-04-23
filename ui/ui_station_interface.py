# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'stationxLxohx.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFontComboBox, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QLayout,
    QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QRadioButton, QSizePolicy, QSpacerItem, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit,
    QVBoxLayout, QWidget)

from mplwidget import (MplWidget, MplWidgetOptimize)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1096, 789)
        MainWindow.setStyleSheet(u"/* Global reset */\n"
"* {\n"
"    border: none;\n"
"    background-color: transparent;\n"
"    padding: 0;\n"
"    margin: 0;\n"
"    color: #fff;\n"
"}\n"
"\n"
"/* Main containers */\n"
"#centralwidget, #instruction_btn, #home_btn, #algorithm_btn, #main_content, QLineEdit {\n"
"    background-color: #1b1b27;\n"
"}\n"
"\n"
"#header, #main_body {\n"
"    background-color: #27263c;\n"
"}\n"
"\n"
"#footer {\n"
"    background: qradialgradient(\n"
"        cx: 0.5, cy: 0.5, radius: 1,\n"
"        fx: 0.5, fy: 0.5,\n"
"        stop: 0 #1b1b27,\n"
"        stop: 1 #27263c\n"
"    );\n"
"}\n"
"\n"
"/* Buttons */\n"
"QPushButton {\n"
"    text-align: left;\n"
"    padding: 3px 10px;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"#home_btn, #reports_btn, #algorithm_btn, #instruction_btn, #account_btn {\n"
"    border-left: 3px solid #cc5bce;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"#add_user_btn {\n"
"    background-color: #1b1b27;\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"#pushButton {\n"
"    border: 1px solid #cc5b"
                        "ce;\n"
"    border-radius: 16px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"#main_content QPushButton, QStackedWidget QPushButton {\n"
"    background-color: #3a3854;\n"
"    color: white;\n"
"    font-size: 9pt;\n"
"    font-family: Arial;\n"
"    text-align: center;\n"
"    border: 1px solid #4a4864;\n"
"    border-radius: 4px;\n"
"    padding: 5px;\n"
"}\n"
"\n"
"#main_content QPushButton:hover, QStackedWidget QPushButton:hover {\n"
"    background-color: #4a4864;\n"
"    color: #a0a0a0;\n"
"    border: 1px solid #5a5874;\n"
"}\n"
"\n"
"#main_content QPushButton:pressed, QStackedWidget QPushButton:pressed {\n"
"    background-color: #2a2844;\n"
"    color: #c0c0c0;\n"
"}\n"
"\n"
"/* Input fields */\n"
"QLineEdit, #main_content QLineEdit {\n"
"    background-color: #2a2a4c;\n"
"    color: #ffffff;\n"
"    border: 1px solid #3a3a5c;\n"
"    padding: 5px 10px;\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"#main_content QLineEdit:focus {\n"
"    border: 1px solid #4a4a6c;\n"
"}\n"
"\n"
"/* Combo boxes */\n"
"QComb"
                        "oBox, #main_content QComboBox {\n"
"    background-color: #2a2a4c;\n"
"    color: #ffffff;\n"
"    border: 1px solid #3a3a5c;\n"
"    padding: 5px;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    background-color: white;\n"
"    color: black;\n"
"}\n"
"\n"
"/* Labels */\n"
"QStackedWidget QLabel, #main_content QLabel {\n"
"    color: white;\n"
"    font-family: Arial;\n"
"    text-align: center;\n"
"}\n"
"\n"
"#main_content QLabel:hover {\n"
"    color: #a0a0a0;\n"
"}\n"
"\n"
"#label_4, #label_5, #language_lbl, #theme_lbl, #forn_lbl, #graphs_lbl, #export_lbl, #choose_alg_lbl {\n"
"    font-size: 12pt;\n"
"}\n"
"\n"
"#data_label, #label_11, #report_label, #label_25, #label_3 {\n"
"    font-size: 14pt;\n"
"}\n"
"\n"
"/* Table widget */\n"
"QTableWidget {\n"
"    background-color: #1b1b27;\n"
"    border: 1px solid #3a3a5c;\n"
"    border-radius: 4px;\n"
"    gridline-color: #3a3a5c;\n"
"    font-size: 14px;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"    background-color: #27263c;\n"
"    color: #ffffff;\n"
""
                        "    padding: 8px;\n"
"    border: none;\n"
"    border-bottom: 2px solid #3a3a5c;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QTableWidget::item {\n"
"    color: #ffffff;\n"
"    background-color: #1b1b27;\n"
"    padding: 6px;\n"
"}\n"
"\n"
"QTableWidget::item:alternate {\n"
"    background-color: #252536;\n"
"}\n"
"\n"
"QTableWidget::item:selected, QTableWidget::item:hover {\n"
"    background-color: #3a3a5c;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"    background: #1b1b27;\n"
"    width: 12px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"    background: #3a3a5c;\n"
"    min-height: 20px;\n"
"    border-radius: 6px;\n"
"}\n"
"\n"
"QTableCornerButton::section {\n"
"    background-color: #27263c;\n"
"    border: none;\n"
"    border-bottom: 2px solid #3a3a5c;\n"
"}\n"
"\n"
"/* Progress bar */\n"
"#main_content QProgressBar {\n"
"    background-color: #1b1b27;\n"
"    border: 1px solid #3a3a5c;\n"
"}\n"
"\n"
"#main_content QProgressBar::chunk {\n"
"    background-color: #3a3a5c;\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.header = QWidget(self.centralwidget)
        self.header.setObjectName(u"header")
        self.header.setMinimumSize(QSize(0, 70))
        self.horizontalLayout_2 = QHBoxLayout(self.header)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(-1, 0, 0, 0)
        self.frame = QFrame(self.header)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.menu_btn = QPushButton(self.frame)
        self.menu_btn.setObjectName(u"menu_btn")
        font = QFont()
        font.setPointSize(14)
        self.menu_btn.setFont(font)
        self.menu_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.AddressBookNew))
        self.menu_btn.setIcon(icon)
        self.menu_btn.setIconSize(QSize(20, 20))

        self.horizontalLayout_4.addWidget(self.menu_btn, 0, Qt.AlignmentFlag.AlignLeft)


        self.horizontalLayout_2.addWidget(self.frame, 0, Qt.AlignmentFlag.AlignLeft)

        self.label = QLabel(self.header)
        self.label.setObjectName(u"label")
        font1 = QFont()
        font1.setFamilies([u"Noto Serif Thai"])
        font1.setPointSize(16)
        font1.setBold(True)
        self.label.setFont(font1)

        self.horizontalLayout_2.addWidget(self.label)

        self.frame_2 = QFrame(self.header)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.notifications = QPushButton(self.frame_2)
        self.notifications.setObjectName(u"notifications")
        self.notifications.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.notifications.setIcon(icon)

        self.horizontalLayout_3.addWidget(self.notifications, 0, Qt.AlignmentFlag.AlignRight)

        self.signin = QPushButton(self.frame_2)
        self.signin.setObjectName(u"signin")
        self.signin.setFont(font)
        self.signin.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_3.addWidget(self.signin, 0, Qt.AlignmentFlag.AlignRight)


        self.horizontalLayout_2.addWidget(self.frame_2, 0, Qt.AlignmentFlag.AlignRight)


        self.verticalLayout.addWidget(self.header, 0, Qt.AlignmentFlag.AlignTop)

        self.main_body = QWidget(self.centralwidget)
        self.main_body.setObjectName(u"main_body")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.main_body.sizePolicy().hasHeightForWidth())
        self.main_body.setSizePolicy(sizePolicy)
        self.horizontalLayout = QHBoxLayout(self.main_body)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 10, 0, 0)
        self.left_menu = QWidget(self.main_body)
        self.left_menu.setObjectName(u"left_menu")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.left_menu.sizePolicy().hasHeightForWidth())
        self.left_menu.setSizePolicy(sizePolicy1)
        self.left_menu.setMinimumSize(QSize(50, 0))
        self.left_menu.setMaximumSize(QSize(200, 16777215))
        self.verticalLayout_3 = QVBoxLayout(self.left_menu)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.widget_2 = QWidget(self.left_menu)
        self.widget_2.setObjectName(u"widget_2")
        font2 = QFont()
        font2.setPointSize(12)
        self.widget_2.setFont(font2)
        self.verticalLayout_4 = QVBoxLayout(self.widget_2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, -1, 0, 0)
        self.frame_3 = QFrame(self.widget_2)
        self.frame_3.setObjectName(u"frame_3")
        font3 = QFont()
        font3.setPointSize(11)
        font3.setBold(False)
        self.frame_3.setFont(font3)
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame_3)
        self.verticalLayout_5.setSpacing(15)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(5, 5, 0, 0)
        self.home_btn = QPushButton(self.frame_3)
        self.home_btn.setObjectName(u"home_btn")
        font4 = QFont()
        font4.setFamilies([u"Segoe UI Emoji"])
        font4.setPointSize(12)
        font4.setBold(True)
        self.home_btn.setFont(font4)
        self.home_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.home_btn.setIcon(icon)

        self.verticalLayout_5.addWidget(self.home_btn)

        self.algorithm_btn = QPushButton(self.frame_3)
        self.algorithm_btn.setObjectName(u"algorithm_btn")
        self.algorithm_btn.setFont(font4)
        self.algorithm_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.algorithm_btn.setMouseTracking(True)
        self.algorithm_btn.setIcon(icon)

        self.verticalLayout_5.addWidget(self.algorithm_btn)

        self.instruction_btn = QPushButton(self.frame_3)
        self.instruction_btn.setObjectName(u"instruction_btn")
        self.instruction_btn.setFont(font4)
        self.instruction_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.instruction_btn.setIcon(icon)

        self.verticalLayout_5.addWidget(self.instruction_btn)

        self.reports_btn = QPushButton(self.frame_3)
        self.reports_btn.setObjectName(u"reports_btn")
        self.reports_btn.setFont(font4)
        self.reports_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.reports_btn.setIcon(icon)

        self.verticalLayout_5.addWidget(self.reports_btn)

        self.account_btn = QPushButton(self.frame_3)
        self.account_btn.setObjectName(u"account_btn")
        self.account_btn.setEnabled(True)
        self.account_btn.setFont(font4)
        self.account_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.account_btn.setIcon(icon)

        self.verticalLayout_5.addWidget(self.account_btn)


        self.verticalLayout_4.addWidget(self.frame_3)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer)

        self.frame_4 = QFrame(self.widget_2)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.frame_4)
        self.verticalLayout_6.setSpacing(6)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(5, -1, 0, 0)
        self.help_btn = QPushButton(self.frame_4)
        self.help_btn.setObjectName(u"help_btn")
        self.help_btn.setFont(font2)
        self.help_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.help_btn.setIcon(icon)

        self.verticalLayout_6.addWidget(self.help_btn)

        self.settings_btn = QPushButton(self.frame_4)
        self.settings_btn.setObjectName(u"settings_btn")
        self.settings_btn.setFont(font2)
        self.settings_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.settings_btn.setIcon(icon)

        self.verticalLayout_6.addWidget(self.settings_btn)

        self.about_btn = QPushButton(self.frame_4)
        self.about_btn.setObjectName(u"about_btn")
        self.about_btn.setFont(font2)
        self.about_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.about_btn.setIcon(icon)

        self.verticalLayout_6.addWidget(self.about_btn)


        self.verticalLayout_4.addWidget(self.frame_4)


        self.verticalLayout_3.addWidget(self.widget_2)


        self.horizontalLayout.addWidget(self.left_menu)

        self.main_content = QWidget(self.main_body)
        self.main_content.setObjectName(u"main_content")
        sizePolicy1.setHeightForWidth(self.main_content.sizePolicy().hasHeightForWidth())
        self.main_content.setSizePolicy(sizePolicy1)
        self.verticalLayout_2 = QVBoxLayout(self.main_content)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.stackedWidget = QStackedWidget(self.main_content)
        self.stackedWidget.setObjectName(u"stackedWidget")
        sizePolicy1.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy1)
        self.home_page = QWidget()
        self.home_page.setObjectName(u"home_page")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.home_page.sizePolicy().hasHeightForWidth())
        self.home_page.setSizePolicy(sizePolicy2)
        self.home_page.setMaximumSize(QSize(16667, 16667))
        self.horizontalLayout_24 = QHBoxLayout(self.home_page)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.widget_14 = QWidget(self.home_page)
        self.widget_14.setObjectName(u"widget_14")
        sizePolicy2.setHeightForWidth(self.widget_14.sizePolicy().hasHeightForWidth())
        self.widget_14.setSizePolicy(sizePolicy2)
        self.widget_14.setMinimumSize(QSize(0, 20))
        self.horizontalLayout_21 = QHBoxLayout(self.widget_14)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.textEdit = QTextEdit(self.widget_14)
        self.textEdit.setObjectName(u"textEdit")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy3)
        self.textEdit.setMinimumSize(QSize(500, 0))
        self.textEdit.setMaximumSize(QSize(500, 16777215))

        self.horizontalLayout_21.addWidget(self.textEdit, 0, Qt.AlignmentFlag.AlignHCenter)


        self.horizontalLayout_24.addWidget(self.widget_14)

        self.stackedWidget.addWidget(self.home_page)
        self.algorithm_page = QWidget()
        self.algorithm_page.setObjectName(u"algorithm_page")
        self.algorithm_page.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.algorithm_page.sizePolicy().hasHeightForWidth())
        self.algorithm_page.setSizePolicy(sizePolicy1)
        self.algorithm_page.setMinimumSize(QSize(590, 663))
        self.algorithm_page.setMaximumSize(QSize(16777214, 16777214))
        self.verticalLayout_10 = QVBoxLayout(self.algorithm_page)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.widget_6 = QWidget(self.algorithm_page)
        self.widget_6.setObjectName(u"widget_6")
        self.widget_6.setMinimumSize(QSize(100, 17))
        self.horizontalLayout_7 = QHBoxLayout(self.widget_6)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.next_page_1 = QPushButton(self.widget_6)
        self.next_page_1.setObjectName(u"next_page_1")
        self.next_page_1.setMinimumSize(QSize(0, 22))
        font5 = QFont()
        font5.setFamilies([u"Arial"])
        font5.setPointSize(9)
        self.next_page_1.setFont(font5)
        self.next_page_1.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_7.addWidget(self.next_page_1)


        self.verticalLayout_10.addWidget(self.widget_6, 0, Qt.AlignmentFlag.AlignRight)

        self.widget_3 = QWidget(self.algorithm_page)
        self.widget_3.setObjectName(u"widget_3")
        sizePolicy1.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy1)
        self.widget_3.setMinimumSize(QSize(0, 0))
        self.widget_3.setMaximumSize(QSize(16777215, 16777215))
        self.widget_3.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.widget_3.setAutoFillBackground(False)
        self.verticalLayout_12 = QVBoxLayout(self.widget_3)
        self.verticalLayout_12.setSpacing(0)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.verticalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.frame_6 = QFrame(self.widget_3)
        self.frame_6.setObjectName(u"frame_6")
        sizePolicy1.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy1)
        self.frame_6.setMinimumSize(QSize(0, 0))
        self.frame_6.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_11 = QVBoxLayout(self.frame_6)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.data_label = QLabel(self.frame_6)
        self.data_label.setObjectName(u"data_label")
        sizePolicy2.setHeightForWidth(self.data_label.sizePolicy().hasHeightForWidth())
        self.data_label.setSizePolicy(sizePolicy2)
        self.data_label.setMaximumSize(QSize(16777215, 100))
        font6 = QFont()
        font6.setFamilies([u"Arial"])
        font6.setPointSize(14)
        self.data_label.setFont(font6)

        self.verticalLayout_11.addWidget(self.data_label)


        self.verticalLayout_12.addWidget(self.frame_6, 0, Qt.AlignmentFlag.AlignHCenter)

        self.frame_7 = QFrame(self.widget_3)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.frame_10 = QFrame(self.frame_7)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_10.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_15 = QVBoxLayout(self.frame_10)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.load_btn = QPushButton(self.frame_10)
        self.load_btn.setObjectName(u"load_btn")
        self.load_btn.setFont(font5)
        self.load_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_15.addWidget(self.load_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.name_file_label = QLabel(self.frame_10)
        self.name_file_label.setObjectName(u"name_file_label")

        self.verticalLayout_15.addWidget(self.name_file_label)

        self.progressBar = QProgressBar(self.frame_10)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setMinimumSize(QSize(300, 0))
        self.progressBar.setValue(0)

        self.verticalLayout_15.addWidget(self.progressBar)

        self.correlation_btn = QPushButton(self.frame_10)
        self.correlation_btn.setObjectName(u"correlation_btn")

        self.verticalLayout_15.addWidget(self.correlation_btn)


        self.horizontalLayout_6.addWidget(self.frame_10)

        self.frame_8 = QFrame(self.frame_7)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setMinimumSize(QSize(50, 0))
        self.frame_8.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_6.addWidget(self.frame_8)

        self.frame_11 = QFrame(self.frame_7)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_16 = QVBoxLayout(self.frame_11)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.frame_12 = QFrame(self.frame_11)
        self.frame_12.setObjectName(u"frame_12")
        self.frame_12.setMinimumSize(QSize(100, 0))
        self.frame_12.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_12)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.box_x = QComboBox(self.frame_12)
        self.box_x.setObjectName(u"box_x")

        self.horizontalLayout_9.addWidget(self.box_x)

        self.frame_9 = QFrame(self.frame_12)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setMinimumSize(QSize(25, 0))
        self.frame_9.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_9.addWidget(self.frame_9)

        self.box_y = QComboBox(self.frame_12)
        self.box_y.setObjectName(u"box_y")

        self.horizontalLayout_9.addWidget(self.box_y)


        self.verticalLayout_16.addWidget(self.frame_12)

        self.data_graph_btn = QPushButton(self.frame_11)
        self.data_graph_btn.setObjectName(u"data_graph_btn")
        self.data_graph_btn.setMinimumSize(QSize(0, 0))
        self.data_graph_btn.setFont(font5)
        self.data_graph_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_16.addWidget(self.data_graph_btn, 0, Qt.AlignmentFlag.AlignHCenter)


        self.horizontalLayout_6.addWidget(self.frame_11)


        self.verticalLayout_12.addWidget(self.frame_7)


        self.verticalLayout_10.addWidget(self.widget_3, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.widget_5 = QWidget(self.algorithm_page)
        self.widget_5.setObjectName(u"widget_5")
        sizePolicy3.setHeightForWidth(self.widget_5.sizePolicy().hasHeightForWidth())
        self.widget_5.setSizePolicy(sizePolicy3)
        self.widget_5.setMinimumSize(QSize(0, 0))
        self.widget_5.setMaximumSize(QSize(167777, 1677777))
        self.horizontalLayout_20 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_20.setSpacing(0)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_3 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_3)

        self.table_data = QTableWidget(self.widget_5)
        self.table_data.setObjectName(u"table_data")
        self.table_data.setMinimumSize(QSize(700, 200))

        self.horizontalLayout_20.addWidget(self.table_data)

        self.horizontalSpacer_4 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_4)


        self.verticalLayout_10.addWidget(self.widget_5)

        self.widget_91 = QWidget(self.algorithm_page)
        self.widget_91.setObjectName(u"widget_91")
        sizePolicy1.setHeightForWidth(self.widget_91.sizePolicy().hasHeightForWidth())
        self.widget_91.setSizePolicy(sizePolicy1)
        self.widget_91.setMinimumSize(QSize(0, 300))
        self.widget_91.setMaximumSize(QSize(16777215, 600))
        self.verticalLayout_13 = QVBoxLayout(self.widget_91)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.widget_4 = MplWidget(self.widget_91)
        self.widget_4.setObjectName(u"widget_4")
        sizePolicy1.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy1)
        self.widget_4.setMinimumSize(QSize(0, 0))

        self.verticalLayout_13.addWidget(self.widget_4)


        self.verticalLayout_10.addWidget(self.widget_91)

        self.stackedWidget.addWidget(self.algorithm_page)
        self.second_page = QWidget()
        self.second_page.setObjectName(u"second_page")
        self.verticalLayout_18 = QVBoxLayout(self.second_page)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.verticalLayout_18.setContentsMargins(0, 9, 0, 0)
        self.widget_7 = QWidget(self.second_page)
        self.widget_7.setObjectName(u"widget_7")
        sizePolicy1.setHeightForWidth(self.widget_7.sizePolicy().hasHeightForWidth())
        self.widget_7.setSizePolicy(sizePolicy1)
        self.horizontalLayout_8 = QHBoxLayout(self.widget_7)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.widget_9 = QWidget(self.widget_7)
        self.widget_9.setObjectName(u"widget_9")
        self.verticalLayout_17 = QVBoxLayout(self.widget_9)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.frame_13 = QFrame(self.widget_9)
        self.frame_13.setObjectName(u"frame_13")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.frame_13.sizePolicy().hasHeightForWidth())
        self.frame_13.setSizePolicy(sizePolicy4)
        self.frame_13.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_13.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_12 = QHBoxLayout(self.frame_13)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.label_11 = QLabel(self.frame_13)
        self.label_11.setObjectName(u"label_11")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy5)
        self.label_11.setFont(font6)

        self.horizontalLayout_12.addWidget(self.label_11)


        self.verticalLayout_17.addWidget(self.frame_13, 0, Qt.AlignmentFlag.AlignHCenter)

        self.frame_20 = QFrame(self.widget_9)
        self.frame_20.setObjectName(u"frame_20")
        self.frame_20.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_20.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_13 = QHBoxLayout(self.frame_20)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_12 = QLabel(self.frame_20)
        self.label_12.setObjectName(u"label_12")
        font7 = QFont()
        font7.setFamilies([u"Arial"])
        font7.setPointSize(10)
        self.label_12.setFont(font7)

        self.horizontalLayout_13.addWidget(self.label_12)

        self.gran_val = QLineEdit(self.frame_20)
        self.gran_val.setObjectName(u"gran_val")
        self.gran_val.setEnabled(False)

        self.horizontalLayout_13.addWidget(self.gran_val)


        self.verticalLayout_17.addWidget(self.frame_20)

        self.frame_14 = QFrame(self.widget_9)
        self.frame_14.setObjectName(u"frame_14")
        self.frame_14.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_14.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_10 = QHBoxLayout(self.frame_14)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.frame_15 = QFrame(self.frame_14)
        self.frame_15.setObjectName(u"frame_15")
        self.frame_15.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_15.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_16 = QHBoxLayout(self.frame_15)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.x_min = QLineEdit(self.frame_15)
        self.x_min.setObjectName(u"x_min")

        self.horizontalLayout_16.addWidget(self.x_min)


        self.horizontalLayout_10.addWidget(self.frame_15)

        self.frame_16 = QFrame(self.frame_14)
        self.frame_16.setObjectName(u"frame_16")
        self.frame_16.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_16.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_17 = QHBoxLayout(self.frame_16)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.x_max = QLineEdit(self.frame_16)
        self.x_max.setObjectName(u"x_max")

        self.horizontalLayout_17.addWidget(self.x_max)


        self.horizontalLayout_10.addWidget(self.frame_16)


        self.verticalLayout_17.addWidget(self.frame_14)

        self.frame_18 = QFrame(self.widget_9)
        self.frame_18.setObjectName(u"frame_18")
        self.frame_18.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_18.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.frame_18)
        self.horizontalLayout_14.setSpacing(0)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.optimize = QPushButton(self.frame_18)
        self.optimize.setObjectName(u"optimize")
        self.optimize.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_14.addWidget(self.optimize, 0, Qt.AlignmentFlag.AlignHCenter)


        self.verticalLayout_17.addWidget(self.frame_18)

        self.frame_19 = QFrame(self.widget_9)
        self.frame_19.setObjectName(u"frame_19")
        self.frame_19.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_19.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_15 = QHBoxLayout(self.frame_19)
        self.horizontalLayout_15.setSpacing(0)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.graph_opt = QPushButton(self.frame_19)
        self.graph_opt.setObjectName(u"graph_opt")
        self.graph_opt.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_15.addWidget(self.graph_opt, 0, Qt.AlignmentFlag.AlignHCenter)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_5)

        self.add_stationar = QPushButton(self.frame_19)
        self.add_stationar.setObjectName(u"add_stationar")
        self.add_stationar.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_15.addWidget(self.add_stationar)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_6)

        self.refresh_talbe = QPushButton(self.frame_19)
        self.refresh_talbe.setObjectName(u"refresh_talbe")
        self.refresh_talbe.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_15.addWidget(self.refresh_talbe)


        self.verticalLayout_17.addWidget(self.frame_19)


        self.horizontalLayout_8.addWidget(self.widget_9)

        self.widget_10 = QWidget(self.widget_7)
        self.widget_10.setObjectName(u"widget_10")
        sizePolicy1.setHeightForWidth(self.widget_10.sizePolicy().hasHeightForWidth())
        self.widget_10.setSizePolicy(sizePolicy1)
        self.widget_10.setMinimumSize(QSize(470, 0))
        self.widget_10.setMaximumSize(QSize(16777215, 400))
        self.verticalLayout_19 = QVBoxLayout(self.widget_10)
        self.verticalLayout_19.setSpacing(0)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.verticalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.label_8 = QLabel(self.widget_10)
        self.label_8.setObjectName(u"label_8")

        self.verticalLayout_19.addWidget(self.label_8, 0, Qt.AlignmentFlag.AlignHCenter)

        self.table_classification = QTableWidget(self.widget_10)
        self.table_classification.setObjectName(u"table_classification")
        sizePolicy1.setHeightForWidth(self.table_classification.sizePolicy().hasHeightForWidth())
        self.table_classification.setSizePolicy(sizePolicy1)
        self.table_classification.setMinimumSize(QSize(370, 0))

        self.verticalLayout_19.addWidget(self.table_classification)


        self.horizontalLayout_8.addWidget(self.widget_10)


        self.verticalLayout_18.addWidget(self.widget_7, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.frame_17 = QFrame(self.second_page)
        self.frame_17.setObjectName(u"frame_17")
        self.frame_17.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_17.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_11 = QHBoxLayout(self.frame_17)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.search_k = QPushButton(self.frame_17)
        self.search_k.setObjectName(u"search_k")
        self.search_k.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_11.addWidget(self.search_k)

        self.k_1 = QLabel(self.frame_17)
        self.k_1.setObjectName(u"k_1")
        self.k_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_11.addWidget(self.k_1)

        self.k_2 = QLabel(self.frame_17)
        self.k_2.setObjectName(u"k_2")
        self.k_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_11.addWidget(self.k_2)

        self.k_3 = QLabel(self.frame_17)
        self.k_3.setObjectName(u"k_3")
        self.k_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_11.addWidget(self.k_3)

        self.F_score = QLabel(self.frame_17)
        self.F_score.setObjectName(u"F_score")

        self.horizontalLayout_11.addWidget(self.F_score)


        self.verticalLayout_18.addWidget(self.frame_17, 0, Qt.AlignmentFlag.AlignTop)

        self.widget_11 = QWidget(self.second_page)
        self.widget_11.setObjectName(u"widget_11")
        self.widget_11.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_18 = QHBoxLayout(self.widget_11)
        self.horizontalLayout_18.setSpacing(0)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.horizontalLayout_18.setContentsMargins(0, 0, 0, 9)
        self.graph_result = QPushButton(self.widget_11)
        self.graph_result.setObjectName(u"graph_result")
        self.graph_result.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_18.addWidget(self.graph_result, 0, Qt.AlignmentFlag.AlignHCenter)


        self.verticalLayout_18.addWidget(self.widget_11, 0, Qt.AlignmentFlag.AlignHCenter)

        self.widget_8 = QWidget(self.second_page)
        self.widget_8.setObjectName(u"widget_8")
        sizePolicy1.setHeightForWidth(self.widget_8.sizePolicy().hasHeightForWidth())
        self.widget_8.setSizePolicy(sizePolicy1)
        self.horizontalLayout_19 = QHBoxLayout(self.widget_8)
        self.horizontalLayout_19.setSpacing(0)
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.horizontalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.widget_12 = MplWidgetOptimize(self.widget_8)
        self.widget_12.setObjectName(u"widget_12")

        self.horizontalLayout_19.addWidget(self.widget_12)


        self.verticalLayout_18.addWidget(self.widget_8)

        self.stackedWidget.addWidget(self.second_page)
        self.instruction_page = QWidget()
        self.instruction_page.setObjectName(u"instruction_page")
        sizePolicy1.setHeightForWidth(self.instruction_page.sizePolicy().hasHeightForWidth())
        self.instruction_page.setSizePolicy(sizePolicy1)
        self.verticalLayout_27 = QVBoxLayout(self.instruction_page)
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.widget_19 = QWidget(self.instruction_page)
        self.widget_19.setObjectName(u"widget_19")
        sizePolicy1.setHeightForWidth(self.widget_19.sizePolicy().hasHeightForWidth())
        self.widget_19.setSizePolicy(sizePolicy1)
        self.verticalLayout_28 = QVBoxLayout(self.widget_19)
        self.verticalLayout_28.setObjectName(u"verticalLayout_28")
        self.text_edit_instruction = QTextEdit(self.widget_19)
        self.text_edit_instruction.setObjectName(u"text_edit_instruction")
        self.text_edit_instruction.setReadOnly(True)

        self.verticalLayout_28.addWidget(self.text_edit_instruction)


        self.verticalLayout_27.addWidget(self.widget_19)

        self.stackedWidget.addWidget(self.instruction_page)
        self.report_page = QWidget()
        self.report_page.setObjectName(u"report_page")
        sizePolicy1.setHeightForWidth(self.report_page.sizePolicy().hasHeightForWidth())
        self.report_page.setSizePolicy(sizePolicy1)
        self.verticalLayout_23 = QVBoxLayout(self.report_page)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.widget_15 = QWidget(self.report_page)
        self.widget_15.setObjectName(u"widget_15")
        sizePolicy1.setHeightForWidth(self.widget_15.sizePolicy().hasHeightForWidth())
        self.widget_15.setSizePolicy(sizePolicy1)
        self.verticalLayout_24 = QVBoxLayout(self.widget_15)
        self.verticalLayout_24.setSpacing(0)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(0, 0, 0, 0)
        self.frame_23 = QFrame(self.widget_15)
        self.frame_23.setObjectName(u"frame_23")
        sizePolicy1.setHeightForWidth(self.frame_23.sizePolicy().hasHeightForWidth())
        self.frame_23.setSizePolicy(sizePolicy1)
        font8 = QFont()
        font8.setFamilies([u"Microsoft Yi Baiti"])
        font8.setPointSize(14)
        self.frame_23.setFont(font8)
        self.frame_23.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_23.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_22 = QHBoxLayout(self.frame_23)
        self.horizontalLayout_22.setSpacing(0)
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.horizontalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.report_label = QLabel(self.frame_23)
        self.report_label.setObjectName(u"report_label")
        sizePolicy1.setHeightForWidth(self.report_label.sizePolicy().hasHeightForWidth())
        self.report_label.setSizePolicy(sizePolicy1)
        font9 = QFont()
        font9.setFamilies([u"Arial"])
        font9.setPointSize(14)
        font9.setBold(False)
        self.report_label.setFont(font9)

        self.horizontalLayout_22.addWidget(self.report_label, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)


        self.verticalLayout_24.addWidget(self.frame_23, 0, Qt.AlignmentFlag.AlignTop)


        self.verticalLayout_23.addWidget(self.widget_15, 0, Qt.AlignmentFlag.AlignTop)

        self.widget_16 = QWidget(self.report_page)
        self.widget_16.setObjectName(u"widget_16")
        self.horizontalLayout_23 = QHBoxLayout(self.widget_16)
        self.horizontalLayout_23.setSpacing(0)
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.horizontalLayout_23.setContentsMargins(0, 0, 0, 0)
        self.frame_24 = QFrame(self.widget_16)
        self.frame_24.setObjectName(u"frame_24")
        self.frame_24.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_24.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_25 = QHBoxLayout(self.frame_24)
        self.horizontalLayout_25.setSpacing(0)
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.horizontalLayout_25.setContentsMargins(0, 0, 0, 0)
        self.print_btn = QPushButton(self.frame_24)
        self.print_btn.setObjectName(u"print_btn")
        self.print_btn.setEnabled(True)
        self.print_btn.setMinimumSize(QSize(70, 0))
        self.print_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_25.addWidget(self.print_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.export_pdf_btn = QPushButton(self.frame_24)
        self.export_pdf_btn.setObjectName(u"export_pdf_btn")
        self.export_pdf_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_25.addWidget(self.export_pdf_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.export_csv_btn = QPushButton(self.frame_24)
        self.export_csv_btn.setObjectName(u"export_csv_btn")

        self.horizontalLayout_25.addWidget(self.export_csv_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.export_json_btn = QPushButton(self.frame_24)
        self.export_json_btn.setObjectName(u"export_json_btn")

        self.horizontalLayout_25.addWidget(self.export_json_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.pushButton_4 = QPushButton(self.frame_24)
        self.pushButton_4.setObjectName(u"pushButton_4")

        self.horizontalLayout_25.addWidget(self.pushButton_4, 0, Qt.AlignmentFlag.AlignHCenter)


        self.horizontalLayout_23.addWidget(self.frame_24)


        self.verticalLayout_23.addWidget(self.widget_16)

        self.widget_17 = QWidget(self.report_page)
        self.widget_17.setObjectName(u"widget_17")
        sizePolicy1.setHeightForWidth(self.widget_17.sizePolicy().hasHeightForWidth())
        self.widget_17.setSizePolicy(sizePolicy1)
        self.widget_17.setMinimumSize(QSize(700, 0))
        self.horizontalLayout_26 = QHBoxLayout(self.widget_17)
        self.horizontalLayout_26.setSpacing(0)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.horizontalLayout_26.setContentsMargins(0, 0, 0, 0)
        self.report_table = QTableWidget(self.widget_17)
        self.report_table.setObjectName(u"report_table")

        self.horizontalLayout_26.addWidget(self.report_table)


        self.verticalLayout_23.addWidget(self.widget_17)

        self.stackedWidget.addWidget(self.report_page)
        self.about_page = QWidget()
        self.about_page.setObjectName(u"about_page")
        font10 = QFont()
        font10.setFamilies([u"Ubuntu Mono"])
        font10.setStyleStrategy(QFont.PreferDefault)
        self.about_page.setFont(font10)
        self.verticalLayout_29 = QVBoxLayout(self.about_page)
        self.verticalLayout_29.setObjectName(u"verticalLayout_29")
        self.widget_20 = QWidget(self.about_page)
        self.widget_20.setObjectName(u"widget_20")
        self.verticalLayout_30 = QVBoxLayout(self.widget_20)
        self.verticalLayout_30.setObjectName(u"verticalLayout_30")
        self.about_browser = QTextBrowser(self.widget_20)
        self.about_browser.setObjectName(u"about_browser")

        self.verticalLayout_30.addWidget(self.about_browser)


        self.verticalLayout_29.addWidget(self.widget_20)

        self.stackedWidget.addWidget(self.about_page)
        self.account_page = QWidget()
        self.account_page.setObjectName(u"account_page")
        self.verticalLayout_37 = QVBoxLayout(self.account_page)
        self.verticalLayout_37.setSpacing(0)
        self.verticalLayout_37.setObjectName(u"verticalLayout_37")
        self.verticalLayout_37.setContentsMargins(-1, 0, 0, 0)
        self.widget_24 = QWidget(self.account_page)
        self.widget_24.setObjectName(u"widget_24")
        self.verticalLayout_38 = QVBoxLayout(self.widget_24)
        self.verticalLayout_38.setSpacing(0)
        self.verticalLayout_38.setObjectName(u"verticalLayout_38")
        self.verticalLayout_38.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_37.addWidget(self.widget_24)

        self.stackedWidget.addWidget(self.account_page)
        self.help_page = QWidget()
        self.help_page.setObjectName(u"help_page")
        self.verticalLayout_26 = QVBoxLayout(self.help_page)
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.widget_18 = QWidget(self.help_page)
        self.widget_18.setObjectName(u"widget_18")
        self.widget_18.setMinimumSize(QSize(600, 0))
        self.verticalLayout_35 = QVBoxLayout(self.widget_18)
        self.verticalLayout_35.setObjectName(u"verticalLayout_35")
        self.frame_68 = QFrame(self.widget_18)
        self.frame_68.setObjectName(u"frame_68")
        self.frame_68.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_68.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_65 = QHBoxLayout(self.frame_68)
        self.horizontalLayout_65.setSpacing(0)
        self.horizontalLayout_65.setObjectName(u"horizontalLayout_65")
        self.horizontalLayout_65.setContentsMargins(0, 0, 0, 0)
        self.label_25 = QLabel(self.frame_68)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setFont(font6)

        self.horizontalLayout_65.addWidget(self.label_25, 0, Qt.AlignmentFlag.AlignHCenter)


        self.verticalLayout_35.addWidget(self.frame_68)

        self.frame_59 = QFrame(self.widget_18)
        self.frame_59.setObjectName(u"frame_59")
        self.frame_59.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_59.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_59 = QHBoxLayout(self.frame_59)
        self.horizontalLayout_59.setSpacing(0)
        self.horizontalLayout_59.setObjectName(u"horizontalLayout_59")
        self.horizontalLayout_59.setContentsMargins(0, 0, 0, 0)
        self.frame_61 = QFrame(self.frame_59)
        self.frame_61.setObjectName(u"frame_61")
        self.frame_61.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_61.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_63 = QHBoxLayout(self.frame_61)
        self.horizontalLayout_63.setSpacing(0)
        self.horizontalLayout_63.setObjectName(u"horizontalLayout_63")
        self.horizontalLayout_63.setContentsMargins(0, 0, 0, 0)
        self.label_22 = QLabel(self.frame_61)
        self.label_22.setObjectName(u"label_22")

        self.horizontalLayout_63.addWidget(self.label_22, 0, Qt.AlignmentFlag.AlignHCenter)


        self.horizontalLayout_59.addWidget(self.frame_61)

        self.frame_62 = QFrame(self.frame_59)
        self.frame_62.setObjectName(u"frame_62")
        sizePolicy4.setHeightForWidth(self.frame_62.sizePolicy().hasHeightForWidth())
        self.frame_62.setSizePolicy(sizePolicy4)
        self.frame_62.setMinimumSize(QSize(0, 0))
        self.frame_62.setMaximumSize(QSize(300, 16777215))
        self.frame_62.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_62.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_14 = QVBoxLayout(self.frame_62)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.verticalLayout_14.addItem(self.verticalSpacer_2)

        self.textBrowser = QTextBrowser(self.frame_62)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setMaximumSize(QSize(16777215, 50))

        self.verticalLayout_14.addWidget(self.textBrowser)


        self.horizontalLayout_59.addWidget(self.frame_62)


        self.verticalLayout_35.addWidget(self.frame_59)

        self.frame_60 = QFrame(self.widget_18)
        self.frame_60.setObjectName(u"frame_60")
        self.frame_60.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_60.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_60 = QHBoxLayout(self.frame_60)
        self.horizontalLayout_60.setSpacing(0)
        self.horizontalLayout_60.setObjectName(u"horizontalLayout_60")
        self.horizontalLayout_60.setContentsMargins(0, 0, 0, 0)
        self.frame_63 = QFrame(self.frame_60)
        self.frame_63.setObjectName(u"frame_63")
        self.frame_63.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_63.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_62 = QHBoxLayout(self.frame_63)
        self.horizontalLayout_62.setSpacing(0)
        self.horizontalLayout_62.setObjectName(u"horizontalLayout_62")
        self.horizontalLayout_62.setContentsMargins(0, 0, 0, 0)
        self.label_23 = QLabel(self.frame_63)
        self.label_23.setObjectName(u"label_23")

        self.horizontalLayout_62.addWidget(self.label_23, 0, Qt.AlignmentFlag.AlignHCenter)


        self.horizontalLayout_60.addWidget(self.frame_63)

        self.frame_64 = QFrame(self.frame_60)
        self.frame_64.setObjectName(u"frame_64")
        self.frame_64.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_64.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_67 = QHBoxLayout(self.frame_64)
        self.horizontalLayout_67.setObjectName(u"horizontalLayout_67")
        self.download_btn = QPushButton(self.frame_64)
        self.download_btn.setObjectName(u"download_btn")
        self.download_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_67.addWidget(self.download_btn)


        self.horizontalLayout_60.addWidget(self.frame_64)


        self.verticalLayout_35.addWidget(self.frame_60)

        self.frame_65 = QFrame(self.widget_18)
        self.frame_65.setObjectName(u"frame_65")
        self.frame_65.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_65.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_61 = QHBoxLayout(self.frame_65)
        self.horizontalLayout_61.setSpacing(0)
        self.horizontalLayout_61.setObjectName(u"horizontalLayout_61")
        self.horizontalLayout_61.setContentsMargins(0, 0, 0, 0)
        self.frame_66 = QFrame(self.frame_65)
        self.frame_66.setObjectName(u"frame_66")
        self.frame_66.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_66.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_64 = QHBoxLayout(self.frame_66)
        self.horizontalLayout_64.setSpacing(0)
        self.horizontalLayout_64.setObjectName(u"horizontalLayout_64")
        self.horizontalLayout_64.setContentsMargins(0, 0, 0, 0)
        self.label_24 = QLabel(self.frame_66)
        self.label_24.setObjectName(u"label_24")

        self.horizontalLayout_64.addWidget(self.label_24, 0, Qt.AlignmentFlag.AlignHCenter)


        self.horizontalLayout_61.addWidget(self.frame_66)

        self.frame_67 = QFrame(self.frame_65)
        self.frame_67.setObjectName(u"frame_67")
        self.frame_67.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_67.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_68 = QHBoxLayout(self.frame_67)
        self.horizontalLayout_68.setObjectName(u"horizontalLayout_68")
        self.logs_btn = QPushButton(self.frame_67)
        self.logs_btn.setObjectName(u"logs_btn")
        self.logs_btn.setEnabled(False)
        self.logs_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_68.addWidget(self.logs_btn)


        self.horizontalLayout_61.addWidget(self.frame_67)


        self.verticalLayout_35.addWidget(self.frame_65)


        self.verticalLayout_26.addWidget(self.widget_18, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.stackedWidget.addWidget(self.help_page)
        self.settings_page = QWidget()
        self.settings_page.setObjectName(u"settings_page")
        self.verticalLayout_31 = QVBoxLayout(self.settings_page)
        self.verticalLayout_31.setSpacing(0)
        self.verticalLayout_31.setObjectName(u"verticalLayout_31")
        self.verticalLayout_31.setContentsMargins(0, 0, 0, 9)
        self.widget_21 = QWidget(self.settings_page)
        self.widget_21.setObjectName(u"widget_21")
        sizePolicy1.setHeightForWidth(self.widget_21.sizePolicy().hasHeightForWidth())
        self.widget_21.setSizePolicy(sizePolicy1)
        self.widget_21.setMinimumSize(QSize(650, 0))
        self.verticalLayout_32 = QVBoxLayout(self.widget_21)
        self.verticalLayout_32.setObjectName(u"verticalLayout_32")
        self.verticalLayout_32.setContentsMargins(-1, 0, 9, 30)
        self.frame_28 = QFrame(self.widget_21)
        self.frame_28.setObjectName(u"frame_28")
        self.frame_28.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_28.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_29 = QHBoxLayout(self.frame_28)
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.label_3 = QLabel(self.frame_28)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font6)

        self.horizontalLayout_29.addWidget(self.label_3, 0, Qt.AlignmentFlag.AlignHCenter)


        self.verticalLayout_32.addWidget(self.frame_28)

        self.frame_29 = QFrame(self.widget_21)
        self.frame_29.setObjectName(u"frame_29")
        self.frame_29.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_29.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_30 = QHBoxLayout(self.frame_29)
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.frame_34 = QFrame(self.frame_29)
        self.frame_34.setObjectName(u"frame_34")
        self.frame_34.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_34.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_31 = QHBoxLayout(self.frame_34)
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.label_4 = QLabel(self.frame_34)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_31.addWidget(self.label_4, 0, Qt.AlignmentFlag.AlignHCenter)


        self.horizontalLayout_30.addWidget(self.frame_34)

        self.frame_35 = QFrame(self.frame_29)
        self.frame_35.setObjectName(u"frame_35")
        self.frame_35.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_35.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_32 = QHBoxLayout(self.frame_35)
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.label_5 = QLabel(self.frame_35)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_32.addWidget(self.label_5, 0, Qt.AlignmentFlag.AlignHCenter)


        self.horizontalLayout_30.addWidget(self.frame_35)


        self.verticalLayout_32.addWidget(self.frame_29)

        self.frame_30 = QFrame(self.widget_21)
        self.frame_30.setObjectName(u"frame_30")
        self.frame_30.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_30.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_39 = QHBoxLayout(self.frame_30)
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.frame_36 = QFrame(self.frame_30)
        self.frame_36.setObjectName(u"frame_36")
        self.frame_36.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_36.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_40 = QHBoxLayout(self.frame_36)
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.language_lbl = QLabel(self.frame_36)
        self.language_lbl.setObjectName(u"language_lbl")

        self.horizontalLayout_40.addWidget(self.language_lbl)


        self.horizontalLayout_39.addWidget(self.frame_36)

        self.frame_39 = QFrame(self.frame_30)
        self.frame_39.setObjectName(u"frame_39")
        self.frame_39.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_39.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_45 = QHBoxLayout(self.frame_39)
        self.horizontalLayout_45.setObjectName(u"horizontalLayout_45")
        self.comboBox = QComboBox(self.frame_39)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setEnabled(False)

        self.horizontalLayout_45.addWidget(self.comboBox)


        self.horizontalLayout_39.addWidget(self.frame_39)


        self.verticalLayout_32.addWidget(self.frame_30)

        self.frame_31 = QFrame(self.widget_21)
        self.frame_31.setObjectName(u"frame_31")
        self.frame_31.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_31.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_38 = QHBoxLayout(self.frame_31)
        self.horizontalLayout_38.setObjectName(u"horizontalLayout_38")
        self.frame_40 = QFrame(self.frame_31)
        self.frame_40.setObjectName(u"frame_40")
        self.frame_40.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_40.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_41 = QHBoxLayout(self.frame_40)
        self.horizontalLayout_41.setObjectName(u"horizontalLayout_41")
        self.theme_lbl = QLabel(self.frame_40)
        self.theme_lbl.setObjectName(u"theme_lbl")

        self.horizontalLayout_41.addWidget(self.theme_lbl)


        self.horizontalLayout_38.addWidget(self.frame_40)

        self.frame_41 = QFrame(self.frame_31)
        self.frame_41.setObjectName(u"frame_41")
        self.frame_41.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_41.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_44 = QHBoxLayout(self.frame_41)
        self.horizontalLayout_44.setObjectName(u"horizontalLayout_44")
        self.radioButton = QRadioButton(self.frame_41)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setEnabled(False)

        self.horizontalLayout_44.addWidget(self.radioButton)

        self.radioButton_2 = QRadioButton(self.frame_41)
        self.radioButton_2.setObjectName(u"radioButton_2")
        self.radioButton_2.setChecked(True)

        self.horizontalLayout_44.addWidget(self.radioButton_2)

        self.radioButton_3 = QRadioButton(self.frame_41)
        self.radioButton_3.setObjectName(u"radioButton_3")
        self.radioButton_3.setEnabled(False)

        self.horizontalLayout_44.addWidget(self.radioButton_3)


        self.horizontalLayout_38.addWidget(self.frame_41)


        self.verticalLayout_32.addWidget(self.frame_31)

        self.frame_32 = QFrame(self.widget_21)
        self.frame_32.setObjectName(u"frame_32")
        self.frame_32.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_32.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_37 = QHBoxLayout(self.frame_32)
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.frame_42 = QFrame(self.frame_32)
        self.frame_42.setObjectName(u"frame_42")
        self.frame_42.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_42.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_42 = QHBoxLayout(self.frame_42)
        self.horizontalLayout_42.setObjectName(u"horizontalLayout_42")
        self.font_lbl = QLabel(self.frame_42)
        self.font_lbl.setObjectName(u"font_lbl")

        self.horizontalLayout_42.addWidget(self.font_lbl)


        self.horizontalLayout_37.addWidget(self.frame_42)

        self.frame_43 = QFrame(self.frame_32)
        self.frame_43.setObjectName(u"frame_43")
        self.frame_43.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_43.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_46 = QHBoxLayout(self.frame_43)
        self.horizontalLayout_46.setObjectName(u"horizontalLayout_46")
        self.fontComboBox = QFontComboBox(self.frame_43)
        self.fontComboBox.setObjectName(u"fontComboBox")
        self.fontComboBox.setEnabled(False)

        self.horizontalLayout_46.addWidget(self.fontComboBox)


        self.horizontalLayout_37.addWidget(self.frame_43)


        self.verticalLayout_32.addWidget(self.frame_32)

        self.frame_33 = QFrame(self.widget_21)
        self.frame_33.setObjectName(u"frame_33")
        self.frame_33.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_33.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_36 = QHBoxLayout(self.frame_33)
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.frame_44 = QFrame(self.frame_33)
        self.frame_44.setObjectName(u"frame_44")
        self.frame_44.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_44.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_43 = QHBoxLayout(self.frame_44)
        self.horizontalLayout_43.setObjectName(u"horizontalLayout_43")
        self.graphs_lbl = QLabel(self.frame_44)
        self.graphs_lbl.setObjectName(u"graphs_lbl")

        self.horizontalLayout_43.addWidget(self.graphs_lbl)


        self.horizontalLayout_36.addWidget(self.frame_44)

        self.frame_45 = QFrame(self.frame_33)
        self.frame_45.setObjectName(u"frame_45")
        sizePolicy1.setHeightForWidth(self.frame_45.sizePolicy().hasHeightForWidth())
        self.frame_45.setSizePolicy(sizePolicy1)
        self.frame_45.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_45.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_34 = QVBoxLayout(self.frame_45)
        self.verticalLayout_34.setSpacing(0)
        self.verticalLayout_34.setObjectName(u"verticalLayout_34")
        self.verticalLayout_34.setContentsMargins(0, 0, 0, 0)
        self.frame_51 = QFrame(self.frame_45)
        self.frame_51.setObjectName(u"frame_51")
        sizePolicy1.setHeightForWidth(self.frame_51.sizePolicy().hasHeightForWidth())
        self.frame_51.setSizePolicy(sizePolicy1)
        self.frame_51.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_51.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_51 = QHBoxLayout(self.frame_51)
        self.horizontalLayout_51.setSpacing(0)
        self.horizontalLayout_51.setObjectName(u"horizontalLayout_51")
        self.horizontalLayout_51.setContentsMargins(0, 0, 0, 0)
        self.frame_53 = QFrame(self.frame_51)
        self.frame_53.setObjectName(u"frame_53")
        sizePolicy1.setHeightForWidth(self.frame_53.sizePolicy().hasHeightForWidth())
        self.frame_53.setSizePolicy(sizePolicy1)
        self.frame_53.setMinimumSize(QSize(0, 10))
        self.frame_53.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_53.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_52 = QHBoxLayout(self.frame_53)
        self.horizontalLayout_52.setSpacing(0)
        self.horizontalLayout_52.setObjectName(u"horizontalLayout_52")
        self.horizontalLayout_52.setContentsMargins(0, 0, 0, 0)
        self.label_20 = QLabel(self.frame_53)
        self.label_20.setObjectName(u"label_20")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy6)
        self.label_20.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_52.addWidget(self.label_20, 0, Qt.AlignmentFlag.AlignRight)


        self.horizontalLayout_51.addWidget(self.frame_53)

        self.frame_54 = QFrame(self.frame_51)
        self.frame_54.setObjectName(u"frame_54")
        self.frame_54.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_54.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_53 = QHBoxLayout(self.frame_54)
        self.horizontalLayout_53.setSpacing(0)
        self.horizontalLayout_53.setObjectName(u"horizontalLayout_53")
        self.horizontalLayout_53.setContentsMargins(0, 0, -1, 0)
        self.comboBox_3 = QComboBox(self.frame_54)
        self.comboBox_3.setObjectName(u"comboBox_3")
        self.comboBox_3.setEnabled(False)

        self.horizontalLayout_53.addWidget(self.comboBox_3)


        self.horizontalLayout_51.addWidget(self.frame_54)


        self.verticalLayout_34.addWidget(self.frame_51)

        self.frame_52 = QFrame(self.frame_45)
        self.frame_52.setObjectName(u"frame_52")
        self.frame_52.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_52.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_54 = QHBoxLayout(self.frame_52)
        self.horizontalLayout_54.setSpacing(0)
        self.horizontalLayout_54.setObjectName(u"horizontalLayout_54")
        self.horizontalLayout_54.setContentsMargins(0, 0, -1, 0)
        self.frame_55 = QFrame(self.frame_52)
        self.frame_55.setObjectName(u"frame_55")
        self.frame_55.setMinimumSize(QSize(0, 0))
        self.frame_55.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_55.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_55 = QHBoxLayout(self.frame_55)
        self.horizontalLayout_55.setSpacing(0)
        self.horizontalLayout_55.setObjectName(u"horizontalLayout_55")
        self.horizontalLayout_55.setContentsMargins(0, 0, -1, 0)
        self.label_21 = QLabel(self.frame_55)
        self.label_21.setObjectName(u"label_21")

        self.horizontalLayout_55.addWidget(self.label_21, 0, Qt.AlignmentFlag.AlignRight)


        self.horizontalLayout_54.addWidget(self.frame_55)

        self.frame_56 = QFrame(self.frame_52)
        self.frame_56.setObjectName(u"frame_56")
        self.frame_56.setMinimumSize(QSize(0, 0))
        self.frame_56.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_56.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_56 = QHBoxLayout(self.frame_56)
        self.horizontalLayout_56.setSpacing(0)
        self.horizontalLayout_56.setObjectName(u"horizontalLayout_56")
        self.horizontalLayout_56.setContentsMargins(0, 0, 0, 0)
        self.comboBox_4 = QComboBox(self.frame_56)
        self.comboBox_4.setObjectName(u"comboBox_4")
        self.comboBox_4.setEnabled(False)

        self.horizontalLayout_56.addWidget(self.comboBox_4)


        self.horizontalLayout_54.addWidget(self.frame_56)


        self.verticalLayout_34.addWidget(self.frame_52)


        self.horizontalLayout_36.addWidget(self.frame_45)


        self.verticalLayout_32.addWidget(self.frame_33)

        self.frame_46 = QFrame(self.widget_21)
        self.frame_46.setObjectName(u"frame_46")
        self.frame_46.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_46.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_47 = QHBoxLayout(self.frame_46)
        self.horizontalLayout_47.setObjectName(u"horizontalLayout_47")
        self.frame_47 = QFrame(self.frame_46)
        self.frame_47.setObjectName(u"frame_47")
        self.frame_47.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_47.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_48 = QHBoxLayout(self.frame_47)
        self.horizontalLayout_48.setObjectName(u"horizontalLayout_48")
        self.export_lbl = QLabel(self.frame_47)
        self.export_lbl.setObjectName(u"export_lbl")

        self.horizontalLayout_48.addWidget(self.export_lbl)


        self.horizontalLayout_47.addWidget(self.frame_47)

        self.frame_48 = QFrame(self.frame_46)
        self.frame_48.setObjectName(u"frame_48")
        self.frame_48.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_48.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_33 = QVBoxLayout(self.frame_48)
        self.verticalLayout_33.setObjectName(u"verticalLayout_33")
        self.frame_50 = QFrame(self.frame_48)
        self.frame_50.setObjectName(u"frame_50")
        self.frame_50.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_50.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_49 = QHBoxLayout(self.frame_50)
        self.horizontalLayout_49.setObjectName(u"horizontalLayout_49")
        self.label_18 = QLabel(self.frame_50)
        self.label_18.setObjectName(u"label_18")

        self.horizontalLayout_49.addWidget(self.label_18, 0, Qt.AlignmentFlag.AlignRight)

        self.comboBox_2 = QComboBox(self.frame_50)
        self.comboBox_2.setObjectName(u"comboBox_2")
        self.comboBox_2.setEnabled(False)

        self.horizontalLayout_49.addWidget(self.comboBox_2)


        self.verticalLayout_33.addWidget(self.frame_50)

        self.frame_49 = QFrame(self.frame_48)
        self.frame_49.setObjectName(u"frame_49")
        self.frame_49.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_49.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_50 = QHBoxLayout(self.frame_49)
        self.horizontalLayout_50.setObjectName(u"horizontalLayout_50")
        self.label_19 = QLabel(self.frame_49)
        self.label_19.setObjectName(u"label_19")

        self.horizontalLayout_50.addWidget(self.label_19)

        self.edil_path = QLineEdit(self.frame_49)
        self.edil_path.setObjectName(u"edil_path")
        self.edil_path.setEnabled(False)

        self.horizontalLayout_50.addWidget(self.edil_path)

        self.save_path_btn = QPushButton(self.frame_49)
        self.save_path_btn.setObjectName(u"save_path_btn")
        self.save_path_btn.setEnabled(False)

        self.horizontalLayout_50.addWidget(self.save_path_btn)


        self.verticalLayout_33.addWidget(self.frame_49)


        self.horizontalLayout_47.addWidget(self.frame_48)


        self.verticalLayout_32.addWidget(self.frame_46)

        self.frame_69 = QFrame(self.widget_21)
        self.frame_69.setObjectName(u"frame_69")
        self.frame_69.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_69.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_69 = QHBoxLayout(self.frame_69)
        self.horizontalLayout_69.setSpacing(0)
        self.horizontalLayout_69.setObjectName(u"horizontalLayout_69")
        self.horizontalLayout_69.setContentsMargins(0, 0, 0, 0)
        self.frame_70 = QFrame(self.frame_69)
        self.frame_70.setObjectName(u"frame_70")
        self.frame_70.setMinimumSize(QSize(0, 10))
        self.frame_70.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_70.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_70 = QHBoxLayout(self.frame_70)
        self.horizontalLayout_70.setSpacing(0)
        self.horizontalLayout_70.setObjectName(u"horizontalLayout_70")
        self.horizontalLayout_70.setContentsMargins(0, 0, 0, 0)
        self.choose_alg_lbl = QLabel(self.frame_70)
        self.choose_alg_lbl.setObjectName(u"choose_alg_lbl")

        self.horizontalLayout_70.addWidget(self.choose_alg_lbl)


        self.horizontalLayout_69.addWidget(self.frame_70)

        self.frame_71 = QFrame(self.frame_69)
        self.frame_71.setObjectName(u"frame_71")
        self.frame_71.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_71.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_71 = QHBoxLayout(self.frame_71)
        self.horizontalLayout_71.setSpacing(0)
        self.horizontalLayout_71.setObjectName(u"horizontalLayout_71")
        self.horizontalLayout_71.setContentsMargins(0, 0, 0, 0)
        self.alg_box = QComboBox(self.frame_71)
        self.alg_box.setObjectName(u"alg_box")

        self.horizontalLayout_71.addWidget(self.alg_box)


        self.horizontalLayout_69.addWidget(self.frame_71)


        self.verticalLayout_32.addWidget(self.frame_69)


        self.verticalLayout_31.addWidget(self.widget_21, 0, Qt.AlignmentFlag.AlignHCenter)

        self.widget_22 = QWidget(self.settings_page)
        self.widget_22.setObjectName(u"widget_22")
        sizePolicy5.setHeightForWidth(self.widget_22.sizePolicy().hasHeightForWidth())
        self.widget_22.setSizePolicy(sizePolicy5)
        self.widget_22.setMinimumSize(QSize(650, 0))
        self.horizontalLayout_28 = QHBoxLayout(self.widget_22)
        self.horizontalLayout_28.setSpacing(0)
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.horizontalLayout_28.setContentsMargins(10, 0, 0, 0)
        self.frame_58 = QFrame(self.widget_22)
        self.frame_58.setObjectName(u"frame_58")
        self.frame_58.setMinimumSize(QSize(0, 0))
        self.frame_58.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_58.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_58 = QHBoxLayout(self.frame_58)
        self.horizontalLayout_58.setSpacing(0)
        self.horizontalLayout_58.setObjectName(u"horizontalLayout_58")
        self.horizontalLayout_58.setContentsMargins(0, 0, 0, 0)
        self.advanced_btn = QPushButton(self.frame_58)
        self.advanced_btn.setObjectName(u"advanced_btn")
        self.advanced_btn.setEnabled(False)
        self.advanced_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_58.addWidget(self.advanced_btn)


        self.horizontalLayout_28.addWidget(self.frame_58, 0, Qt.AlignmentFlag.AlignLeft)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_28.addItem(self.horizontalSpacer)

        self.frame_57 = QFrame(self.widget_22)
        self.frame_57.setObjectName(u"frame_57")
        sizePolicy4.setHeightForWidth(self.frame_57.sizePolicy().hasHeightForWidth())
        self.frame_57.setSizePolicy(sizePolicy4)
        self.frame_57.setMinimumSize(QSize(0, 0))
        self.frame_57.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_57.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_57 = QHBoxLayout(self.frame_57)
        self.horizontalLayout_57.setSpacing(0)
        self.horizontalLayout_57.setObjectName(u"horizontalLayout_57")
        self.horizontalLayout_57.setContentsMargins(10, 0, 0, 0)
        self.cancel_btn = QPushButton(self.frame_57)
        self.cancel_btn.setObjectName(u"cancel_btn")
        self.cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_57.addWidget(self.cancel_btn)

        self.horizontalSpacer_2 = QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_57.addItem(self.horizontalSpacer_2)

        self.save_btn = QPushButton(self.frame_57)
        self.save_btn.setObjectName(u"save_btn")
        self.save_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_57.addWidget(self.save_btn)


        self.horizontalLayout_28.addWidget(self.frame_57)


        self.verticalLayout_31.addWidget(self.widget_22, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignBottom)

        self.stackedWidget.addWidget(self.settings_page)

        self.verticalLayout_2.addWidget(self.stackedWidget, 0, Qt.AlignmentFlag.AlignTop)


        self.horizontalLayout.addWidget(self.main_content)

        self.right_menu = QWidget(self.main_body)
        self.right_menu.setObjectName(u"right_menu")
        sizePolicy1.setHeightForWidth(self.right_menu.sizePolicy().hasHeightForWidth())
        self.right_menu.setSizePolicy(sizePolicy1)
        self.right_menu.setMinimumSize(QSize(50, 0))
        self.right_menu.setMaximumSize(QSize(150, 16777215))
        self.verticalLayout_7 = QVBoxLayout(self.right_menu)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.widget = QWidget(self.right_menu)
        self.widget.setObjectName(u"widget")
        self.verticalLayout_8 = QVBoxLayout(self.widget)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, -1)
        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(70, 70))
        self.label_2.setMaximumSize(QSize(70, 70))
        self.label_2.setPixmap(QPixmap(u"../../../../.designer/icons/user.png"))
        self.label_2.setScaledContents(True)

        self.verticalLayout_8.addWidget(self.label_2, 0, Qt.AlignmentFlag.AlignHCenter)

        self.frame_5 = QFrame(self.widget)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.frame_5)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.userName = QLineEdit(self.frame_5)
        self.userName.setObjectName(u"userName")

        self.verticalLayout_9.addWidget(self.userName)

        self.email = QLineEdit(self.frame_5)
        self.email.setObjectName(u"email")

        self.verticalLayout_9.addWidget(self.email)

        self.phoneNo = QLineEdit(self.frame_5)
        self.phoneNo.setObjectName(u"phoneNo")

        self.verticalLayout_9.addWidget(self.phoneNo)


        self.verticalLayout_8.addWidget(self.frame_5)

        self.add_user_btn = QPushButton(self.widget)
        self.add_user_btn.setObjectName(u"add_user_btn")
        font11 = QFont()
        font11.setBold(False)
        font11.setItalic(False)
        self.add_user_btn.setFont(font11)
        self.add_user_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.verticalLayout_8.addWidget(self.add_user_btn, 0, Qt.AlignmentFlag.AlignHCenter)


        self.verticalLayout_7.addWidget(self.widget, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)


        self.horizontalLayout.addWidget(self.right_menu)


        self.verticalLayout.addWidget(self.main_body)

        self.footer = QWidget(self.centralwidget)
        self.footer.setObjectName(u"footer")
        self.horizontalLayout_5 = QHBoxLayout(self.footer)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_9 = QLabel(self.footer)
        self.label_9.setObjectName(u"label_9")
        font12 = QFont()
        font12.setPointSize(9)
        self.label_9.setFont(font12)

        self.horizontalLayout_5.addWidget(self.label_9, 0, Qt.AlignmentFlag.AlignHCenter)


        self.verticalLayout.addWidget(self.footer)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.menu_btn.setText(QCoreApplication.translate("MainWindow", u"Menu", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"         Stationary App", None))
        self.notifications.setText("")
        self.signin.setText(QCoreApplication.translate("MainWindow", u"Sign in ", None))
        self.home_btn.setText(QCoreApplication.translate("MainWindow", u"Home", None))
        self.algorithm_btn.setText(QCoreApplication.translate("MainWindow", u"Algorithm", None))
        self.instruction_btn.setText(QCoreApplication.translate("MainWindow", u"Instruction", None))
        self.reports_btn.setText(QCoreApplication.translate("MainWindow", u"Reports", None))
        self.account_btn.setText(QCoreApplication.translate("MainWindow", u"My account", None))
        self.help_btn.setText(QCoreApplication.translate("MainWindow", u"Help", None))
        self.settings_btn.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.about_btn.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.textEdit.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">\u041d\u0430\u0447\u0430\u043b\u043e \u0440\u0430\u0431\u043e\u0442\u044b</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">	</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px"
                        ";\">	<span style=\" font-size:12pt; font-style:italic;\">\u041f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0430 \u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432\u043b\u044f\u0435\u0442 \u0443\u0434\u043e\u0431\u043d\u044b\u0439 \u0433\u0440\u0430\u0444\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441 \u0434\u043b\u044f \u0432\u0432\u0430\u0434 \u0434\u0430\u043d\u043d\u044b\u0445 \u0438 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442 \u0430\u043b\u0433\u043e\u0440\u0442\u0438\u043c \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u0438 \u0434\u043b\u044f \u0438\u0445 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0438.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">	</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span sty"
                        "le=\" font-size:12pt;\">	</span><span style=\" font-size:12pt; font-weight:700;\">\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0444\u0443\u043d\u043a\u0446\u0438\u043e\u043d\u0430\u043b:</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> - </span><span style=\" font-size:12pt; text-decoration: underline;\">Home page</span><span style=\" font-size:12pt;\"> (\u041d\u0430\u0447\u0430\u043b\u044c\u043d\u0430\u044f \u0442\u0441\u0440\u0430\u043d\u0438\u0446\u0430)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> - </span><span style=\" font-size:12pt; text-decoration: underline;\">Algorithm page</spa"
                        "n><span style=\" font-size:12pt;\"> (\u0421\u0442\u0440\u0430\u043d\u0438\u0446\u0430 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0438 \u0432\u0430\u0448\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> - </span><span style=\" font-size:12pt; text-decoration: underline;\">Instruction</span><span style=\" font-size:12pt;\"> (\u0418\u043d\u0442\u0441\u0440\u0443\u043a\u0446\u0438\u044f \u043f\u043e \u0440\u0430\u0431\u043e\u0442\u0435 \u0441 \u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c\u043e\u043c)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> - </span><span style=\" font-size:12pt; text-decoration: underline;\">Reports</span><span style=\" font-size:12pt;\"> (\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u0434\u0430"
                        "\u043d\u043d\u044b\u0445 \u0438 \u043f\u0435\u0447\u0430\u0442\u044c)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> - </span><span style=\" font-size:12pt; text-decoration: underline;\">My account</span><span style=\" font-size:12pt;\"> </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> - </span><span style=\" font-size:12pt; text-decoration: underline;\">Help page</span><span style=\" font-size:12pt;\"> (\u0422\u0435\u0445\u043d\u0438\u0447\u0435\u0441\u043a\u0430\u044f \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0430)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> - </span><span style=\" font-size:12pt; text-decoration: underline;\">Settings "
                        "page</span><span style=\" font-size:12pt;\"> (\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u0430)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> - </span><span style=\" font-size:12pt; text-decoration: underline;\">About page</span><span style=\" font-size:12pt;\"> (\u0418\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u044f \u043e \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0447\u0438\u043a\u0435, \u0432\u0435\u0440\u0441\u0438\u0438 \u041f\u041e, \u043b\u0438\u0446\u0435\u043d\u0437\u0438\u044f\u0445 \u0438 \u0431\u043b\u0430\u0433\u043e\u0434\u0430\u0440\u043d\u043e\u0441\u0442\u044f\u0445)</span></p></body></html>", None))
        self.next_page_1.setText(QCoreApplication.translate("MainWindow", u"NEXT PAGE", None))
        self.data_label.setText(QCoreApplication.translate("MainWindow", u"\u0418\u0441\u0445\u043e\u0434\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435", None))
        self.load_btn.setText(QCoreApplication.translate("MainWindow", u"Pick File", None))
        self.name_file_label.setText("")
        self.correlation_btn.setText(QCoreApplication.translate("MainWindow", u"correlation", None))
        self.box_x.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u041e\u0441\u044c X", None))
        self.box_y.setPlaceholderText(QCoreApplication.translate("MainWindow", u"\u041e\u0441\u044c Y", None))
        self.data_graph_btn.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"\u041d\u0430\u0439\u0442\u0440\u043e\u043a\u0430 \u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c\u0430 \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u0438", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"\u0413\u0440\u0430\u043d\u0438\u0447\u043d\u043e\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435", None))
        self.gran_val.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Auto", None))
        self.x_min.setPlaceholderText(QCoreApplication.translate("MainWindow", u"x_min", None))
        self.x_max.setPlaceholderText(QCoreApplication.translate("MainWindow", u"x_max", None))
        self.optimize.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0442\u043e\u0431\u0440\u0430\u0437\u0438\u0442\u044c \u0442\u0430\u0431\u043b\u0438\u0446\u0443 \u043a\u043b\u0441\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0439", None))
        self.graph_opt.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a ", None))
        self.add_stationar.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c", None))
        self.refresh_talbe.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0431\u043d\u0443\u043b\u0438\u0442\u044c", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"\u0422\u0430\u0431\u043b\u0438\u0446\u0430 \u043a\u043b\u0430\u0441\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0438 ", None))
        self.search_k.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0438\u0441\u043a \u043a\u043e\u044d\u0444\u0444\u0438\u0446\u0438\u0435\u043d\u0442\u043e\u0432", None))
        self.k_1.setText(QCoreApplication.translate("MainWindow", u"K_1 = ?", None))
        self.k_2.setText(QCoreApplication.translate("MainWindow", u"K_2 = ?", None))
        self.k_3.setText(QCoreApplication.translate("MainWindow", u"K_3 = ?", None))
        self.F_score.setText(QCoreApplication.translate("MainWindow", u"F1-score =?", None))
        self.graph_result.setText(QCoreApplication.translate("MainWindow", u"\u041e\u043f\u0442\u0438\u043c\u0438\u0437\u0438\u0440\u043e\u0432\u0430\u0442\u044c", None))
        self.text_edit_instruction.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\" bgcolor=\"transparent\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:14pt; color:#ffffff;\">\u0418\u043d\u0442\u0441\u0440\u0443\u043a\u0446\u0438\u044f </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; font-weight:70"
                        "0; color:#ffffff;\">\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0434\u0430\u043d\u043d\u044b\u0445:</span><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\"> </span></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u0435 CSV-\u0444\u0430\u0439\u043b \u0441 \u0434\u0430\u043d\u043d\u044b\u043c\u0438 \u0447\u0435\u0440\u0435\u0437 \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441 \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u044b. </li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; font-weight:700; colo"
                        "r:#ffffff;\">\u041e\u0442\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u0434\u0430\u043d\u043d\u044b\u0445:</span><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\"> </span></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u041f\u043e\u0441\u043b\u0435 \u0437\u0430\u0433\u0440\u0443\u0437\u043a\u0438 \u0444\u0430\u0439\u043b\u0430 \u0434\u0430\u043d\u043d\u044b\u0435 \u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u043e\u0442\u043e\u0431\u0440\u0430\u0437\u044f\u0442\u0441\u044f \u0432 \u0432\u0438\u0434\u0435 \u0442\u0430\u0431\u043b\u0438\u0446\u044b \u0432 \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u0435. </li></ul>\n"
"<p style=\" margin-top:12px; margin-"
                        "bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; font-weight:700; color:#ffffff;\">\u0412\u044b\u0431\u043e\u0440 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432:</span><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\"> </span></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u0412 \u0432\u044b\u043f\u0430\u0434\u0430\u044e\u0449\u0438\u0445 \u0441\u043f\u0438\u0441\u043a\u0430\u0445 \u0432\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0434\u043b\u044f \u0430\u043d\u0430\u043b\u0438"
                        "\u0437\u0430. </li>\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u041f\u043e\u0441\u0442\u0440\u043e\u0439\u0442\u0435 \u0433\u0440\u0430\u0444\u0438\u043a \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u044f \u0432\u044b\u0431\u0440\u0430\u043d\u043d\u043e\u0433\u043e \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u0430. </li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; font-weight:700; color:#ffffff;\">\u0410\u043d\u0430\u043b\u0438\u0437 \u0441\u0442\u0430\u0446\u0438\u043e\u043d\u0430\u0440\u043d\u044b\u0445 \u0440\u0435\u0436\u0438\u043c\u043e\u0432:</span><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\"> </span></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom"
                        ": 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u041f\u0435\u0440\u0435\u0439\u0434\u0438\u0442\u0435 \u043d\u0430 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0443\u044e \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0443 \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u0430. </li>\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u0423\u043a\u0430\u0436\u0438\u0442\u0435 \u043d\u0430\u0447\u0430\u043b\u043e (x_min) \u0438 \u043a\u043e\u043d\u0435\u0446 (x_max) \u0443\u0447\u0430\u0441\u0442\u043a\u0430 \u0434\u0430\u043d\u043d\u044b\u0445, \u043d\u0430 \u043a\u043e\u0442\u043e\u0440\u043e\u043c \u0431\u0443\u0434\u0435\u0442 \u043f\u0440"
                        "\u043e\u0432\u043e\u0434\u0438\u0442\u044c\u0441\u044f \u043e\u0446\u0435\u043d\u043a\u0430 \u0441\u0442\u0430\u0446\u0438\u043e\u043d\u0430\u0440\u043d\u044b\u0445 \u0440\u0435\u0436\u0438\u043c\u043e\u0432. \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f \u0432\u044b\u0431\u0438\u0440\u0430\u0442\u044c \u0434\u0438\u0430\u043f\u0430\u0437\u043e\u043d \u043e\u0442 3000 \u0434\u043e 10000 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0439. </li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; font-weight:700; color:#ffffff;\">\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b \u0430\u043d\u0430\u043b\u0438\u0437\u0430:</span><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\"> </span></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;"
                        "\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u041f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0430 \u0432\u044b\u0432\u0435\u0434\u0435\u0442 \u0442\u0430\u0431\u043b\u0438\u0446\u0443 \u0441 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0430\u043c\u0438: \n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 2;\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;stationary&quot; \u2014 \u043e\u0446\u0435\u043d\u043a\u0430 \u0441\u0442\u0430\u0446\u0438\u043e\u043d\u0430\u0440\u043d\u043e\u0441\u0442\u0438. </li>\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:0px; margin-botto"
                        "m:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&quot;assessment&quot; \u2014 \u0432\u0430\u0448\u0430 \u043b\u0438\u0447\u043d\u0430\u044f \u043e\u0446\u0435\u043d\u043a\u0430. </li></ul></li>\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u041f\u043e\u0441\u0442\u0440\u043e\u0439\u0442\u0435 \u0433\u0440\u0430\u0444\u0438\u043a, \u043d\u0430 \u043a\u043e\u0442\u043e\u0440\u043e\u043c \u0432\u044b \u0441\u043c\u043e\u0436\u0435\u0442\u0435 \u0432\u0440\u0443\u0447\u043d\u0443\u044e \u0432\u044b\u0434\u0435\u043b\u0438\u0442\u044c \u0441\u0442\u0430\u0446\u0438\u043e\u043d\u0430\u0440\u043d\u044b\u0435 \u0443\u0447\u0430\u0441\u0442\u043a\u0438. \u042d\u0442\u0438 \u0434\u0430\u043d\u043d\u044b\u0435 \u0442\u0430\u043a\u0436\u0435 \u0431\u0443\u0434\u0443\u0442 \u0437\u0430\u043f\u0438\u0441\u0430\u043d\u044b \u0432 \u0442\u0430"
                        "\u0431\u043b\u0438\u0446\u0443. </li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; font-weight:700; color:#ffffff;\">\u041e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u044f:</span><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\"> </span></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u043a\u043d\u043e\u043f\u043a\u0443 &quot;Optimize&quot;, \u0447\u0442\u043e\u0431\u044b \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0430 \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u043b\u0430 \u043e\u043f\u0442\u0438"
                        "\u043c\u0430\u043b\u044c\u043d\u044b\u0435 \u0432\u0435\u0441\u043e\u0432\u044b\u0435 \u043a\u043e\u044d\u0444\u0444\u0438\u0446\u0438\u0435\u043d\u0442\u044b. </li>\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u043c \u0431\u0443\u0434\u0435\u0442 \u043c\u0435\u0442\u0440\u0438\u043a\u0430 \u043a\u043b\u0430\u0441\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0438 F1-score. </li></ul>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:12pt; font-weight:700; color:#ffffff;\">\u0412\u0438\u0437\u0443\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f \u0438 \u044d\u043a\u0441\u043f\u043e\u0440\u0442 \u0434\u0430\u043d\u043d\u044b\u0445:</span><span style=\" font-fa"
                        "mily:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\"> </span></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u041f\u043e\u0441\u043b\u0435 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u0438\u044f \u0430\u043d\u0430\u043b\u0438\u0437\u0430 \u0432\u044b \u043c\u043e\u0436\u0435\u0442\u0435: \n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 2;\">\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u041f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0433\u0440\u0430\u0444\u0438\u043a \u0437\u0430 \u0432\u0435\u0441\u044c \u043f"
                        "\u0435\u0440\u0438\u043e\u0434, \u043d\u0430 \u043a\u043e\u0442\u043e\u0440\u043e\u043c \u0431\u0443\u0434\u0443\u0442 \u043e\u0442\u043c\u0435\u0447\u0435\u043d\u044b \u0441\u0442\u0430\u0446\u0438\u043e\u043d\u0430\u0440\u043d\u044b\u0435 \u0438 \u043d\u0435\u0441\u0442\u0430\u0446\u0438\u043e\u043d\u0430\u0440\u043d\u044b\u0435 \u0443\u0447\u0430\u0441\u0442\u043a\u0438. </li>\n"
"<li style=\" font-family:'Arial','sans-serif'; font-size:12pt; color:#ffffff;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\u0412\u044b\u0433\u0440\u0443\u0437\u0438\u0442\u044c \u0440\u0430\u0437\u043c\u0435\u0447\u0435\u043d\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 \u0432 \u0444\u0430\u0439\u043b \u0434\u043b\u044f \u0434\u0430\u043b\u044c\u043d\u0435\u0439\u0448\u0435\u0433\u043e \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u044f. </li></ul></li></ul></body></html>", None))
        self.report_label.setText(QCoreApplication.translate("MainWindow", u"Reports", None))
        self.print_btn.setText(QCoreApplication.translate("MainWindow", u"Print", None))
        self.export_pdf_btn.setText(QCoreApplication.translate("MainWindow", u"Export to pdf", None))
        self.export_csv_btn.setText(QCoreApplication.translate("MainWindow", u"Export to CSV", None))
        self.export_json_btn.setText(QCoreApplication.translate("MainWindow", u"Export to JSON", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"Export to Excel", None))
        self.about_browser.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\" bgcolor=\"transparent\">\n"
"<h1 style=\" margin-top:18px; margin-bottom:10px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; font-size:xx-large; font-weight:700; color:#ffffff;\">\u041f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u043d\u043e\u0435 \u043e\u0431\u0435\u0441\u043f\u0435\u0447\u0435\u043d\u0438\u0435 </span></h1>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; t"
                        "ext-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; font-weight:700; color:#ffffff;\">\u0412\u0435\u0440\u0441\u0438\u044f:</span><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\"> 1.0.0 </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; font-weight:700; color:#ffffff;\">\u0420\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u0447\u0438\u043a:</span><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\"> Mart-igor </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; font-weight:700; color:#ffffff;\">\u041b\u0438\u0446\u0435\u043d\u0437\u0438\u044f:</span><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\"> - </span></p>\n"
"<p style=\" margi"
                        "n-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; font-weight:700; color:#ffffff;\">\u0414\u0430\u0442\u0430 \u0441\u0431\u043e\u0440\u043a\u0438:</span><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\"> 2025 </span></p>\n"
"<h2 style=\" margin-top:15px; margin-bottom:5px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; font-size:x-large; font-weight:700; color:#ffffff;\">\u041e\u043f\u0438\u0441\u0430\u043d\u0438\u0435 </span></h2>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\">\u042d\u0442\u0430 \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0430 \u043f\u0440\u0435\u0434\u043d\u0430\u0437\u043d\u0430\u0447\u0435\u043d"
                        "\u0430 \u0434\u043b\u044f \u0430\u043d\u0430\u043b\u0438\u0437\u0430 \u0434\u0430\u043d\u043d\u044b\u0445, \u043e\u0446\u0435\u043d\u043a\u0438 \u0441\u0442\u0430\u0446\u0438\u043e\u043d\u0430\u0440\u043d\u044b\u0445 \u0440\u0435\u0436\u0438\u043c\u043e\u0432 \u0438 \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u0438 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432. </span></p>\n"
"<h2 style=\" margin-top:15px; margin-bottom:5px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; font-size:x-large; font-weight:700; color:#ffffff;\">\u041a\u043e\u043d\u0442\u0430\u043a\u0442\u044b </span></h2>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\">Email: </span><a href=\"mailto:your.email@example.com\"><span style=\" font-family:'Arial','sans-"
                        "serif'; text-decoration: underline; color:#4da6ff;\">my.email@example.com</span></a><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\">GitHub: </span><a href=\"https://github.com/Mart-igor\"><span style=\" font-family:'Arial','sans-serif'; text-decoration: underline; color:#4da6ff;\">github.com/Mart-igor</span></a><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\"> </span></p>\n"
"<h2 style=\" margin-top:15px; margin-bottom:5px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; font-size:x-large; font-weight:700; color:#ffffff;\">\u0411\u043b\u0430\u0433\u043e\u0434\u0430\u0440\u043d\u043e\u0441\u0442\u0438 </span></h2>\n"
"<p style=\" margin-top:12px; margin-bottom:15px;"
                        " margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\">    <span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\">        \u0421\u043f\u0430\u0441\u0438\u0431\u043e \u0437\u0430 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u0435 \u043d\u0430\u0448\u0435\u0433\u043e \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u043d\u043e\u0433\u043e \u043e\u0431\u0435\u0441\u043f\u0435\u0447\u0435\u043d\u0438\u044f! \u0412\u0430\u0448\u0430 \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0430 \u043e\u0447\u0435\u043d\u044c \u0432\u0430\u0436\u043d\u0430 \u0434\u043b\u044f \u043d\u0430\u0441.    </span></p>\n"
"<h3 style=\" margin-top:20px; margin-bottom:10px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Arial','sans-serif'; font-size:large; font-weight:600; color:#ffffff;\">    \u041f\u043e\u0434\u0434\u0435\u0440\u0436\u0430\u0442\u044c \u043f\u0440\u043e\u0435\u043a\u0442:</span></h3>\n"
""
                        "<p style=\" margin-top:0px; margin-bottom:30px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">    <span style=\" font-family:'monospace'; color:#ffffff; background-color:#27263c;\">\u0442\u0443\u0442 \u043c\u043e\u0439 \u043a\u0440\u0438\u043f\u0442\u043e \u043a\u043e\u0448\u0435\u043b\u0435\u043a))</span>    <span style=\" vertical-align:middle;\">        </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><span style=\" font-family:'Arial','sans-serif'; color:#ffffff;\">\u00a9 2025 \u0412\u0441\u0435 \u043f\u0440\u0430\u0432\u0430 \u0437\u0430\u0449\u0438\u0449\u0435\u043d\u044b.</span></p></body></html>", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Help page", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043a\u0446\u0438\u044f", None))
        self.textBrowser.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:150%;\"><a href=\"https://github.com/Mart-igor\"><span style=\" font-family:'Arial','sans-serif'; text-decoration: underline; color:#4da6ff;\">\u0421\u0441\u044b\u043b\u043a\u0430 \u043d\u0430 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0446\u0438\u044e</span></a><span style=\" font-family:'Arial','sans-serif'; text-decoration: underline; color:#ffffff;\"> </sp"
                        "an></p></body></html>", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u0438\u043c\u0435\u0440\u044b \u0434\u0430\u043d\u043d\u044b\u0445", None))
        self.download_btn.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c \u0442\u0435\u0441\u0442\u043e\u0432\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0448\u0438\u0431\u043a\u0438 \u0438 \u043b\u043e\u0433\u0438", None))
        self.logs_btn.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440 \u043b\u043e\u0433\u043e\u0432", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u0413\u0440\u0443\u043f\u043f\u0430", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u041e\u043f\u0446\u0438\u0438", None))
        self.language_lbl.setText(QCoreApplication.translate("MainWindow", u"Language", None))
        self.theme_lbl.setText(QCoreApplication.translate("MainWindow", u"Theme", None))
        self.radioButton.setText(QCoreApplication.translate("MainWindow", u"Light", None))
        self.radioButton_2.setText(QCoreApplication.translate("MainWindow", u"Dark", None))
        self.radioButton_3.setText(QCoreApplication.translate("MainWindow", u"System", None))
        self.font_lbl.setText(QCoreApplication.translate("MainWindow", u"Font", None))
        self.graphs_lbl.setText(QCoreApplication.translate("MainWindow", u"Graphs", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"\u0421\u0442\u0438\u043b\u044c \u043b\u0438\u043d\u0438\u0439 (Solid, Dashed, Dotted)", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"\u0426\u0432\u0435\u0442\u043e\u0432\u0430\u044f \u0441\u0445\u0435\u043c\u0430 \u0433\u0440\u0430\u0444\u0438\u043a\u043e\u0432 (Viridis, Plasma, Inferno)", None))
        self.export_lbl.setText(QCoreApplication.translate("MainWindow", u"Export", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"\u0424\u043e\u0440\u043c\u0430\u0442 \u044d\u043a\u0441\u043f\u043e\u0440\u0442\u0430 \u0434\u0430\u043d\u043d\u044b\u0445 (CSV, Excel, JSON)", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0443\u0442\u044c \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u044f", None))
        self.save_path_btn.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.choose_alg_lbl.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0431\u043e\u0440 \u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c\u0430 \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u0438", None))
        self.alg_box.setCurrentText("")
        self.advanced_btn.setText(QCoreApplication.translate("MainWindow", u"Advanced", None))
        self.cancel_btn.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
        self.save_btn.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.label_2.setText("")
        self.userName.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Username", None))
        self.email.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Email", None))
        self.phoneNo.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Phone Number", None))
        self.add_user_btn.setText(QCoreApplication.translate("MainWindow", u"Add User", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"\u00a9 Created by Mart-igor 2025", None))
    # retranslateUi

