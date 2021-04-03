# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog

import os
from os.path import expanduser
import sys

from utility.dialog import Ui_Dialog, Ui_WarningPaths, Ui_WarningName, Ui_WarningModels
import config



"""
from .ui to .py
pyuic5 -o pyfilename.py design.ui
 -x executable  if __name__ == "__main__":
"""

# class TimerThread
import time
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread  # QRunnable


class TimerThread(QThread):
    signal_timer = pyqtSignal(int)
    signal_timer_finish = pyqtSignal()

    @pyqtSlot(int)
    def slot_timer_start(self, value):
        self.seconds = value
        print("Start value:", value)

    def run(self):
        count = self.seconds
        while count > 0:
            time.sleep(1)
            count -= 1
            # print("seconds remaining",count)
            self.signal_timer.emit(count)
        self.signal_timer_finish.emit()



# class ModelSeletionThread

import auto
from utility.data import load_DS_as_np, load_CD_as_list, CD_processing

class ModelSeletionThread(QThread):
    signal_model_selection_finish = pyqtSignal()

    def run(self):
        cfg = config.load_config()

        DS_path = cfg['paths']['DS_abs_path']
        CD_path = cfg['paths']['CD_abs_path']
        DS = load_DS_as_np(DS_path)
        num_cols, cat_cols, txt_cols, label_col = CD_processing(CD_path)

        MS = auto.ModelSelection(
            cfg['experiment_name'],
            cfg['search_options']['duration'],
            cfg['model_requirements']['min_accuracy'],
            cfg['model_requirements']['max_memory'],
            cfg['model_requirements']['max_single_predict_time'],
            cfg['model_requirements']['max_train_time'],
            cfg['search_space'],
            cfg['search_options']['metric'],
            cfg['search_options']['validation'],               # TODO change API
            cfg['search_options']['iterations']
        )

        MS.fit(
            x = DS, # may contain columns that will not be used ('AUX',y, etc)
            y = DS[:, label_col],
            num_features=num_cols,
            cat_features=cat_cols,
            txt_features=txt_cols,
        )
        MS.save_results(n_best=cfg['search_options']['saved_top_models_amount'])

        print("ModelSeletionThread finish")
        self.signal_model_selection_finish.emit()





# main GUI class

class Ui_MainWindow(QMainWindow):
    signal_start_timer = pyqtSignal(int)

    def __init__(self):
        super(Ui_MainWindow, self).__init__()

        self.lcd_bool = True

        self.dialog_settings = Ui_Dialog()
        self.warning_paths = Ui_WarningPaths()
        self.warning_name = Ui_WarningName()

        self.warning_models = Ui_WarningModels()

        self.setupUi()

    def setupUi(self):
        cfg = config.load_config()

        self.setObjectName("MainWindow")
        self.setEnabled(True)
        self.resize(320, 480)
        self.setWindowIcon(QtGui.QIcon("utility/logo.png"))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(320, 480))
        self.setMaximumSize(QtCore.QSize(320, 480))
        self.setBaseSize(QtCore.QSize(900, 500))
        font = QtGui.QFont()
        font.setKerning(True)
        self.setFont(font)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.new_experiment = QtWidgets.QFrame(self.centralwidget)
        self.new_experiment.setEnabled(True)
        self.new_experiment.setGeometry(QtCore.QRect(10, 10, 301, 431))
        font = QtGui.QFont()
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.new_experiment.setFont(font)
        self.new_experiment.setAutoFillBackground(True)
        self.new_experiment.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.new_experiment.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.new_experiment.setObjectName("new_experiment")
        self.label_4 = QtWidgets.QLabel(self.new_experiment)
        self.label_4.setGeometry(QtCore.QRect(36, -10, 231, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.label_4.setFont(font)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_2 = QtWidgets.QLabel(self.new_experiment)
        self.label_2.setGeometry(QtCore.QRect(20, 90, 170, 30))
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.spinBox_all_time = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_all_time.setGeometry(QtCore.QRect(200, 90, 82, 26))
        self.spinBox_all_time.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.spinBox_all_time.setMinimum(5)
        self.spinBox_all_time.setMaximum(100000000)
        self.spinBox_all_time.setSingleStep(100)
        #        self.spinBox_all_time.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox_all_time.setProperty("value", cfg['search_options']['duration'])
        self.spinBox_all_time.setObjectName("spinBox_all_time")
        self.label_5 = QtWidgets.QLabel(self.new_experiment)
        self.label_5.setGeometry(QtCore.QRect(20, 170, 170, 30))
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName("label_5")
        self.spinBox_model_max_memory = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_model_max_memory.setGeometry(QtCore.QRect(200, 170, 82, 26))
        self.spinBox_model_max_memory.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.spinBox_model_max_memory.setMinimum(30)
        self.spinBox_model_max_memory.setMaximum(100000000)
        self.spinBox_model_max_memory.setSingleStep(100)
        #        self.spinBox_model_max_memory.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox_model_max_memory.setProperty("value", cfg['model_requirements']['max_memory'])
        self.spinBox_model_max_memory.setObjectName("spinBox_model_max_memory")
        self.lineEdit_experiment_name = QtWidgets.QLineEdit(self.new_experiment)
        self.lineEdit_experiment_name.setGeometry(QtCore.QRect(20, 50, 261, 21))
        font = QtGui.QFont()
        font.setItalic(False)
        font.setUnderline(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.lineEdit_experiment_name.setFont(font)
        self.lineEdit_experiment_name.setObjectName("lineEdit_experiment_name")
        self.spinBox_max_predict_time = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_max_predict_time.setGeometry(QtCore.QRect(200, 210, 82, 26))
        self.spinBox_max_predict_time.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.spinBox_max_predict_time.setMinimum(30)
        self.spinBox_max_predict_time.setMaximum(100000000)
        self.spinBox_max_predict_time.setSingleStep(100)
        #        self.spinBox_max_predict_time.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox_max_predict_time.setProperty("value", cfg['model_requirements']['max_single_predict_time'])
        self.spinBox_max_predict_time.setObjectName("spinBox_max_predict_time")
        self.label_6 = QtWidgets.QLabel(self.new_experiment)
        self.label_6.setGeometry(QtCore.QRect(20, 210, 170, 30))
        self.label_6.setTextFormat(QtCore.Qt.AutoText)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.label_12 = QtWidgets.QLabel(self.new_experiment)
        self.label_12.setGeometry(QtCore.QRect(20, 340, 170, 30))
        self.label_12.setTextFormat(QtCore.Qt.AutoText)
        self.label_12.setWordWrap(True)
        self.label_12.setObjectName("label_12")
        self.label_14 = QtWidgets.QLabel(self.new_experiment)
        self.label_14.setGeometry(QtCore.QRect(20, 315, 170, 30))
        self.label_14.setTextFormat(QtCore.Qt.AutoText)
        self.label_14.setWordWrap(True)
        self.label_14.setObjectName("label_14")
        self.btnLoadColumnsDescription = QtWidgets.QPushButton(self.new_experiment)
        self.btnLoadColumnsDescription.setGeometry(QtCore.QRect(200, 345, 82, 26))
        font = QtGui.QFont()
        font.setUnderline(True)
        self.btnLoadColumnsDescription.setFont(font)
        self.btnLoadColumnsDescription.setObjectName("btnLoadColumnsDescription")
        self.btnLoadDataset = QtWidgets.QPushButton(self.new_experiment)
        self.btnLoadDataset.setGeometry(QtCore.QRect(200, 315, 82, 26))
        font = QtGui.QFont()
        font.setUnderline(True)
        self.btnLoadDataset.setFont(font)
        self.btnLoadDataset.setObjectName("btnLoadDataset")
        self.btn_exp_back = QtWidgets.QPushButton(self.new_experiment)
        self.btn_exp_back.setGeometry(QtCore.QRect(20, 390, 71, 23))
        self.btn_exp_back.setObjectName("btn_exp_back")
        self.btnStart = QtWidgets.QPushButton(self.new_experiment)
        self.btnStart.setGeometry(QtCore.QRect(210, 390, 71, 23))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnStart.setFont(font)
        self.btnStart.setObjectName("btnStart")
        self.btn_settings = QtWidgets.QPushButton(self.new_experiment)
        self.btn_settings.setGeometry(QtCore.QRect(100, 390, 100, 23))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.btn_settings.setFont(font)
        self.btn_settings.setObjectName("btn_settings")
        self.label_3 = QtWidgets.QLabel(self.new_experiment)
        self.label_3.setGeometry(QtCore.QRect(20, 130, 170, 30))
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.spinBox_min_accuracy = QtWidgets.QDoubleSpinBox(self.new_experiment)
        #        self.spinBox_min_accuracy = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_min_accuracy.setGeometry(QtCore.QRect(200, 130, 82, 26))
        self.spinBox_min_accuracy.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.spinBox_min_accuracy.setMinimum(0)
        self.spinBox_min_accuracy.setMaximum(1)
        self.spinBox_min_accuracy.setSingleStep(0.05)
        #        self.spinBox_min_accuracy.setStepType(QtWidgets.QAbstractSpinBox.DefaultStepType)
        self.spinBox_min_accuracy.setProperty("value", cfg['model_requirements']['min_accuracy'])
        #        self.spinBox_min_accuracy.setDisplayIntegerBase(10)
        self.spinBox_min_accuracy.setObjectName("spinBox_min_accuracy")

        self.label_7 = QtWidgets.QLabel(self.new_experiment)
        self.label_7.setGeometry(QtCore.QRect(20, 250, 170, 30))
        self.label_7.setWordWrap(True)
        self.label_7.setObjectName("label_7")
        self.spinBox_all_time_2 = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_all_time_2.setGeometry(QtCore.QRect(200, 250, 82, 26))
        self.spinBox_all_time_2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.spinBox_all_time_2.setMinimum(30)
        self.spinBox_all_time_2.setMaximum(100000000)
        self.spinBox_all_time_2.setSingleStep(100)
        #        self.spinBox_all_time_2.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.spinBox_all_time_2.setProperty("value", cfg['model_requirements']['max_train_time'])
        self.spinBox_all_time_2.setObjectName("spinBox_all_time_2")

        #######

        self.label_9 = QtWidgets.QLabel(self.new_experiment)
        self.label_9.setGeometry(QtCore.QRect(20, 280, 170, 30))
        self.label_9.setWordWrap(True)
        self.label_9.setObjectName("label_8")
        self.spinBox_iter = QtWidgets.QSpinBox(self.new_experiment)
        self.spinBox_iter.setGeometry(QtCore.QRect(200, 280, 82, 26))
        self.spinBox_iter.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.spinBox_iter.setMinimum(1)
        self.spinBox_iter.setMaximum(100000000)
        self.spinBox_iter.setSingleStep(50)
        self.spinBox_iter.setProperty("value", cfg['search_options']['iterations'])
        self.spinBox_iter.setObjectName("spinBox_iter")

        ########

        self.search = QtWidgets.QFrame(self.centralwidget)
        self.search.setEnabled(True)
        self.search.setGeometry(QtCore.QRect(10, 10, 301, 431))
        font = QtGui.QFont()
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.search.setFont(font)
        self.search.setAutoFillBackground(True)
        self.search.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.search.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.search.setObjectName("search")
        self.lcd = QtWidgets.QLCDNumber(self.search)
        self.lcd.setGeometry(QtCore.QRect(-14, -10, 311, 131))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.lcd.setFont(font)
        self.lcd.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lcd.setAutoFillBackground(False)
        self.lcd.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lcd.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lcd.setLineWidth(3)
        self.lcd.setMidLineWidth(3)
        self.lcd.setSmallDecimalPoint(False)
        self.lcd.setDigitCount(10)
        self.lcd.setMode(QtWidgets.QLCDNumber.Dec)
        self.lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd.setProperty("value", 223.0)
        self.lcd.setProperty("intValue", 223)
        self.lcd.setObjectName("lcd")
        self.btn_goto_menu = QtWidgets.QPushButton(self.search)
        self.btn_goto_menu.setEnabled(False)
        self.btn_goto_menu.setGeometry(QtCore.QRect(110, 240, 71, 23))
        self.btn_goto_menu.setObjectName("btn_goto_menu")

        self.label = QtWidgets.QLabel(self.search)
        self.label.setGeometry(QtCore.QRect(20, 130, 261, 71))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")

        self.label_8 = QtWidgets.QLabel(self.search)
        self.label_8.setGeometry(QtCore.QRect(20, 160, 261, 71))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")

        self.menu = QtWidgets.QFrame(self.centralwidget)
        self.menu.setEnabled(True)
        self.menu.setGeometry(QtCore.QRect(10, 10, 301, 431))
        font = QtGui.QFont()
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.menu.setFont(font)
        self.menu.setAutoFillBackground(True)
        self.menu.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.menu.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.menu.setObjectName("menu")
        self.btn_menu_search = QtWidgets.QPushButton(self.menu)
        self.btn_menu_search.setGeometry(QtCore.QRect(10, 150, 281, 41))
        self.btn_menu_search.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setKerning(True)
        self.btn_menu_search.setFont(font)
        self.btn_menu_search.setObjectName("btn_menu_search")
        self.btn_menu_exit = QtWidgets.QPushButton(self.menu)
        self.btn_menu_exit.setGeometry(QtCore.QRect(10, 200, 281, 41))
        self.btn_menu_exit.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.btn_menu_exit.setFont(font)
        self.btn_menu_exit.setObjectName("btn_menu_exit")
        self.test = QtWidgets.QFrame(self.centralwidget)
        self.test.setEnabled(True)
        self.test.setGeometry(QtCore.QRect(10, 10, 301, 431))
        font = QtGui.QFont()
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.test.setFont(font)
        self.test.setAutoFillBackground(True)
        self.test.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.test.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.test.setObjectName("test")
        self.label_10 = QtWidgets.QLabel(self.test)
        self.label_10.setGeometry(QtCore.QRect(36, -10, 231, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.label_10.setFont(font)
        self.label_10.setTextFormat(QtCore.Qt.AutoText)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_16 = QtWidgets.QLabel(self.test)
        self.label_16.setGeometry(QtCore.QRect(20, 101, 171, 61))
        self.label_16.setTextFormat(QtCore.Qt.AutoText)
        self.label_16.setWordWrap(True)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.test)
        self.label_17.setGeometry(QtCore.QRect(20, 51, 171, 61))
        self.label_17.setTextFormat(QtCore.Qt.AutoText)
        self.label_17.setWordWrap(True)
        self.label_17.setObjectName("label_17")
        self.btnLoadColumnsDescription_2 = QtWidgets.QPushButton(self.test)
        self.btnLoadColumnsDescription_2.setGeometry(QtCore.QRect(200, 116, 81, 31))
        self.btnLoadColumnsDescription_2.setObjectName("btnLoadColumnsDescription_2")
        self.btnLoadDataset_2 = QtWidgets.QPushButton(self.test)
        self.btnLoadDataset_2.setGeometry(QtCore.QRect(200, 60, 81, 30))
        self.btnLoadDataset_2.setObjectName("btnLoadDataset_2")
        self.pushButton_10 = QtWidgets.QPushButton(self.test)
        self.pushButton_10.setGeometry(QtCore.QRect(60, 390, 71, 23))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButtonNext1_6 = QtWidgets.QPushButton(self.test)
        self.pushButtonNext1_6.setGeometry(QtCore.QRect(140, 390, 121, 23))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonNext1_6.setFont(font)
        self.pushButtonNext1_6.setObjectName("pushButtonNext1_6")
        self.btnSettings_4 = QtWidgets.QPushButton(self.test)
        self.btnSettings_4.setGeometry(QtCore.QRect(10, 180, 281, 41))
        self.btnSettings_4.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.btnSettings_4.setFont(font)
        self.btnSettings_4.setObjectName("btnSettings_4")
        self.test.raise_()
        self.new_experiment.raise_()
        self.search.raise_()
        self.menu.raise_()
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 320, 21))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # connect
        self.retranslateUi(self)
        self.btn_menu_search.clicked.connect(self.new_experiment.raise_)
        self.btn_menu_exit.clicked.connect(sys.exit)
        self.btnStart.clicked.connect(self.start_selection)  # start
        self.btn_exp_back.clicked.connect(self.menu.raise_)

        # защита от повторного нажатия после перезахода
        self.btn_goto_menu.clicked.connect(self.btn_goto_menu.setEnabled)
        self.btn_goto_menu.clicked.connect(self.menu.raise_)

        self.btnLoadDataset.clicked.connect(self.load_dataset_dialog)
        self.btnLoadColumnsDescription.clicked.connect(self.load_column_description_dialog)

        # Доп. настройки
        self.btn_settings.clicked.connect(self.dialog_settings.show)

        QtCore.QMetaObject.connectSlotsByName(self)

        self.show()

    # %%

    def retranslateUi(self, MainWindow):  # in russian
        cfg = config.load_config()

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AutoML"))
        self.label_4.setText(_translate("MainWindow", "New experiment"))
        self.label_2.setText(_translate("MainWindow", "Max search duration (sec)"))
        self.label_5.setText(_translate("MainWindow", "Max model memory (byte)"))
        self.lineEdit_experiment_name.setText(_translate("MainWindow", cfg['experiment_name']))
        self.label_6.setText(_translate("MainWindow", "Max predict time (ms)"))
        self.label_12.setText(_translate("MainWindow", "Load column description"))
        self.label_14.setText(_translate("MainWindow", "Load dataset"))
        self.btnLoadColumnsDescription.setText(_translate("MainWindow", "Load"))
        self.btnLoadDataset.setText(_translate("MainWindow", "Load"))
        self.btn_exp_back.setText(_translate("MainWindow", "Back"))
        self.btnStart.setText(_translate("MainWindow", "Start"))
        self.btn_settings.setText(_translate("MainWindow", "Advanced settings"))
        self.label_3.setText(_translate("MainWindow", "Min accuracy"))
        self.label_7.setText(_translate("MainWindow", "Max model training time (sec)"))
        self.btn_goto_menu.setText(_translate("MainWindow", "Menu"))
        self.label.setText(_translate("MainWindow", "Upon completion of the search, the results will be saved in:"))
        self.label_8.setText(_translate("MainWindow", r"C:\Uni\Dip"))
        self.btn_menu_search.setText(_translate("MainWindow", "Model selection"))
        self.btn_menu_exit.setText(_translate("MainWindow", "Exit"))
        self.label_10.setText(_translate("MainWindow", "Testing"))  # !!! not implemented
        self.label_16.setText(_translate("MainWindow", "Load dataset"))  # !!! not implemented
        self.label_17.setText(_translate("MainWindow", "Load model"))  # !!! not implemented
        self.btnLoadColumnsDescription_2.setText(_translate("MainWindow", "Load"))
        self.btnLoadDataset_2.setText(_translate("MainWindow", "Load"))
        self.pushButton_10.setText(_translate("MainWindow", "Back"))
        self.pushButtonNext1_6.setText(_translate("MainWindow", "Make prediction"))  # !!!
        self.btnSettings_4.setText(_translate("MainWindow", "Testing"))  # !!!
        self.label_9.setText(_translate("MainWindow", "Max iterations"))

    # %%

    def load_dataset_dialog(self):
        fileDlg = QFileDialog(self)
        # fileDlg.setDirectory('./')
        fileDlg.setDirectory(expanduser("~"))
        # Nikon (*.nef;*.nrw);;Sony (*.arw;*.srf;*.sr2);;All Files (*.*)
        fpath = fileDlg.getOpenFileName(filter="Dataset (*.csv)")[0]  # ;;Excel (*.xlsx)
        fpath = os.path.normpath(fpath)
        # print('Dataset path:',fpath) #!!! DEBUG

        # if DS file exist
        if os.path.isfile(fpath):
            # config.handle_dataset_path(fpath)
            cfg = config.load_config()
            cfg['paths']['DS_abs_path'] = fpath
            config.save_config(cfg)

            _translate = QtCore.QCoreApplication.translate
            self.btnLoadDataset.setText(_translate("MainWindow", "Loaded"))
            font = QtGui.QFont()
            font.setUnderline(False)
            self.btnLoadDataset.setFont(font)

    # %%

    def load_column_description_dialog(self):
        fileDlg = QFileDialog(self)
        # fileDlg.setDirectory('./')
        fileDlg.setDirectory(expanduser("~"))
        # Nikon (*.nef;*.nrw);;Sony (*.arw;*.srf;*.sr2);;All Files (*.*)
        fpath = fileDlg.getOpenFileName(filter="Column description (*.csv)")[0]
        fpath = os.path.normpath(fpath)
        # print('Columns description file path:',fpath) #!!! DEBUG

        # if CD file exist
        if os.path.isfile(fpath):
            # config.handle_column_description_path(fpath)
            cfg = config.load_config()
            cfg['paths']['CD_abs_path'] = fpath
            config.save_config(cfg)

            _translate = QtCore.QCoreApplication.translate
            self.btnLoadColumnsDescription.setText(_translate("MainWindow", "Loaded"))
            font = QtGui.QFont()
            font.setUnderline(False)
            self.btnLoadColumnsDescription.setFont(font)

    # %%
    def checkbox_state(self, checkbox):
        if checkbox.checkState() == 2:
            return True
        elif checkbox.checkState() == 0:
            return False

    def load_settings_from_gui(self):  # in config.json

        self.lcd_bool = True  # TODO fix

        cfg = config.load_config()

        cfg['experiment_name'] = self.lineEdit_experiment_name.text()

        cfg['model_requirements']['min_accuracy'] = self.spinBox_min_accuracy.value()
        cfg['model_requirements']['max_memory'] = self.spinBox_model_max_memory.value()
        cfg['model_requirements']['max_single_predict_time'] = self.spinBox_max_predict_time.value()
        cfg['model_requirements']['max_train_time'] = self.spinBox_all_time_2.value()

        cfg['search_options']['duration'] = self.spinBox_all_time.value()
        cfg['search_options']['iterations'] = self.spinBox_iter.value()
        cfg['search_options']['metric'] = self.dialog_settings.comboBox_metric.currentText()
        cfg['search_options']['validation'] = self.dialog_settings.comboBox_validation.currentText()  # TODO change API
        cfg['search_options'][
            'saved_top_models_amount'] = self.dialog_settings.comboBox_saved_count.currentText()  # TODO change API

        cfg['search_space'] = {
            'AdaBoost': self.checkbox_state(self.dialog_settings.checkBox_AdaBoost),
            'XGBoost': self.checkbox_state(self.dialog_settings.checkBox_XGBoost),
            'Bagging(SVC)': self.checkbox_state(self.dialog_settings.checkBox_BaggingSVC),
            'MLP': self.checkbox_state(self.dialog_settings.checkBox_MLP),
            'HistGB': self.checkbox_state(self.dialog_settings.checkBox_HistGB),
            'Ridge': self.checkbox_state(self.dialog_settings.checkBox_Ridge),
            'LinearSVC': self.checkbox_state(self.dialog_settings.checkBox_LinearSVC),
            'PassiveAggressive': self.checkbox_state(self.dialog_settings.checkBox_PassiveAggressive),
            'LogisticRegression': self.checkbox_state(self.dialog_settings.checkBox_LogisticRegression),
            'LDA': self.checkbox_state(self.dialog_settings.checkBox_LDA),
            'QDA': self.checkbox_state(self.dialog_settings.checkBox_QDA),
            'Perceptron': self.checkbox_state(self.dialog_settings.checkBox_Perceptron),
            'SVM': self.checkbox_state(self.dialog_settings.checkBox_SVM),
            'RandomForest': self.checkbox_state(self.dialog_settings.checkBox_RandomForest),
            'xRandTrees': self.checkbox_state(self.dialog_settings.checkBox_xRandTrees),
            'ELM': self.checkbox_state(self.dialog_settings.checkBox_ELM),
            'DecisionTree': self.checkbox_state(self.dialog_settings.checkBox_DecisionTree),
            'SGD': self.checkbox_state(self.dialog_settings.checkBox_SGD),
            'KNeighbors': self.checkbox_state(self.dialog_settings.checkBox_KNeighbors),
            'NearestCentroid': self.checkbox_state(self.dialog_settings.checkBox_NearestCentroid),
            'GaussianProcess': self.checkbox_state(self.dialog_settings.checkBox_GaussianProcess),
            'LabelSpreading': self.checkbox_state(self.dialog_settings.checkBox_LabelSpreading),
            'BernoulliNB': self.checkbox_state(self.dialog_settings.checkBox_BernoulliNB),
            'GaussianNB': self.checkbox_state(self.dialog_settings.checkBox_GaussianNB),
            'DBN': self.checkbox_state(self.dialog_settings.checkBox_DBN),
            'FactorizationMachine': self.checkbox_state(self.dialog_settings.checkBox_FactorizationMachine),
            'PolynomialNetwork': self.checkbox_state(self.dialog_settings.checkBox_PolynomialNetwork)
        }

        config.save_config(cfg)

    # %%

    @pyqtSlot(int)
    def slot_timer(self, value):
        if self.lcd_bool == True:
            self.lcd.display(value)

    @pyqtSlot()
    def slot_finish(self):
        self.lcd.display(0)
        self.lcd_bool = False

    #        self.btn_goto_menu.setEnabled(True)
    # print(self.sender())#тот кто отправил
    # print("Timer stops")

    @pyqtSlot()
    def slot_search_end(self):
        self.lcd.display(0)
        self.lcd_bool = False
        self.btn_goto_menu.setEnabled(True)
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("MainWindow", "Search completed, results saved to:"))

    def start_timer(self):
        cfg = config.load_config()

        self.lcd.display(cfg['search_options']['duration'])
        timer_thread = TimerThread(self)

        # thread.finished.connect(app.exit) #закрыть всё по завершению не уверен
        timer_thread.signal_timer.connect(self.slot_timer)
        timer_thread.signal_timer_finish.connect(self.slot_finish)

        self.signal_start_timer.connect(timer_thread.slot_timer_start)
        self.signal_start_timer.emit(cfg['search_options']['duration'])

        timer_thread.start()

    # %%

    def check_models(self):
        cfg = config.load_config()
        for val in cfg['search_space'].values():
            if val == True:
                return True
        return False

    # %%

    def start_selection(self):
        cfg = config.load_config()  # first call
        if (cfg['paths']['DS_abs_path'] != None) and (cfg['paths']['CD_abs_path'] != None):

            # load setting from GUI to config.json
            self.load_settings_from_gui()

            # update var
            cfg = config.load_config()

            if cfg['experiment_name'] != '':

                if self.check_models() == True:
                    self.search.raise_()

                    ###########################
                    self.start_timer()
                    ###########################

                    MS_thread = ModelSeletionThread(self)

                    # thread.finished.connect(app.exit) #закрыть всё по завершению не уверен
                    # MS_thread.signal_timer.connect(self.slot_timer)
                    # MS_thread.signal_timer_finish.connect(self.slot_finish)
                    MS_thread.signal_model_selection_finish.connect(self.slot_search_end)

                    MS_thread.start()
                else:
                    self.warning_models.show()
            else:
                self.warning_name.show()
        else:
            self.warning_paths.show()


# %%

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    sys.exit(app.exec_())
