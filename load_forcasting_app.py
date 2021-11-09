import os
import tkinter as tk
import tkinter.ttk as ttk
from pygubu.widgets.calendarframe import CalendarFrame
from pygubu.widgets.pathchooserinput import PathChooserInput
from V3_functionality import get_arff_csv_file,  import_prev_load_data_for_training_or_analysis, info_gained_data_analysis, Correlation_data_analysis, data_summery, split_data_file,vote,bagging,AdaBoostM1,build_cls, convert_date_format, quick_eval,ten_fold_evaal, find_anom_forcasts_by_class_att,convert_forcast_to_test_data
from plf_v3 import Perform_forcast
import weka.core.jvm as jvm
import pandas as pd
import plotly.express as px

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

PROJECT_UI = os.path.join(PROJECT_PATH, "Load_forcasting_GUI.ui")

class Load_forcasting_app:
    def __init__(self, master=None):
        # build ui
        self.attack_cls_att = None
        self.Toplevel_1 = tk.Tk() if master is None else tk.Toplevel(master)
        self.Notebook_1 = ttk.Notebook(self.Toplevel_1)
        self.Data_Frame = ttk.Frame(self.Notebook_1)
        self.Data_tab_lable = ttk.Label(self.Data_Frame)
        self.Data_tab_lable_TEXT = tk.StringVar(value='Data Analysis & manipulisation')
        self.Data_tab_lable.configure(background='#535353', font='{Arial} 16 {bold italic}', foreground='#ffffff', text='Data Analysis & manipulisation')
        self.Data_tab_lable.configure(textvariable=self.Data_tab_lable_TEXT)
        self.Data_tab_lable.place(anchor='nw', relheight='0.05', relwidth='1.0', x='0', y='0')
        self.DataTab_canvas = tk.Canvas(self.Data_Frame)
        self.DataTab_canvas.configure(background='#595959')
        self.DataTab_canvas.place(anchor='nw', relheight='0.86', relwidth='1.0', relx='0', rely='0.05', x='0', y='0')
        self.data_tab_separator = ttk.Separator(self.Data_Frame)
        self.data_tab_separator.configure(orient='horizontal')
        self.data_tab_separator.place(anchor='nw', relheight='0.757', relwidth='0.005', relx='0.5', rely='0.05', x='0', y='0')
        self.Info_Gained_button = ttk.Button(self.Data_Frame)
        self.Info_Gained_button.configure(text='Infomation Gained Analysis')
        self.Info_Gained_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.675', rely='0.4', x='0', y='0')
        self.Info_Gained_button.configure(command=self.Info_Gained_button_pressed)
        self.Correlation_button = ttk.Button(self.Data_Frame)
        self.Correlation_button.configure(text='Correlation Analysis')
        self.Correlation_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.675', rely='0.65', x='0', y='0')
        self.Correlation_button.configure(command=self.Correlation_button_pressed)
        self.Data_file_path_chooser = PathChooserInput(self.Data_Frame)
        self.Data_file_path_chooser_TEXT = tk.StringVar(value='')
        self.Data_file_path_chooser.configure(textvariable=self.Data_file_path_chooser_TEXT, type='file')
        self.Data_file_path_chooser.place(anchor='nw', relheight='0.055', relwidth='1.0', relx='0.001', rely='0.825', x='0', y='0')
        self.Data_file_import_lable = ttk.Label(self.Data_Frame)
        self.Data_file_import_lable_TEXT = tk.StringVar(value='Data file import')
        self.Data_file_import_lable.configure(font='{Arial} 12 {}', text='Data file import', textvariable=self.Data_file_import_lable_TEXT)
        self.Data_file_import_lable.place(anchor='nw', relwidth='1.0', relx='0.001', rely='0.8', x='0', y='0')
        self.File_selected_lable = ttk.Label(self.Data_Frame)
        self.Data_File_selected_TEXT = tk.StringVar(value='Current File Selected: Warning, No File Selected')
        self.File_selected_lable.configure(font='{Arial} 14 {}', text='Current File Selected: Warning, No File Selected', textvariable=self.Data_File_selected_TEXT)
        self.File_selected_lable.place(anchor='nw', relheight='0.04', relwidth='1.0', relx='0.001', rely='0.87', x='0', y='0')
        self.Data_Summery_button = ttk.Button(self.Data_Frame)
        self.Data_Summery_button.configure(text='Data Summery Analysis')
        self.Data_Summery_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.675', rely='0.15', x='0', y='0')
        self.Data_Summery_button.configure(command=self.Data_Summery_button_pressed)
        self.data_split_percentage = ttk.Scale(self.Data_Frame)
        self.test_data_percentage_variable = tk.StringVar(value='80')
        self.data_split_percentage.configure(from_='0', orient='horizontal', to='100', value='80')
        self.data_split_percentage.configure(variable=self.test_data_percentage_variable)
        self.data_split_percentage.place(anchor='nw', relwidth='0.2', relx='0.15', rely='0.25', x='0', y='0')
        self.split_data_file_Button = ttk.Button(self.Data_Frame)
        self.split_data_file_Button.configure(text='Split Data file')
        self.split_data_file_Button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.15', rely='0.15', x='0', y='0')
        self.split_data_file_Button.configure(command=self.split_data_file)
        self.start_time_entry = ttk.Entry(self.Data_Frame)
        self.start_time_entry_text = tk.StringVar(value='HH-MM-SS')
        self.start_time_entry.configure(textvariable=self.start_time_entry_text)
        _text_ = '''HH-MM-SS'''
        self.start_time_entry.delete('0', 'end')
        self.start_time_entry.insert('0', _text_)
        self.start_time_entry.place(anchor='nw', relwidth='0.075', relx='0.075', rely='0.7', x='0', y='0')
        self.datetimeinputlable = ttk.Label(self.Data_Frame)
        self.datetimeinputlable.configure(background='#595959', font='{Arial} 16 {bold}', foreground='#ffffff', text='Date Time input')
        self.datetimeinputlable.place(anchor='nw', relx='0.16', rely='0.425', x='0', y='0')
        self.tolable = ttk.Label(self.Data_Frame)
        self.tolable.configure(background='#595959', font='{Arial} 16 {bold}', foreground='#ffffff', text='To:')
        self.tolable.place(anchor='nw', relx='0.235', rely='0.55', x='0', y='0')
        self.horizontal_Separator_DATATAB = ttk.Separator(self.Data_Frame)
        self.horizontal_Separator_DATATAB.configure(orient='horizontal')
        self.horizontal_Separator_DATATAB.place(anchor='nw', relheight='0.01', relwidth='0.5', relx='0.001', rely='0.4', x='0', y='0')
        self.hundred_lable = ttk.Label(self.Data_Frame)
        self.hundred_lable.configure(background='#000000', cursor='based_arrow_up', font='{Arial} 12 {}', foreground='#ffffff')
        self.hundred_lable.configure(text='100 %')
        self.hundred_lable.place(anchor='nw', relx='0.35', rely='0.25', x='0', y='0')
        self.zero_lable = ttk.Label(self.Data_Frame)
        self.zero_lable.configure(background='#000000', cursor='based_arrow_up', font='{Arial} 12 {}', foreground='#ffffff')
        self.zero_lable.configure(text='0 %')
        self.zero_lable.place(anchor='nw', relx='0.12', rely='0.25', x='0', y='0')
        self.end_time_input = ttk.Entry(self.Data_Frame)
        self.end_time_input_text = tk.StringVar(value='HH-MM-SS')
        self.end_time_input.configure(textvariable=self.end_time_input_text)
        _text_ = '''HH-MM-SS'''
        self.end_time_input.delete('0', 'end')
        self.end_time_input.insert('0', _text_)
        self.end_time_input.place(anchor='nw', relwidth='0.075', relx='0.375', rely='0.7', x='0', y='0')
        self.Perform_analysis = ttk.Label(self.Data_Frame)
        self.Perform_analysis.configure(background='#595959', font='{Arial} 16 {bold}', foreground='#ffffff', text='Perform Data Analysis')
        self.Perform_analysis.place(anchor='nw', relx='0.65', rely='0.075', x='0', y='0')
        self.data_start_date_calender = CalendarFrame(self.Data_Frame)
        self.data_start_date_calender.configure(relief='flat')
        # TODO - self.data_start_date_calender: code for custom option 'firstweekday' not implemented.
        # TODO - self.data_start_date_calender: code for custom option 'month' not implemented.
        self.data_start_date_calender.place(anchor='nw', relheight='0.2', relwidth='0.175', relx='0.03', rely='0.475', x='0', y='0')
        self.data_end_date_calender = CalendarFrame(self.Data_Frame)
        self.data_end_date_calender.configure(relief='flat')
        # TODO - self.data_end_date_calender: code for custom option 'firstweekday' not implemented.
        # TODO - self.data_end_date_calender: code for custom option 'month' not implemented.
        self.data_end_date_calender.place(anchor='nw', relheight='0.2', relwidth='0.175', relx='0.315', rely='0.475', x='0', y='0')
        self.Data_Frame.configure(height='200', width='200')
        self.Data_Frame.place(anchor='nw', x='0', y='0')
        self.Notebook_1.add(self.Data_Frame, text='Data')
        self.Classifier_frame = ttk.Frame(self.Notebook_1)
        self.ClassifierTab_Lable = ttk.Label(self.Classifier_frame)
        self.ClassifierTab_Lable.configure(background='#535353', font='{Arial} 16 {bold italic}', foreground='#ffffff', text='Classifier Building & Options')
        self.ClassifierTab_Lable.place(anchor='nw', relheight='0.05', relwidth='1.0', x='0', y='0')
        self.ClassifierTab_Canvas = tk.Canvas(self.Classifier_frame)
        self.ClassifierTab_Canvas.configure(background='#595959')
        self.ClassifierTab_Canvas.place(anchor='nw', relheight='0.86', relwidth='1.0', relx='0', rely='0.05', x='0', y='0')
        self.classifier_tab_separator = ttk.Separator(self.Classifier_frame)
        self.classifier_tab_separator.configure(orient='horizontal')
        self.classifier_tab_separator.place(anchor='nw', relheight='0.757', relwidth='0.005', relx='0.5', rely='0.05', x='0', y='0')
        self.Build_Cls_button = ttk.Button(self.Classifier_frame)
        self.Data_Summery_Button_TEXT = tk.StringVar(value='Build Classifier')
        self.Build_Cls_button.configure(text='Build Classifier', textvariable=self.Data_Summery_Button_TEXT)
        self.Build_Cls_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.15', rely='0.4', x='0', y='0')
        self.Build_Cls_button.configure(command=self.Build_Classifier)
        self.Traning_data_Pathchooserinput = PathChooserInput(self.Classifier_frame)
        self.Traning_data_Pathchooserinput_TEXT = tk.StringVar(value='')
        self.Traning_data_Pathchooserinput.configure(textvariable=self.Traning_data_Pathchooserinput_TEXT, type='file')
        self.Traning_data_Pathchooserinput.place(anchor='nw', relheight='0.055', relwidth='1.0', relx='0.001', rely='0.825', x='0', y='0')
        self.Traning_data_lable = ttk.Label(self.Classifier_frame)
        self.Traning_data_lable_TEXT = tk.StringVar(value='Training Data file import')
        self.Traning_data_lable.configure(font='{Arial} 12 {}', text='Training Data file import', textvariable=self.Traning_data_lable_TEXT)
        self.Traning_data_lable.place(anchor='nw', relwidth='1.0', relx='0.001', rely='0.8', x='0', y='0')
        self.Traning_data_file_name_display = ttk.Label(self.Classifier_frame)
        self.Training_Data_File_selected_TEXT = tk.StringVar(value='Current Training Data File Selected: Warning, No File Selected')
        self.Traning_data_file_name_display.configure(font='{Arial} 14 {}', text='Current Training Data File Selected: Warning, No File Selected', textvariable=self.Training_Data_File_selected_TEXT)
        self.Traning_data_file_name_display.place(anchor='nw', relheight='0.04', relwidth='1.0', relx='0.001', rely='0.87', x='0', y='0')
        self.Bagging_Checkbutton = ttk.Checkbutton(self.Classifier_frame)
        self.Bagging_Checkbutton_value = tk.StringVar(value='')
        self.Bagging_Checkbutton.configure(text='Bagging Single Classifier enchancer', variable=self.Bagging_Checkbutton_value)
        self.Bagging_Checkbutton.place(anchor='nw', relx='0.65', rely='0.3', x='0', y='0')
        self.Bagging_Checkbutton.configure(command=self.Bagging_Checkbutton_activated)
        self.AdaBoostM1_Checkbutton = ttk.Checkbutton(self.Classifier_frame)
        self.AdaBoostM1_Checkbutton_value = tk.StringVar(value='')
        self.AdaBoostM1_Checkbutton.configure(text='AdaBoostM1 Single Classifier enchancer', variable=self.AdaBoostM1_Checkbutton_value)
        self.AdaBoostM1_Checkbutton.place(anchor='nw', relx='0.65', rely='0.425', x='0', y='0')
        self.AdaBoostM1_Checkbutton.configure(command=self.AdaBoostM1_Checkbutton_activated)
        self.Classifier_algorithm_choice = tk.StringVar(value='Select Classification Algorithm')
        __values = ['weka.classifiers.functions.PLSClassifier', 'weka.classifiers.functions.SimpleLinearRegression', 'weka.classifiers.functions.SMOreg','weka.classifiers.rules.M5Rules', 'weka.classifiers.trees.M5P']
        self.Classifier_algorithm_picker_menu = tk.OptionMenu(self.Classifier_frame, self.Classifier_algorithm_choice, 'Select Classification Algorithm', *__values, command=self.Classifier_algorithm_picker_choice)
        self.Classifier_algorithm_picker_menu.place(anchor='nw', relx='0.1325', rely='0.3', x='0', y='0')
        self.Classifier_frame.configure(height='200', width='200')
        self.Classifier_frame.place(anchor='nw', x='0', y='0')
        self.Notebook_1.add(self.Classifier_frame, text='Classifier')
        self.Test_data_frame = ttk.Frame(self.Notebook_1)
        self.Test_data_label = ttk.Label(self.Test_data_frame)
        self.Test_data_label.configure(background='#535353', font='{Arial} 16 {bold italic}', foreground='#ffffff', text='Check New Data for Anomolies')
        self.Test_data_label.place(anchor='nw', relheight='0.05', relwidth='1.0', x='0', y='0')
        self.Test_data_canvas = tk.Canvas(self.Test_data_frame)
        self.Test_data_canvas.configure(background='#595959')
        self.Test_data_canvas.place(anchor='nw', relheight='0.86', relwidth='1.0', relx='0', rely='0.05', x='0', y='0')
        self.Test_data_button = ttk.Button(self.Test_data_frame)
        self.Test_Data_Button_TEXT = tk.StringVar(value='Test Data')
        self.Test_data_button.configure(text='Test Data', textvariable=self.Test_Data_Button_TEXT)
        self.Test_data_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.4', rely='0.4', x='0', y='0')
        self.Test_data_button.configure(command=self.Test_test_data)
        self.test_data_path_chosser = PathChooserInput(self.Test_data_frame)
        self.test_data_path_chosser_TEXT = tk.StringVar(value='')
        self.test_data_path_chosser.configure(textvariable=self.test_data_path_chosser_TEXT, type='file')
        self.test_data_path_chosser.place(anchor='nw', relheight='0.055', relwidth='1.0', relx='0.001', rely='0.825', x='0', y='0')
        self.Testing_Data_file_import_lable = ttk.Label(self.Test_data_frame)
        self.Testing_Data_file_import_lable.configure(font='{Arial} 12 {}', text='Testing Data file import')
        self.Testing_Data_file_import_lable.place(anchor='nw', relwidth='1.0', relx='0.001', rely='0.8', x='0', y='0')
        self.Current_Testing_Data_File_lable = ttk.Label(self.Test_data_frame)
        self.Testing_Data_File_selected_TEXT = tk.StringVar(value='Current Testing Data File Selected: Warning, No File Selected')
        self.Current_Testing_Data_File_lable.configure(font='{Arial} 14 {}', text='Current Testing Data File Selected: Warning, No File Selected', textvariable=self.Testing_Data_File_selected_TEXT)
        self.Current_Testing_Data_File_lable.place(anchor='nw', relheight='0.04', relwidth='1.0', relx='0.001', rely='0.87', x='0', y='0')
        self.test_date_cls_selected_lable = ttk.Label(self.Test_data_frame)
        self.test_date_cls_selected_lable_text = tk.StringVar(value='Classifier selected: No CLS found!')
        self.test_date_cls_selected_lable.configure(font='{Arial} 12 {}', text='Classifier selected: No CLS found!', textvariable=self.test_date_cls_selected_lable_text)
        self.test_date_cls_selected_lable.place(anchor='nw', relheight='0.04', relwidth='0.3', relx='0.35', rely='0.2', x='0', y='0')
        self.Test_data_frame.configure(height='200', width='200')
        self.Test_data_frame.place(anchor='nw', x='0', y='0')
        self.Notebook_1.add(self.Test_data_frame, text='Test Data')
        self.Evaluation_Frame = ttk.Frame(self.Notebook_1)
        self.Evaluation_tab_label = ttk.Label(self.Evaluation_Frame)
        self.Evaluation_tab_label.configure(background='#535353', font='{Arial} 16 {bold italic}', foreground='#ffffff', text='Evaluate classifier')
        self.Evaluation_tab_label.place(anchor='nw', relheight='0.05', relwidth='1.0', x='0', y='0')
        self.Evaluation_Canvas = tk.Canvas(self.Evaluation_Frame)
        self.Evaluation_Canvas.configure(background='#595959')
        self.Evaluation_Canvas.place(anchor='nw', relheight='0.86', relwidth='1.0', relx='0', rely='0.05', x='0', y='0')
        self.Quick_eval_button = ttk.Button(self.Evaluation_Frame)
        self.Quick_eval_button_TEXT = tk.StringVar(value='Quick Evaluation')
        self.Quick_eval_button.configure(text='Quick Evaluation', textvariable=self.Quick_eval_button_TEXT)
        self.Quick_eval_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.4', rely='0.5', x='0', y='0')
        self.Quick_eval_button.configure(command=self.perform_Quick_eval)
        self.Evaluation_data_file_path_chosser = PathChooserInput(self.Evaluation_Frame)
        self.Evaluation_data_file_path_chosser_TEXT = tk.StringVar(value='')
        self.Evaluation_data_file_path_chosser.configure(textvariable=self.Evaluation_data_file_path_chosser_TEXT, type='file')
        self.Evaluation_data_file_path_chosser.place(anchor='nw', relheight='0.055', relwidth='1.0', relx='0.001', rely='0.825', x='0', y='0')
        self.Evaluation_file_import_lable = ttk.Label(self.Evaluation_Frame)
        self.Evaluation_file_import_lable.configure(font='{Arial} 12 {}', text='Evaluation Data file import')
        self.Evaluation_file_import_lable.place(anchor='nw', relwidth='1.0', relx='0.001', rely='0.8', x='0', y='0')
        self.Evaluation_Data_File_selected_lable = ttk.Label(self.Evaluation_Frame)
        self.Evaluation_Data_File_selected_TEXT = tk.StringVar(value='Current Evaluation Data File Selected: Warning, No File Selected')
        self.Evaluation_Data_File_selected_lable.configure(font='{Arial} 14 {}', text='Current Evaluation Data File Selected: Warning, No File Selected', textvariable=self.Evaluation_Data_File_selected_TEXT)
        self.Evaluation_Data_File_selected_lable.place(anchor='nw', relheight='0.04', relwidth='1.0', relx='0.001', rely='0.87', x='0', y='0')
        self.TEN_Cross_fold_Evaluation = ttk.Button(self.Evaluation_Frame)
        self.TEN_Cross_fold_Evaluation_TEXT = tk.StringVar(value='10 Cross Fold Evaluation')
        self.TEN_Cross_fold_Evaluation.configure(text='10 Cross Fold Evaluation', textvariable=self.TEN_Cross_fold_Evaluation_TEXT)
        self.TEN_Cross_fold_Evaluation.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.4', rely='0.3', x='0', y='0')
        self.TEN_Cross_fold_Evaluation.configure(command=self.perform_ten_fold_eval)
        self.eval_tab_cls_selection = ttk.Label(self.Evaluation_Frame)
        self.eval_tab_cls_selection_text = tk.StringVar(value='Classifier selected: No CLS found!')
        self.eval_tab_cls_selection.configure(font='{Arial} 12 {}', text='Classifier selected: No CLS found!', textvariable=self.eval_tab_cls_selection_text)
        self.eval_tab_cls_selection.place(anchor='nw', relheight='0.04', relwidth='0.3', relx='0.35', rely='0.2', x='0', y='0')
        self.Evaluation_Frame.configure(height='200', width='200')
        self.Evaluation_Frame.place(anchor='nw', x='0', y='0')
        self.Notebook_1.add(self.Evaluation_Frame, text='Evaluation')
        self.Visualization_Frame = ttk.Frame(self.Notebook_1)
        self.Visualization_tab_lable = ttk.Label(self.Visualization_Frame)
        self.Visualization_tab_lable.configure(background='#535353', font='{Arial} 16 {bold italic}', foreground='#ffffff', text='Visualize Results & Data')
        self.Visualization_tab_lable.place(anchor='nw', relheight='0.05', relwidth='1.0', x='0', y='0')
        self.Visualization_Canvas = tk.Canvas(self.Visualization_Frame)
        self.Visualization_Canvas.configure(background='#595959')
        self.Visualization_Canvas.place(anchor='nw', relheight='0.86', relwidth='1.0', relx='0', rely='0.05', x='0', y='0')
        self.Visualize_results_button = ttk.Button(self.Visualization_Frame)
        self.Visualize_Results_TEXT = tk.StringVar(value='Visualize Results')
        self.Visualize_results_button.configure(text='Visualize Results', textvariable=self.Visualize_Results_TEXT)
        self.Visualize_results_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.65', rely='0.14', x='0', y='0')
        self.Visualize_results_button.configure(command=self.perform_results_visulisation)
        self.vis_data_file_path_chooser = PathChooserInput(self.Visualization_Frame)
        self.vis_data_file_path_chooser_TEXT = tk.StringVar(value='')
        self.vis_data_file_path_chooser.configure(textvariable=self.vis_data_file_path_chooser_TEXT, type='file')
        self.vis_data_file_path_chooser.place(anchor='nw', relwidth='0.275', relx='0.12', rely='0.25', x='0', y='0')
        self.data_File_for_Visualizeation_import_lable = ttk.Label(self.Visualization_Frame)
        self.data_File_for_Visualizeation_import_lable.configure(font='{Arial} 11 {}', text='Data File for Visualizeation import')
        self.data_File_for_Visualizeation_import_lable.place(anchor='nw', relwidth='0.275', relx='0.12', rely='0.225', x='0', y='0')
        self.Current_Data_File_Selected_lable = ttk.Label(self.Visualization_Frame)
        self.Current_Data_File_Selected_lable_text = tk.StringVar(value='Current Data File Selected: Warning, No File Selected')
        self.Current_Data_File_Selected_lable.configure(font='{Arial} 7 {}', text='Current Data File Selected: Warning, No File Selected', textvariable=self.Current_Data_File_Selected_lable_text)
        self.Current_Data_File_Selected_lable.place(anchor='nw', relheight='0.04', relwidth='0.275', relx='0.12', rely='0.29', x='0', y='0')
        self.Visualize_data_button = ttk.Button(self.Visualization_Frame)
        self.Visualize_data_button_TEXT = tk.StringVar(value='Visualize Data')
        self.Visualize_data_button.configure(text='Visualize Data', textvariable=self.Visualize_data_button_TEXT)
        self.Visualize_data_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.15', rely='0.14', x='0', y='0')
        self.Visualize_data_button.configure(command=self.perform_data_Visualize)
        self.vis_results_file_path_chooser = PathChooserInput(self.Visualization_Frame)
        self.vis_results_file_path_chooser_TEXT = tk.StringVar(value='')
        self.vis_results_file_path_chooser.configure(textvariable=self.vis_results_file_path_chooser_TEXT, type='file')
        self.vis_results_file_path_chooser.place(anchor='nw', relwidth='0.275', relx='0.6', rely='0.25', x='0', y='0')
        self.Results_File_for_Visualizeation_lable = ttk.Label(self.Visualization_Frame)
        self.Results_File_for_Visualizeation_lable.configure(font='{Arial} 11 {}', text='Results File for Visualizeation import')
        self.Results_File_for_Visualizeation_lable.place(anchor='nw', relwidth='0.275', relx='0.6', rely='0.225', x='0', y='0')
        self.Current_Results_File_lable = ttk.Label(self.Visualization_Frame)
        self.Current_Results_File_lable_text = tk.StringVar(value='Current Results File Selected: Warning, No File Selected')
        self.Current_Results_File_lable.configure(font='{Arial} 7 {}', text='Current Results File Selected: Warning, No File Selected', textvariable=self.Current_Results_File_lable_text)
        self.Current_Results_File_lable.place(anchor='nw', relheight='0.04', relwidth='0.275', relx='0.6', rely='0.29', x='0', y='0')
        self.vis_date_input_lable1 = ttk.Label(self.Visualization_Frame)
        self.vis_date_input_lable1.configure(background='#595959', font='{Arial} 16 {bold italic}', foreground='#ffffff', text='Date Time input From:')
        self.vis_date_input_lable1.place(anchor='nw', relx='0.375', rely='0.525', x='0', y='0')
        self.vis_start_time_entry = ttk.Entry(self.Visualization_Frame)
        self.vis_start_time_entry_etxt = tk.StringVar(value='HH-MM-SS')
        self.vis_start_time_entry.configure(textvariable=self.vis_start_time_entry_etxt)
        _text_ = '''HH-MM-SS'''
        self.vis_start_time_entry.delete('0', 'end')
        self.vis_start_time_entry.insert('0', _text_)
        self.vis_start_time_entry.place(anchor='nw', relwidth='0.075', relx='0.22', rely='0.8', x='0', y='0')
        self.vis_to_lable = ttk.Label(self.Visualization_Frame)
        self.vis_to_lable.configure(background='#c0c0c0', font='{Arial} 12 {bold}', foreground='#000000', text='To:')
        self.vis_to_lable.place(anchor='nw', relx='0.475', rely='0.6', x='0', y='0')
        self.vis_end_time_entry = ttk.Entry(self.Visualization_Frame)
        self.vis_end_time_entry_etxt = tk.StringVar(value='HH-MM-SS')
        self.vis_end_time_entry.configure(textvariable=self.vis_end_time_entry_etxt)
        _text_ = '''HH-MM-SS'''
        self.vis_end_time_entry.delete('0', 'end')
        self.vis_end_time_entry.insert('0', _text_)
        self.vis_end_time_entry.place(anchor='nw', relwidth='0.075', relx='0.7', rely='0.8', x='0', y='0')
        self.Separator_4 = ttk.Separator(self.Visualization_Frame)
        self.Separator_4.configure(orient='vertical')
        self.Separator_4.place(anchor='nw', relheight='0.46', relwidth='0.01', relx='0.5', rely='0.05', x='0', y='0')
        self.Separator_5 = ttk.Separator(self.Visualization_Frame)
        self.Separator_5.configure(orient='horizontal')
        self.Separator_5.place(anchor='nw', relheight='0.01', relwidth='1', relx='0.001', rely='0.5', x='0', y='0')
        self.vis_start_date_calender = CalendarFrame(self.Visualization_Frame)
        self.vis_start_date_calender.configure(relief='flat')
        # TODO - self.vis_start_date_calender: code for custom option 'firstweekday' not implemented.
        # TODO - self.vis_start_date_calender: code for custom option 'month' not implemented.
        self.vis_start_date_calender.place(anchor='nw', relheight='0.2', relwidth='0.175', relx='0.15', rely='0.55', x='0', y='0')
        self.vis_end_date_calender = CalendarFrame(self.Visualization_Frame)
        self.vis_end_date_calender.configure(relief='flat')
        # TODO - self.vis_end_date_calender: code for custom option 'firstweekday' not implemented.
        # TODO - self.vis_end_date_calender: code for custom option 'month' not implemented.
        self.vis_end_date_calender.place(anchor='nw', relheight='0.2', relwidth='0.175', relx='0.65', rely='0.55', x='0', y='0')
        self.Visualization_Frame.configure(height='200', width='200')
        self.Visualization_Frame.place(anchor='nw', x='0', y='0')
        self.Notebook_1.add(self.Visualization_Frame, text='Visualization')
        self.Forcasting_Tab_Frame = ttk.Frame(self.Notebook_1)
        self.Forcasting_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.Forcasting_lable.configure(background='#535353', font='{Arial} 16 {bold italic}', foreground='#ffffff', text='Forcasting')
        self.Forcasting_lable.place(anchor='nw', relheight='0.05', relwidth='1.0', x='0', y='0')
        self.Forcasting_Canvas = tk.Canvas(self.Forcasting_Tab_Frame)
        self.Forcasting_Canvas.configure(background='#595959')
        self.Forcasting_Canvas.place(anchor='nw', relheight='0.86', relwidth='1.0', relx='0', rely='0.05', x='0', y='0')
        self.Load_data_Dir_Pathchooserinput = PathChooserInput(self.Forcasting_Tab_Frame)
        self.Load_data_Dir_Pathchooserinput_text = tk.StringVar(value='')
        self.Load_data_Dir_Pathchooserinput.configure(textvariable=self.Load_data_Dir_Pathchooserinput_text, type='directory')
        self.Load_data_Dir_Pathchooserinput.place(anchor='nw', relheight='0.04', relwidth='0.35', relx='0.05', rely='0.15', x='0', y='0')
        self.forcasting_input_data_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.forcasting_input_data_lable.configure(background='#595959', font='{Arial} 16 {bold underline}', foreground='#ffffff', text='Forecasting Input Data')
        self.forcasting_input_data_lable.place(anchor='nw', relwidth='0.2', relx='0.1', rely='0.1', x='0', y='0')
        self.Load_data_Dir_output_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.Load_data_Dir_output_lable_TEXT = tk.StringVar(value='Directory Path For Load Data')
        self.Load_data_Dir_output_lable.configure(font='{Arial} 11 {}', text='Directory Path For Load Data', textvariable=self.Load_data_Dir_output_lable_TEXT)
        self.Load_data_Dir_output_lable.place(anchor='nw', relheight='0.04', relwidth='0.35', relx='0.05', rely='0.19', x='0', y='0')
        self.Execute = ttk.Button(self.Forcasting_Tab_Frame)
        self.Execute_text = tk.StringVar(value='Execute')
        self.Execute.configure(text='Execute', textvariable=self.Execute_text)
        self.Execute.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.425', rely='0.8', x='0', y='0')
        self.Execute.configure(command=self.Execute_forcast)
        self.severity_factor_Pathchooserinput = PathChooserInput(self.Forcasting_Tab_Frame)
        self.severity_factor_Pathchooserinput_text = tk.StringVar(value='')
        self.severity_factor_Pathchooserinput.configure(textvariable=self.severity_factor_Pathchooserinput_text, type='file')
        self.severity_factor_Pathchooserinput.place(anchor='nw', relheight='0.04', relwidth='0.35', relx='0.05', rely='0.3', x='0', y='0')
        self.File_Path_For_Severity_Factor_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.File_Path_For_Severity_Factor_lable_text = tk.StringVar(value='File Path For Severity Factor')
        self.File_Path_For_Severity_Factor_lable.configure(font='{Arial} 11 {}', text='File Path For Severity Factor', textvariable=self.File_Path_For_Severity_Factor_lable_text)
        self.File_Path_For_Severity_Factor_lable.place(anchor='nw', relheight='0.04', relwidth='0.35', relx='0.05', rely='0.34', x='0', y='0')
        self.load_time_interval_start_entry = ttk.Entry(self.Forcasting_Tab_Frame)
        self.load_time_interval_start_entry_text = tk.StringVar(value='HH-MM-SS')
        self.load_time_interval_start_entry.configure(textvariable=self.load_time_interval_start_entry_text)
        _text_ = '''HH-MM-SS'''
        self.load_time_interval_start_entry.delete('0', 'end')
        self.load_time_interval_start_entry.insert('0', _text_)
        self.load_time_interval_start_entry.place(anchor='nw', relwidth='0.075', relx='0.05', rely='0.5', x='0', y='0')
        self.load_time_interval_TO_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.load_time_interval_TO_lable.configure(background='#c0c0c0', font='{Arial} 12 {bold}', foreground='#000000', text='To:')
        self.load_time_interval_TO_lable.place(anchor='nw', relx='0.2', rely='0.45', x='0', y='0')
        self.load_time_interval_end_entry = ttk.Entry(self.Forcasting_Tab_Frame)
        self.load_time_interval_end_entry_text = tk.StringVar(value='HH-MM-SS')
        self.load_time_interval_end_entry.configure(textvariable=self.load_time_interval_end_entry_text)
        _text_ = '''HH-MM-SS'''
        self.load_time_interval_end_entry.delete('0', 'end')
        self.load_time_interval_end_entry.insert('0', _text_)
        self.load_time_interval_end_entry.place(anchor='nw', relwidth='0.075', relx='0.3', rely='0.5', x='0', y='0')
        self.Load_Time_Interval_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.Load_Time_Interval_lable_text = tk.StringVar(value='Load Time Interval')
        self.Load_Time_Interval_lable.configure(background='#595959', font='{Arial} 12 {bold italic}', foreground='#ffffff', text='Load Time Interval')
        self.Load_Time_Interval_lable.configure(textvariable=self.Load_Time_Interval_lable_text)
        self.Load_Time_Interval_lable.place(anchor='nw', relheight='0.03', relwidth='0.16', relx='0.14', rely='0.41', x='0', y='0')
        self.Learning_Parameters_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.Learning_Parameters_lable.configure(background='#595959', font='{Arial} 16 {bold underline}', foreground='#ffffff', text='Learning Parameters')
        self.Learning_Parameters_lable.place(anchor='nw', relwidth='0.25', relx='0.675', rely='0.1', x='0', y='0')
        self.Eta_start_Entry = ttk.Entry(self.Forcasting_Tab_Frame)
        self.Eta_start_Entry_text = tk.StringVar(value='Eta Start')
        self.Eta_start_Entry.configure(font='{Arial} 14 {}', textvariable=self.Eta_start_Entry_text)
        _text_ = '''Eta Start'''
        self.Eta_start_Entry.delete('0', 'end')
        self.Eta_start_Entry.insert('0', _text_)
        self.Eta_start_Entry.place(anchor='nw', relwidth='0.09', relx='0.58', rely='0.2', x='0', y='0')
        self.Eta_End_Entry = ttk.Entry(self.Forcasting_Tab_Frame)
        self.Eta_End_Entry_Text = tk.StringVar(value='Eta End')
        self.Eta_End_Entry.configure(font='{Arial} 14 {}', textvariable=self.Eta_End_Entry_Text)
        _text_ = '''Eta End'''
        self.Eta_End_Entry.delete('0', 'end')
        self.Eta_End_Entry.insert('0', _text_)
        self.Eta_End_Entry.place(anchor='nw', relwidth='0.09', relx='0.825', rely='0.2', x='0', y='0')
        self.Forcast_Interval_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.Forcast_Interval_lable.configure(background='#595959', font='{Arial} 16 {bold underline}', foreground='#ffffff', text='Forcast Interval')
        self.Forcast_Interval_lable.place(anchor='nw', relwidth='0.2', relx='0.435', rely='0.6', x='0', y='0')
        self.No_of_Forcast_Days_entry = ttk.Entry(self.Forcasting_Tab_Frame)
        self.No_of_Forcast_Days_entry_text = tk.StringVar(value='No. of Forecast Days')
        self.No_of_Forcast_Days_entry.configure(font='{Arial} 12 {}', textvariable=self.No_of_Forcast_Days_entry_text)
        _text_ = '''No. of Forecast Days'''
        self.No_of_Forcast_Days_entry.delete('0', 'end')
        self.No_of_Forcast_Days_entry.insert('0', _text_)
        self.No_of_Forcast_Days_entry.place(anchor='nw', relwidth='0.175', relx='0.65', rely='0.7', x='0', y='0')
        self.Separator_2 = ttk.Separator(self.Forcasting_Tab_Frame)
        self.Separator_2.configure(orient='horizontal')
        self.Separator_2.place(anchor='nw', relheight='0.01', relwidth='1', relx='0.001', rely='0.55', x='0', y='0')
        self.Separator_3 = ttk.Separator(self.Forcasting_Tab_Frame)
        self.Separator_3.configure(orient='vertical')
        self.Separator_3.place(anchor='nw', relheight='0.51', relwidth='0.01', relx='0.52', rely='0.05', x='0', y='0')
        self.Learn_Start_Date_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.Learn_Start_Date_lable.configure(background='#595959', font='{Arial} 12 {bold italic}', foreground='#ffffff', text='Learn Start Date')
        self.Learn_Start_Date_lable.place(anchor='nw', relheight='0.03', relwidth='0.16', relx='0.56', rely='0.28', x='0', y='0')
        self.Learn_end_date_lable = ttk.Label(self.Forcasting_Tab_Frame)
        self.Learn_end_date_lable.configure(background='#595959', font='{Arial} 12 {bold italic}', foreground='#ffffff', text='Learn end Date')
        self.Learn_end_date_lable.place(anchor='nw', relheight='0.03', relwidth='0.16', relx='0.81', rely='0.28', x='0', y='0')
        self.Learn_end_date_entry = ttk.Entry(self.Forcasting_Tab_Frame)
        self.Learn_end_date_entry_text = tk.StringVar(value='YYYY-MM-DD')
        self.Learn_end_date_entry.configure(font='{Arial} 12 {}', textvariable=self.Learn_end_date_entry_text)
        _text_ = '''YYYY-MM-DD'''
        self.Learn_end_date_entry.delete('0', 'end')
        self.Learn_end_date_entry.insert('0', _text_)
        self.Learn_end_date_entry.place(anchor='nw', relwidth='0.125', relx='0.815', rely='0.4', x='0', y='0')
        self.forcast_start_date_calender = CalendarFrame(self.Forcasting_Tab_Frame)
        self.forcast_start_date_calender.configure(relief='flat')
        # TODO - self.forcast_start_date_calender: code for custom option 'firstweekday' not implemented.
        # TODO - self.forcast_start_date_calender: code for custom option 'month' not implemented.
        self.forcast_start_date_calender.place(anchor='nw', relheight='0.2', relwidth='0.175', relx='0.15', rely='0.625', x='0', y='0')
        self.Learn_Start_date_calender = CalendarFrame(self.Forcasting_Tab_Frame)
        self.Learn_Start_date_calender.configure(relief='flat')
        # TODO - self.Learn_Start_date_calender: code for custom option 'firstweekday' not implemented.
        # TODO - self.Learn_Start_date_calender: code for custom option 'month' not implemented.
        self.Learn_Start_date_calender.place(anchor='nw', relheight='0.2', relwidth='0.175', relx='0.55', rely='0.325', x='0', y='0')
        self.Learn_end_date_calender = CalendarFrame(self.Forcasting_Tab_Frame)
        self.Learn_end_date_calender.configure(relief='flat')
        # TODO - self.Learn_end_date_calender: code for custom option 'firstweekday' not implemented.
        # TODO - self.Learn_end_date_calender: code for custom option 'month' not implemented.
        self.Learn_end_date_calender.place(anchor='nw', relheight='0.2', relwidth='0.175', relx='0.775', rely='0.325', x='0', y='0')
        self.Forcasting_Tab_Frame.configure(height='200', width='200')
        self.Forcasting_Tab_Frame.place(anchor='nw', x='0', y='0')
        self.Notebook_1.add(self.Forcasting_Tab_Frame, text='Forcasting')
        self.Attack_classification_frame = ttk.Frame(self.Notebook_1)
        self.Attack_classification_entry = ttk.Label(self.Attack_classification_frame)
        self.Attack_classification_entry.configure(background='#535353', font='{Arial} 16 {bold italic}', foreground='#ffffff', text='Attack classification')
        self.Attack_classification_entry.place(anchor='nw', relheight='0.05', relwidth='1.0', x='0', y='0')
        self.Attack_classification_canvas = tk.Canvas(self.Attack_classification_frame)
        self.Attack_classification_canvas.configure(background='#595959')
        self.Attack_classification_canvas.place(anchor='nw', relheight='0.86', relwidth='1.0', relx='0', rely='0.05', x='0', y='0')
        self.attack_cls_button = ttk.Button(self.Attack_classification_frame)
        self.attack_cls_button.configure(text='classifie attack')
        self.attack_cls_button.place(anchor='nw', relheight='0.075', relwidth='0.2', relx='0.4', rely='0.4', x='0', y='0')
        self.attack_cls_button.configure(command=self.cls_attacks)
        self.attack_cls_Pathchooserinput = PathChooserInput(self.Attack_classification_frame)
        self.attack_cls_Pathchooserinput_text = tk.StringVar(value='')
        self.attack_cls_Pathchooserinput.configure(textvariable=self.attack_cls_Pathchooserinput_text, type='file')
        self.attack_cls_Pathchooserinput.place(anchor='nw', relheight='0.055', relwidth='1.0', relx='0.001', rely='0.825', x='0', y='0')
        self.Attack_classification_path_lable = ttk.Label(self.Attack_classification_frame)
        self.Attack_classification_path_lable.configure(font='{Arial} 12 {}', text='Attack classification File Path')
        self.Attack_classification_path_lable.place(anchor='nw', relwidth='1.0', relx='0.001', rely='0.8', x='0', y='0')
        self.attack_cls_Pathchooserinput_output_lable = ttk.Label(self.Attack_classification_frame)
        self.attack_cls_Pathchooserinput_output_lable_etxt = tk.StringVar(value='Current Load Data File Selected: Warning, No File Selected')
        self.attack_cls_Pathchooserinput_output_lable.configure(font='{Arial} 14 {}', text='Current Load Data File Selected: Warning, No File Selected', textvariable=self.attack_cls_Pathchooserinput_output_lable_etxt)
        self.attack_cls_Pathchooserinput_output_lable.place(anchor='nw', relheight='0.04', relwidth='1.0', relx='0.001', rely='0.87', x='0', y='0')
        self.attack_cls_message = tk.Message(self.Attack_classification_frame)
        self.attack_cls_message_text = tk.StringVar(value='No Attack Loads Found / Selected')
        self.attack_cls_message.configure(background='#595959', font='{Arial} 11 {bold}', foreground='#ffffff', justify='center')
        self.attack_cls_message.configure(text='No Attack Loads Found / Selected', textvariable=self.attack_cls_message_text)
        self.attack_cls_message.place(anchor='nw', relheight='0.3', relwidth='0.4', relx='0.3', rely='0.1', x='0', y='0')
        self.Attack_classification_frame.configure(height='200', width='200')
        self.Attack_classification_frame.place(anchor='nw', x='0', y='0')
        self.Notebook_1.add(self.Attack_classification_frame, text='Attack classification')
        self.Notebook_1.configure(height='200', width='200')
        self.Notebook_1.place(anchor='nw', relheight='1.0', relwidth='1.0', x='0', y='0')
        self.Line_1_Lable = ttk.Label(self.Toplevel_1)
        self.Line_1_TEXT = tk.StringVar(value='.')
        self.Line_1_Lable.configure(background='#000000', font='{Arial} 12 {bold}', foreground='#00ff00', text='.')
        self.Line_1_Lable.configure(textvariable=self.Line_1_TEXT)
        self.Line_1_Lable.place(anchor='nw', relwidth='1.0', rely='0.97', x='0', y='0')
        self.Line_2_Lable = ttk.Label(self.Toplevel_1)
        self.Line_2_TEXT = tk.StringVar(value='..')
        self.Line_2_Lable.configure(background='#000000', font='{Arial} 12 {bold}', foreground='#00ff00', text='..')
        self.Line_2_Lable.configure(textvariable=self.Line_2_TEXT)
        self.Line_2_Lable.place(anchor='nw', relwidth='1.0', rely='0.94', x='0', y='0')
        self.Line_3_Lable = ttk.Label(self.Toplevel_1)
        self.Line_3_TEXT = tk.StringVar(value='...')
        self.Line_3_Lable.configure(background='#000000', font='{Arial} 12 {bold}', foreground='#00ff00', text='...')
        self.Line_3_Lable.configure(textvariable=self.Line_3_TEXT)
        self.Line_3_Lable.place(anchor='nw', relwidth='1.0', rely='0.91', x='0', y='0')
        self.Sizegrip = ttk.Sizegrip(self.Toplevel_1)
        self.Sizegrip.place(anchor='nw', relx='0.983', rely='0.975', x='0', y='0')
        self.Toplevel_1.configure(background='#000000', height='700', highlightbackground='#000000', highlightcolor='#000000')
        self.Toplevel_1.configure(width='900')
        self.attack_cls = None

        # Main widget
        self.mainwindow = self.Toplevel_1

    def Info_Gained_button_pressed(self):
        try:
            print("|##---##| Button Activated  :  Info Gained button pressed")
            file = self.Data_file_path_chooser_TEXT.get()
            self.Data_File_selected_TEXT.set("Current File Selected: "+str(file))
            start_date = self.data_start_date_calender._selection
            if start_date != None:
                start_date = convert_date_format(start_date)

            end_date = self.data_end_date_calender._selection
            if end_date != None:
                end_date =  convert_date_format(end_date)


            imported_data , Times_list = import_prev_load_data_for_training_or_analysis(file, PROJECT_PATH,start_date, end_date,)
            if imported_data != None:
                newline = "##---##|: Performing Info Gained Analysis on "+str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                info_gained_data_analysis(imported_data, file,PROJECT_PATH)
                print("|##---##| SYSTEM INFO      : Saving Info Gained Analysis results")    
                newline = "##---##|: Info Gained Analysis Completed & saving to Results file"
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)  
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if "final_data.xlsx" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected is not final_data.xlsx, produced from the forcast")

        except Exception as e:
            newline = "Error: " + str(e)
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set(newline)


    def Correlation_button_pressed(self):
        print("|##---##| Button Activated  :  Correlation button pressed")

        try:
            file = self.Data_file_path_chooser_TEXT.get()
            self.Data_File_selected_TEXT.set("Current File Selected: "+str(os.path.basename(file)))
            start_date = self.data_start_date_calender._selection
            if start_date != None:
                start_date = convert_date_format(start_date)

            end_date = self.data_end_date_calender._selection
            if end_date != None:
                end_date =  convert_date_format(end_date)

            imported_data , Times_list= import_prev_load_data_for_training_or_analysis(file, PROJECT_PATH,start_date, end_date,)
            if imported_data != None:
                newline = "##---##|: Performing Correlation Analysis on "+str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                Correlation_data_analysis(imported_data, file,PROJECT_PATH)
                print("|##---##| SYSTEM INFO      : Saving Correlation Analysis results")
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("##---##|: Correlation Analysis Completed" + str(os.path.basename(file)))
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if "final_data.xlsx" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected is not final_data.xlsx, produced from the forcast")

        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))

    def Data_Summery_button_pressed(self):
        print("|##---##| Button Activated  :  Data summery button pressed")
        try:

            file = self.Data_file_path_chooser_TEXT.get()
            self.Data_File_selected_TEXT.set("Current File Selected: "+str(file))
            newline = "##---##|: Performing Correlation Analysis on "+str(os.path.basename(file))
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set(newline) 
            start_date = self.data_start_date_calender._selection
            if start_date != None:
                start_date = convert_date_format(start_date)

            end_date = self.data_end_date_calender._selection
            if end_date != None:
                end_date =  convert_date_format(end_date)
            imported_data  , Times_list = import_prev_load_data_for_training_or_analysis(file, PROJECT_PATH,start_date, end_date,)
            if imported_data != None:

                data_summery(imported_data, file,PROJECT_PATH)

                print("|##---##| SYSTEM INFO      : Saving Data Summery results")
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("##---##|: Data Summery Completed on " + str(os.path.basename(file)))
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if "final_data.xlsx" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected is not final_data.xlsx, produced from the forcast")

        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))
        

    def split_data_file(self):
        print("|##---##| Button Activated  :  Split data button pressed")
        try:
            file = self.Data_file_path_chooser_TEXT.get()
            self.Data_File_selected_TEXT.set("Current File Selected: "+str(file))
            start_date = self.data_start_date_calender._selection
            if start_date != None:
                start_date = convert_date_format(start_date)
            end_date = self.data_end_date_calender._selection
            if end_date != None:
                end_date =  convert_date_format(end_date)
            imported_data , Times_list = import_prev_load_data_for_training_or_analysis(file, PROJECT_PATH,start_date, end_date,)
            if imported_data != None:
                first_group, second_group = split_data_file(imported_data, float(self.test_data_percentage_variable.get()) )
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Split Used:" +str(self.test_data_percentage_variable.get()))
                save_arff_file(first_group, "Split_1_"+str(os.path.basename(file)))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("First Split File saved as Split_1_" +str(os.path.basename(file)))
                save_arff_file(second_group, "Split_2_"+str(os.path.basename(file)))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Second Split File saved as Split_2_" +str(os.path.basename(file)))
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if "final_data.xlsx" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected is not final_data.xlsx, produced from the forcast")

        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))


    def Build_Classifier(self):
        try:
            print("|##---##| Button Activated  :  Build classifier button pressed")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Building classifier")
            bagging = False
            AdaBoostM1 = False

            if self.Bagging_Checkbutton_value.get() == 1:
                bagging == True

            if self.AdaBoostM1_Checkbutton_value.get() == 1:
                AdaBoostM1 == True

            file = self.Traning_data_Pathchooserinput_TEXT.get()
            self.Training_Data_File_selected_TEXT.set("Current File Selected: "+str(file))
            start_date = None
            end_date = None
            imported_data, self.Times_list = import_prev_load_data_for_training_or_analysis(file, PROJECT_PATH, start_date, end_date,)
            if imported_data != None:
                self.Training_data = imported_data

                Classifier_algorithm_choice = self.Classifier_algorithm_choice.get()
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("classifier choice: "+ str(Classifier_algorithm_choice))
                self.test_date_cls_selected_lable_text.set("classifier Has been created")
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("classifier Has been built")
                self.test_date_cls_selected_lable_text.set("A Classifier Has been created!!")
                self.eval_tab_cls_selection_text.set("A Classifier Has been created!!")
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if "final_data.xlsx" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected is not final_data.xlsx, produced from the forcast")
        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))

    def Bagging_Checkbutton_activated(self):
        self.Line_3_TEXT.set(self.Line_2_TEXT.get())
        self.Line_2_TEXT.set(self.Line_1_TEXT.get())
        self.Line_1_TEXT.set("Bagging will be applied to the classifier")


    def AdaBoostM1_Checkbutton_activated(self):
        self.Line_3_TEXT.set(self.Line_2_TEXT.get())
        self.Line_2_TEXT.set(self.Line_1_TEXT.get())
        self.Line_1_TEXT.set("AdaBoostM1 will be applied to the classifier")


    def Classifier_algorithm_picker_choice(self, option):
        self.Line_3_TEXT.set(self.Line_2_TEXT.get())
        self.Line_2_TEXT.set(self.Line_1_TEXT.get())
        self.Line_1_TEXT.set("Classifier algorithm selected: "+ str(option))


    def Test_test_data(self):
        print("|##---##| Button Activated  :  test data button pressed")
        try:
            file = self.test_data_path_chosser_TEXT.get()
            self.Testing_Data_File_selected_TEXT.set("Current File Selected: "+str(os.path.basename(file)))
            start_date = None
            end_date = None
            imported_data = convert_forcast_to_test_data(file, PROJECT_PATH, self.Times_list)
            
            if imported_data != None:
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Performing anomalious analysis on data")
                Classifier_algorithm_choice= self.Classifier_algorithm_choice.get()
                bagging = False
                AdaBoostM1 = False

                if self.Bagging_Checkbutton_value.get() == 1:
                    bagging == True

                if self.AdaBoostM1_Checkbutton_value.get() == 1:
                    AdaBoostM1 == True

                attack_cls_att = self.attack_cls_att
                if attack_cls_att == None:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("Attack classifier not created yet")
                    return

                attack_cls = self.attack_cls
                anom_forcasts_count = find_anom_forcasts_by_class_att(Classifier_algorithm_choice, imported_data, self.Training_data, file, bagging, AdaBoostM1, attack_cls_att, attack_cls,PROJECT_PATH)
                df = pd.read_csv(PROJECT_PATH+"/"+"Results/Results_Tolerance_results_file.csv")
                fig = px.scatter(df, x = 'Load', y = ["Diffrence" ], title="Load vs Differance")
                fig.show()
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Analysis Completed, results file saved")
                if anom_forcasts_count == -1 :
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("An Attack has been deteched, check results file for more imformation ")
                else:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("Total Anomalious Forcasts found: "+str(anom_forcasts_count))

                if anom_forcasts_count == 0:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No Anomalious Forcasts Found!")
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if "forecast_data.xlsx" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected is not forecast_data.xlsx, produced from the forcast")
        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))
        

    def perform_Quick_eval(self):
        print("|##---##| Button Activated  :  quick evaluation button pressed")
        try:
            file = self.Evaluation_data_file_path_chosser_TEXT.get()
            self.Evaluation_Data_File_selected_TEXT.set("Current File Selected: "+str(os.path.basename(file)))
            imported_data = convert_forcast_to_test_data(file, PROJECT_PATH, self.Times_list)
            if imported_data != None:
                Classifier_algorithm_choice= self.Classifier_algorithm_choice.get()
                bagging = False
                AdaBoostM1 = False

                if self.Bagging_Checkbutton_value.get() == 1:
                    bagging == True

                if self.AdaBoostM1_Checkbutton_value.get() == 1:
                    AdaBoostM1 == True

                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Performing Quick Evaluation on " + str(Classifier_algorithm_choice))
                quick_eval(imported_data, Classifier_algorithm_choice, file, bagging, AdaBoostM1, self.Training_data, file,PROJECT_PATH)
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Quick Evaluation Completed, results file saved")         
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if "forecast_data.xlsx" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected is not forecast_data.xlsx, produced from the forcast")
        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: "+ str(e))



    def perform_ten_fold_eval(self):
        print("|##---##| Button Activated  :  Ten Fold Cross eval button pressed")
        try:
            file = self.Evaluation_data_file_path_chosser_TEXT.get()
            self.Evaluation_Data_File_selected_TEXT.set("Current File Selected: "+str(os.path.basename(file)))
            imported_data = convert_forcast_to_test_data(file, PROJECT_PATH, self.Times_list)
            if imported_data != None:
                Classifier_algorithm_choice= self.Classifier_algorithm_choice.get()
                bagging = False
                AdaBoostM1 = False

                if self.Bagging_Checkbutton_value.get() == 1:
                    bagging == True

                if self.AdaBoostM1_Checkbutton_value.get() == 1:
                    AdaBoostM1 == True

                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Performing Ten Fold cross Evaluation on " + str(Classifier_algorithm_choice))
                ten_fold_evaal(imported_data, Classifier_algorithm_choice, file, bagging, AdaBoostM1, self.Training_data, file,PROJECT_PATH)
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Ten Fold cross Evaluation Completed, results file saved")         
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if "forecast_data.xlsx" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected is not forecast_data.xlsx, produced from the forcast")
        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: "+ str(e))
 


    def perform_results_visulisation(self):
        print("|##---##| Button Activated  :  Results visulisation button pressed")
        try:
            file = self.vis_results_file_path_chooser_TEXT.get()
            self.vis_results_file_path_chooser_TEXT.set("Current File Selected: "+str(os.path.basename(file)))
            
            if file != None:
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Performing Results visulisation ")

                if "attack_cls_Results_" in os.path.basename(file):
                    df = pd.read_csv(file)
                    fig = px.bar(df, x = "test data", y = ["attack event detected"], title='Attack classification on suspisous instances')
                    fig.show()

                elif "Results_Tolerance" in os.path.basename(file):
                    df = pd.read_csv(file)
                    fig = px.scatter(df, x = 'Load', y = ["Diffrence"], title='Load plotted with predictionsction')
                    fig.show()

                elif "Data_summery_" in os.path.basename(file):
                    df = pd.read_csv(file)
                    fig = px.line(df, x = 'Name', y = ['Type', "Nom",'Int', "Real",'Missing', "Unique"], title='Data summery')
                    fig.show()

                elif "Correlation" in os.path.basename(file):
                    df = pd.read_csv(file)
                    fig = px.scatter(df, x = 'Attribute', y = ['Correlation_Score'], title='Correlation analysis')
                    fig.show()

                elif "Evaluation" in os.path.basename(file):
                    df = pd.read_csv(file)

                    fig = px.scatter(df, x = ["numInstances","mean_absolute_error","mean_prior_absolute_error","root_mean_squared_error", "root_mean_prior_squared_error", "root_relative_squared_error"],y = ["evaluation_score"], title='Evaluation')
                    fig.show()

                elif "InfoGain_" in file:
                    df = pd.read_csv(file)
                    fig = px.scatter(df, x = 'Attribute', y = ['InfoGain_Score'], title='Info gained analysis')
                    fig.show()
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Results visulisation Completed")         
            else:
                newline = "Failed to import " + str(os.path.basename(file))
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set(newline)
        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))


    def perform_data_Visualize(self):
        try:
            print("start date selected: " + str(self.vis_start_date_calender._selection))
            print("End date selected: " + str(self.vis_end_date_calender._selection))
            print("|##---##| Button Activated  :  Data visulisation button pressed")
            file = self.vis_data_file_path_chooser_TEXT.get()
            self.Current_Data_File_Selected_lable_text.set("Current File Selected: "+str(file))
            start_date = self.vis_start_date_calender._selection
            if start_date != None:
                start_date = convert_date_format(start_date)

            end_date = self.vis_end_date_calender._selection
            if end_date != None:
                end_date =  convert_date_format(end_date)
            print("|##---##| Date Start  :  "+str(start_date))
            print("|##---##| Date End  :  "+str(end_date))

            if "final_data.xlsx" in file:
                imported_data = pd.DataFrame(pd.read_excel(file))

                date_list = []
                first_coloum = []
                for insta in imported_data.iloc[:, 0]:
                    first_coloum.append(insta)
                for insta in imported_data:
                    if insta != 'Unnamed: 0':
                        date_list.append(insta)

                fig = px.line(imported_data, x =first_coloum , y = date_list,title='data plotted ')
                fig.show()
            if "forecast_data.xlsx" in file:
                imported_data = pd.DataFrame(pd.read_excel(file).transpose())

                fig = px.line(imported_data,title='data plotted ')
                fig.show()

        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))
     

    def Execute_forcast(self):
        print("|##---##| Button Activated  :  Forcasting button pressed")

        load_data_folder_path = self.Load_data_Dir_Pathchooserinput_text.get()+"/"
        if load_data_folder_path == "":
            print("No Load Dir found, please select one")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: No Load Dir has been entered, please select one")
            return
        order_related_severity_matrix_path = self.severity_factor_Pathchooserinput_text.get()
        if order_related_severity_matrix_path == "":
            print("No order related severity matrix file found, please select one")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: no order related severity matrix file entered, please select one")
            return
        forecast_start_date = self.forcast_start_date_calender._selection
        if forecast_start_date == None:
            print("No forcast start date")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: No Forcast start date has been provided, please select one")
            return
        forecast_start_date = convert_date_format(forecast_start_date)
        forecast_days = self.No_of_Forcast_Days_entry_text.get()
        if "No. of Forecast Days" in forecast_days:
            print("Number of days for forcast not entered")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: Number of days for forcast not entered")
            return
        try:
            forecast_days = int(forecast_days)
        except Exception as e:
            print("Error in forcast days")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: forcast days is not a number")
            return
        start_date = self.Learn_Start_date_calender._selection
        if start_date == None:
            start_date = ""
        else:
            start_date = convert_date_format(start_date)

        end_date = self.Learn_end_date_calender._selection
        if end_date == None:
            end_date = ""
        else:
            end_date =  convert_date_format(end_date)

        eta_start = self.Eta_start_Entry_text.get()
        if eta_start == "Eta Start":
            eta_start = 2

        try:
            eta_start = int(eta_start)
        except Exception as e:
            print("ETA start not definded")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: No ETA start has been provided, please select one")
            return
            
        eta_end = self.Eta_End_Entry_Text.get()
        if eta_end == "Eta End":
            eta_end = 7
        try:
            eta_end = int(eta_end)
        except Exception as e:
            print("ETA End not definded")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error: No ETA End has been provided, please select one")
            return

        time_interval_start = self.load_time_interval_start_entry_text.get()
        time_interval_end = self.load_time_interval_end_entry_text.get()
        time_interval_start_list  = time_interval_start.split("-")
        time_interval_end_list = time_interval_end.split("-")
        start = 144
        end = 144
        if time_interval_start_list[1] == "MM":
            start=144
        if time_interval_end_list[1] == "MM":
            end=144

        if time_interval_start_list[1] == "00":
            start=0
        if time_interval_end_list[1] == "10":
            end=144
        if time_interval_end_list[1] == "01":
            end=1440
        if time_interval_end_list[1] == "1":
            end=1440
        if time_interval_end_list[1] == "5":
            end=288
        if time_interval_end_list[1] == "05":
            end=288


        interval = start + end 
        try:
            print("Starting forcasting")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Starting forcasting for " + str(forecast_start_date) + " For " + str(forecast_days)+" days's")

            Perform_forcast(load_data_folder_path, order_related_severity_matrix_path, forecast_start_date, forecast_days, start_date, end_date, eta_start, eta_end, interval)
            print("forcasting Complete")
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Forcasting Complete, forcast saved in excel worksheet forecast_data.xlsx")

            imported_data = pd.DataFrame(pd.read_excel(PROJECT_PATH+"/Load_Data/Forcasting/forecast_data.xlsx").transpose())

            fig = px.line(imported_data,title='data plotted ')
            fig.show()
        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))

    def cls_attacks(self):
        print("|##---##| Button Activated  :  Attack classification button pressed")
        try:
            bagging_choice = False
            AdaBoostM1_choice = False

            file = self.attack_cls_Pathchooserinput_text.get()
            self.attack_cls_Pathchooserinput_output_lable_etxt.set("Current File Selected: "+str(file))

            imported_data = get_arff_csv_file(file)
            if imported_data != None:

                attack_cls, self.attack_cls_att  = build_cls("weka.classifiers.meta.RotationForest", imported_data, bagging_choice, AdaBoostM1_choice)
                self.attack_cls = attack_cls
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("Attack classifier Produced")
                self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                self.Line_1_TEXT.set("File " + str(file) + " Was used")
                self.attack_cls_message_text.set("Attack classifier Created")
            else:
                if file == "":
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("No file Selected")
                if ".csv" not in file:
                    self.Line_3_TEXT.set(self.Line_2_TEXT.get())
                    self.Line_2_TEXT.set(self.Line_1_TEXT.get())
                    self.Line_1_TEXT.set("File selected May not be Attack classification data")

        except Exception as e:
            self.Line_3_TEXT.set(self.Line_2_TEXT.get())
            self.Line_2_TEXT.set(self.Line_1_TEXT.get())
            self.Line_1_TEXT.set("Error"+ str(e))
        

    def run(self):
        self.mainwindow.mainloop()


if __name__ == '__main__':
    try:
        jvm.start(packages=True,max_heap_size="30G")
        app = Load_forcasting_app()
        app.run()
    except Exception as e:
        raise e
    finally:
        jvm.stop()
    

