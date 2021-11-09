import os
import csv
import numpy as np
from weka.core.converters import Loader
from weka.classifiers import Classifier, SingleClassifierEnhancer, MultipleClassifiersCombiner, Evaluation
from weka.core.dataset import Instances
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.filters import Filter
from weka.core.classes import Random
import pandas as pd
import plotly.express as px



def convert_date_format(date):
    print("|##---##| Converting date :  "+str(date))
    year = str(date[0])
    month = None
    if date[1] <= 9:
        month = "0"+str(date[1])
    else:
        month = str(date[1])
    day = None
    if date[2] <= 9:
        day = "0"+str(date[2])
    else:
        day = str(date[2])
    new_date = year+"-"+month+"-"+day
    return new_date


def get_arff_csv_file(file_name):
    imported_data = None
    try:
        print("|##---##| Importing File : "+ str(os.path.basename(file_name)))
        if ".csv" in file_name:
            loader = Loader(classname="weka.core.converters.CSVLoader")
            imported_data = loader.load_file(file_name)
            return imported_data   

        if ".arff" in file_name:
            loader = Loader(classname="weka.core.converters.ArffLoader")
            imported_data = loader.load_file(file_name)
            return imported_data
    except Exception as e:
        return imported_data


def info_gained_data_analysis(data,test_file_name,PROJECT_PATH):
    print("|##---##| Data Analysis        :   Starting info gained data analysis")

    data.randomize(Random(42))
    NumericToNominal_filter= Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal")
    NumericToNominal_filter.inputformat(data)
    data = NumericToNominal_filter.filter(data)
    num_of_att = data.num_attributes
    data.class_index = 3 + int((num_of_att-3)/2)
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation("weka.attributeSelection.InfoGainAttributeEval")
    InfoGain = AttributeSelection()
    InfoGain.ranking(True)
    InfoGain.folds(2)
    InfoGain.crossvalidation(True)
    InfoGain.seed(42)
    InfoGain.search(search)
    InfoGain.evaluator(evaluation)
    InfoGain.select_attributes(data)
    att_list =[]
    for i in range(data.num_attributes -1):
        att_list.append([i,data.attribute(i).name])

    with open(PROJECT_PATH+"/"+"Results/InfoGain_"+str(os.path.basename(test_file_name))+".csv", 'wt', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        info_gained_results = np.array(InfoGain.ranked_attributes).tolist()
        writer.writerow(["Attribute", "InfoGain_Score"])
        for row in info_gained_results:
            for att in att_list:
                if (str(att[0])+".0") == str(row[0]):
                    writer.writerow([att[1], row[1]])
    df = pd.read_csv(PROJECT_PATH+"/"+"Results/InfoGain_"+str(os.path.basename(test_file_name))+".csv")
    fig = px.scatter(df, x = 'Attribute', y = ['InfoGain_Score'], title='Info gained analysis')
    fig.show()

def split_data_file(data, split):
    data.randomize(Random(42))
    first_group, second_group = data.train_test_split(split)
    return first_group, second_group


def Correlation_data_analysis(data,test_file_name,PROJECT_PATH):
    print("|##---##| Data Analysis         :   Starting correlation data analysis")

    data.randomize(Random(42))

    NumericToNominal_filter= Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal")
    NumericToNominal_filter.inputformat(data)
    data = NumericToNominal_filter.filter(data)

    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation("weka.attributeSelection.CorrelationAttributeEval")
    Correlation = AttributeSelection()
    Correlation.ranking(True)
    Correlation.folds(2)
    Correlation.crossvalidation(True)
    Correlation.seed(42)
    Correlation.search(search)
    Correlation.evaluator(evaluation)
    Correlation.select_attributes(data)
    att_list =[]
    for i in range(data.num_attributes -1):
        att_list.append([i,data.attribute(i).name])

    with open(PROJECT_PATH+"/"+"Results/CorrelationAttributeEval_"+str(os.path.basename(test_file_name))+".csv", 'wt', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        Correlation_results = np.array(Correlation.ranked_attributes).tolist()
        writer.writerow(["Attribute", "Correlation_Score"])
        for row in Correlation_results:
            for att in att_list:
                if (str(att[0])+".0") == str(row[0]):
                    writer.writerow([att[1], row[1]])
    df = pd.read_csv(PROJECT_PATH+"/"+"Results/CorrelationAttributeEval_"+str(os.path.basename(test_file_name))+".csv")
    fig = px.scatter(df, x = 'Attribute', y = ['Correlation_Score'], title='Correlation analysis')
    fig.show()


def data_summery(data, test_file_name,PROJECT_PATH):
    print("|##---##| Data Analysis         :   Starting data summery")
    with open(PROJECT_PATH+"/"+"Results/Data_summery_"+str(os.path.basename(test_file_name))+".csv", 'wt', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        summery_list = data.summary(data).split()
        writer.writerow([summery_list[9],summery_list[10],summery_list[11],summery_list[12],summery_list[13],summery_list[14],summery_list[15],summery_list[16]])
        for i in range(18,len(summery_list)-11,13):
            writer.writerow([summery_list[i],summery_list[i+1],summery_list[i+2],summery_list[i+3],summery_list[i+4],summery_list[i+5]+summery_list[i+6]+summery_list[i+7],summery_list[i+8]+summery_list[i+9]+summery_list[i+10],summery_list[i+11]])
    df = pd.read_csv(PROJECT_PATH+"/"+"Results/Data_summery_"+str(os.path.basename(test_file_name))+".csv")
    fig = px.line(df, x = 'Name', y = ['Type', "Nom",'Int', "Real",'Missing', "Unique"], title='Data summery')
    fig.show()


def build_cls(class_name, data, bagging_choice, AdaBoostM1_choice):
    print("|##---##| Building Classifier         :   Starting data summery")
    data.class_is_last()
    
    if (bagging_choice == True) and (AdaBoostM1_choice ==True) :
        meta_vote = vote(class_name) 
        print("|##---##| CLASSIFIER :   Applying VOTE Classifier")
        meta_vote.build_classifier(filtered_data)
        return meta_vote, data.class_attribute

    if (bagging_choice == True) and (AdaBoostM1_choice ==False) :
        meta_vote = bagging(class_name) 
        print("|##---##| CLASSIFIER :   Applying Bagging Classifier")
        meta_vote.build_classifier(filtered_data)
        return meta_vote, data.class_attribute
    if (bagging_choice == False) and (AdaBoostM1_choice ==True) :
        meta_AdaBoostM1 = AdaBoostM1(class_name) 
        meta_AdaBoostM1.build_classifier(data)
        print("|##---##| CLASSIFIER :   Applying AdaBoostM1 Classifier")
        return meta_AdaBoostM1, data.class_attribute
    if (bagging_choice == False) and (AdaBoostM1_choice ==False) :
        normal_cls = Classifier(classname=class_name, options=[])
        normal_cls.build_classifier(data)
        return normal_cls, data.class_attribute


def vote(class_name):
        meta_bagging = bagging(class_name)
        meta_AdaBoostM1 = AdaBoostM1(class_name)
        MultipleClassifiersCombiner_meta = MultipleClassifiersCombiner(classname=".meta.Vote")
        classifiers_to_combine = [meta_bagging, meta_AdaBoostM1]
        MultipleClassifiersCombiner_meta.classifiers = classifiers_to_combine
        return  MultipleClassifiersCombiner_meta


def bagging(class_name):
        meta = SingleClassifierEnhancer(classname="weka.classifiers.meta.Bagging")
        meta.classifier = Classifier(classname=class_name, options=[])
        return meta


def AdaBoostM1(class_name):
    meta = SingleClassifierEnhancer(classname=".AdaBoostM1")
    meta.classifier = Classifier(classname=class_name, options=[])
    return meta


def Find_rel_tolerance(act):
    diff = abs(act/25)
    return diff


def create_cls_by_class_att(class_name, training_data, bagging_choice, AdaBoostM1_choice, class_att):
    training_data.class_index = class_att

    if (bagging_choice == True) and (AdaBoostM1_choice ==True) :
        meta_vote = vote(class_name) 
        print("|##---##| CLASSIFIER :   Applying VOTE Classifier")
        meta_vote.build_classifier(training_data)
        return meta_vote
    if (bagging_choice == True) and (AdaBoostM1_choice ==False) :
        meta_vote = bagging(class_name) 
        print("|##---##| CLASSIFIER :   Applying Bagging Classifier")
        meta_vote.build_classifier(training_data)
        return meta_vote
    if (bagging_choice == False) and (AdaBoostM1_choice ==True) :
        meta_AdaBoostM1 = AdaBoostM1(class_name) 
        meta_AdaBoostM1.build_classifier(training_data)
        print("|##---##| CLASSIFIER :   Applying AdaBoostM1 Classifier")
        return meta_AdaBoostM1
    if (bagging_choice == False) and (AdaBoostM1_choice ==False) :
        normal_cls = Classifier(classname=class_name, options=[])
        normal_cls.build_classifier(training_data)
        return normal_cls


def cls_test_data(test_data,attack_cls_att, test_file_name, attack_cls ,PROJECT_PATH):
    print("|##---##| Classification         :   performing event classification")

    try:
        test_data.insert_attribute(attack_cls_att, test_data.num_attributes)
        test_data.class_is_last() 
    except Exception as e:
        print(e)
    for attack_cls_test_instace in test_data:

        inst_cls = attack_cls.classify_instance(attack_cls_test_instace)
        if inst_cls != 0.0:
            with open(PROJECT_PATH+"/"+"Results/attack_cls_Results_"+str(os.path.basename(test_file_name))+".csv", 'wt', newline='') as clscsvfile: 
                file_header = ["test data","attack event detected"]
                clswriter = csv.writer(clscsvfile, delimiter=',')
                clswriter.writerow(file_header)
                for row in test_data:
                    act_cls = "No Event"
                    claa_pred = attack_cls.classify_instance(row)
                    
                    if claa_pred == 1.0:
                        act_cls = "Random ATTACK"
                    if claa_pred == 2.0:
                        act_cls = "Pulse ATTACK"
                    if claa_pred == 3.0:
                        act_cls = "Type 1 Ramping ATTACK"
                    if claa_pred == 4.0:
                        act_cls = "Type 2 Ramping ATTACK"
                    if claa_pred == 5.0:
                        act_cls = "Scaling ATTACK"
                    if claa_pred == 6.0:
                        act_cls = "Smooth Curve ATTACK"
                    print("Test instance classified as "+ str(act_cls))
                    clswriter.writerow([row, act_cls])
                df = pd.read_csv(PROJECT_PATH+"/"+"Results/attack_cls_Results_"+str(os.path.basename(test_file_name))+".csv")
                fig = px.bar(df, x = "test data", y = ["attack event detected"], title='Attack classification on suspisous instances')
                fig.show()
                return True
    return False


def find_anom_forcasts_by_class_att(Classifier_algorithm_choice, test_data, training_data, test_file_name, bagging_choice, AdaBoostM1_choice, attack_cls_att, attack_cls,PROJECT_PATH):
    print("|##---##| Amon forcast         :   Searching for Anomalous forcasts")
    anom_forcasts_count = 0
    instance_tested = 0 
    inst_anom_count = 0 
    with open(PROJECT_PATH+"/"+"Results/Results_Tolerance_results_file.csv", 'wt', newline='') as csvfile:
        file_header = ["Load","Prediction","Diffrence","Anomalous"]
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(file_header)
        total_diff = 0
        total_rel_diff = 0
        rel_diff_list = []
        num_att = test_data.num_attributes 
        average_dif = 0      
        for inst in test_data:           
            for i in range(3, num_att-1):
                classifier = create_cls_by_class_att(Classifier_algorithm_choice, training_data, bagging_choice, AdaBoostM1_choice, i)
                anomalous = False
                instance_tested = instance_tested + 1
                pred = classifier.classify_instance(inst)
                inst_list = []
                for att in inst:
                    inst_list.append(att)
                act = inst_list[i]
                if act != "?":
                    diffrence = abs(float(pred) - float(act))
                    rel_diff = diffrence/act 
                    rel_diff_list.append(rel_diff)

                    total_diff = total_diff + diffrence
                    average_dif = total_diff/instance_tested
                    if diffrence >= Find_rel_tolerance(act):
                        anom_forcasts_count = anom_forcasts_count + 1
                        inst_anom_count = inst_anom_count + 1
                        anomalous = True
                        data_row = [str(act), str(pred),str(diffrence), str(anomalous)  ]
                        writer.writerow(data_row)
            print("-------------------")
            print("Total Anom :"  + str(anom_forcasts_count))
            print("Average diffrence :" + str(average_dif))
            print("total_tested:" + str(instance_tested))
            print("Max rel diff:" + str(max(rel_diff_list)))
            if inst_anom_count> 0:
                attack_found = cls_test_data(test_data,attack_cls_att, test_file_name , attack_cls,PROJECT_PATH)
                if attack_found == True:
                    return -1
                else:
                    return anom_forcasts_count
    return anom_forcasts_count


def quick_eval(test_data, Classifier_algorithm_choice, file, bagging, AdaBoostM1, training_data,test_file_name,PROJECT_PATH):
    print("|##---##| Evaluation         :   performing Quick Evaluation")
    
    numInstances =0
    total_mean_absolute_error=0
    total_mean_prior_absolute_error=0
    total_root_mean_squared_error=0
    total_root_mean_prior_squared_error=0
    total_root_relative_squared_error=0
    total_tests = 0
    Evaluation_type = "Quick_Evaluation"

    num_att = training_data.num_attributes         
        
    for i in range(3, num_att-1):
        total_tests = total_tests +1
        classifier = create_cls_by_class_att(Classifier_algorithm_choice, training_data, bagging, AdaBoostM1, i)
        test_data.class_index = i
        evaluation = Evaluation(test_data)
        evaluation.test_model(classifier, test_data)
        numInstances = numInstances + int(evaluation.num_instances)
        total_mean_absolute_error = evaluation.mean_absolute_error + total_mean_absolute_error
        total_mean_prior_absolute_error = evaluation.mean_prior_absolute_error +total_mean_prior_absolute_error
        total_root_mean_squared_error = evaluation.root_mean_squared_error +total_root_mean_squared_error
        total_root_mean_prior_squared_error = evaluation.root_mean_prior_squared_error +total_root_mean_prior_squared_error
        total_root_relative_squared_error = evaluation.root_relative_squared_error +total_root_relative_squared_error

    num_instances = numInstances
    mean_absolute_error = total_mean_absolute_error/num_instances
    mean_prior_absolute_error =total_mean_prior_absolute_error/num_instances
    root_mean_squared_error = total_root_mean_squared_error/num_instances
    root_mean_prior_squared_error =total_root_mean_prior_squared_error/num_instances
    root_relative_squared_error = total_root_relative_squared_error/num_instances
    with open(PROJECT_PATH+"/"+"Results/Classifier_quick_Evaluation_"+str(os.path.basename(test_file_name))+".csv", 'wt', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = ["Evaluation_attribute","numInstances","mean_absolute_error","mean_prior_absolute_error","root_mean_squared_error","root_mean_prior_squared_error","root_relative_squared_error","Evaluation_type"]
        writer.writerow(header)
        writer.writerow(["evaluation_score",str(num_instances), str(mean_absolute_error), str(mean_prior_absolute_error), str(root_mean_squared_error), str(root_mean_prior_squared_error), str(root_relative_squared_error),"quick_Evaluation"])
    df = pd.read_csv(PROJECT_PATH+"/"+"Results/Classifier_quick_Evaluation_"+str(os.path.basename(test_file_name))+".csv")
    fig = px.scatter(df, x = ["numInstances","mean_absolute_error","mean_prior_absolute_error","root_mean_squared_error", "root_mean_prior_squared_error", "root_relative_squared_error"],y = ["evaluation_score"], title='Evaluation')
    fig.show()


def ten_fold_evaal(test_data, Classifier_algorithm_choice, file, bagging, AdaBoostM1, training_data, test_file_name,PROJECT_PATH):
    print("|##---##| Evaluation         :   performing Ten Fold Cross Evaluation")
    numInstances =0
    total_mean_absolute_error=0
    total_mean_prior_absolute_error=0
    total_root_mean_squared_error=0
    total_root_mean_prior_squared_error=0
    total_root_relative_squared_error=0
    total_tests = 0
    Evaluation_type = "Ten_Fold_Cross_Evaluation"
    num_att = training_data.num_attributes 
                    
    for i in range(3, num_att-1):
        total_tests = total_tests +1
        classifier = create_cls_by_class_att(Classifier_algorithm_choice, training_data, bagging, AdaBoostM1, i)
        test_data.class_index = i
        evaluation = Evaluation(test_data)
        evaluation.crossvalidate_model(classifier, test_data, 10, Random(42))
        numInstances = numInstances + int(evaluation.num_instances)
        total_mean_absolute_error = evaluation.mean_absolute_error + total_mean_absolute_error
        total_mean_prior_absolute_error = evaluation.mean_prior_absolute_error +total_mean_prior_absolute_error
        total_root_mean_squared_error = evaluation.root_mean_squared_error +total_root_mean_squared_error
        total_root_mean_prior_squared_error = evaluation.root_mean_prior_squared_error +total_root_mean_prior_squared_error
        total_root_relative_squared_error = evaluation.root_relative_squared_error +total_root_relative_squared_error

    num_instances = numInstances
    mean_absolute_error = total_mean_absolute_error/num_instances
    mean_prior_absolute_error =total_mean_prior_absolute_error/num_instances
    root_mean_squared_error = total_root_mean_squared_error/num_instances
    root_mean_prior_squared_error =total_root_mean_prior_squared_error/num_instances
    root_relative_squared_error = total_root_relative_squared_error/num_instances
    with open(PROJECT_PATH+"/"+"Results/Classifier_tenfold_Evaluation_"+str(os.path.basename(test_file_name))+".csv", 'wt', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = ["Evaluation_attribute","numInstances","mean_absolute_error","mean_prior_absolute_error","root_mean_squared_error","root_mean_prior_squared_error","root_relative_squared_error","Evaluation_type"]
        writer.writerow(header)
        writer.writerow(["evaluation_score",str(num_instances), str(mean_absolute_error), str(mean_prior_absolute_error), str(root_mean_squared_error), str(root_mean_prior_squared_error), str(root_relative_squared_error),Evaluation_type])
    df = pd.read_csv(PROJECT_PATH+"/"+"Results/Classifier_tenfold_Evaluation_"+str(os.path.basename(test_file_name))+".csv")
    fig = px.scatter(df, x = ["numInstances","mean_absolute_error","mean_prior_absolute_error","root_mean_squared_error", "root_mean_prior_squared_error", "root_relative_squared_error"],y = ["evaluation_score"], title='Evaluation')
    fig.show()


def import_prev_load_data_for_training_or_analysis(import_file_name, PROJECT_PATH, start_date, end_date):

    if start_date == None:
        start_date = convert_date_format((1970,1,1))
    if end_date == None:
        end_date = convert_date_format((2100,1,1))
    print("|##---##| Importing         :   Importing Load data for Training/Analysis")
    if "final_data.xlsx" not in import_file_name:
        return None
    return_list = []
    imported_data = pd.read_excel(import_file_name)
    if ".xlsx" in import_file_name:
        new_name = None
        for char in import_file_name:
            if char != ".":
                if new_name == None:
                    new_name = char
                else:
                    new_name = new_name + str(char)
            if char == ".":
                break
    import_file_name = new_name
    date_list = []
    Times_list = []
    for insta in imported_data.iloc[:, 0]:
        Times_list.append(insta)
    for insta in imported_data:
        if insta != 'Unnamed: 0':
            date_list.append(insta)
    imported_data = imported_data.transpose()
    imported_data.to_csv(PROJECT_PATH+"/"+"Processes/"+os.path.basename(import_file_name)+".csv", index=False, na_rep='?')
    imported_data = None
    i = -1 
    imported_data = get_arff_csv_file(PROJECT_PATH+"/"+"Processes/"+os.path.basename(import_file_name)+".csv")
    first_row = True
    start_date_list = start_date.split("-")

    start_year = int(start_date_list[0])
    start_month = int(start_date_list[1])
    start_day = int(start_date_list[2])

    end_date_list = end_date.split("-")

    end_year = int(end_date_list[0])
    end_month = int(end_date_list[1])
    end_day = int(end_date_list[2])

    for row in imported_data: 
        date = date_list[i]
        indiv_date_comp_list = []
        if first_row == True: 
            indiv_date_comp_list = ["Year","Month","Day"]
            
        if indiv_date_comp_list == []:
            indiv_date_comp_list  = date.split("-")
        row_list = []

        for date_entry in indiv_date_comp_list:
            row_list.append(date_entry) 
                    
        for entry in row:
            row_list.append(entry)

        if first_row == True:
            return_list.append(row_list)
            first_row = False
        else:
            if (start_year <= int(row_list[0])) and (end_year  >= int(row_list[0])):
                return_list.append(row_list)
            else:
                if start_year <= int(row_list[0]) <= end_year:
                    if (start_month < int(row_list[1])) or (end_month > int(row_list[1])):
                        return_list.append(row_list)
                    elif start_month == int(row_list[1]):
                        if start_day <= int(row_list[2]):
                            return_list.append(row_list)

                    elif end_month == int(row_list[1]):
                        if int(row_list[2]) <= end_day:
                            return_list.append(row_list)
        i = i+1
    with open(PROJECT_PATH+"/"+"Processes/IGNORE_load_data_coversion_file.csv","w", newline='') as w:
        writer = csv.writer(w)
        writer.writerows(return_list)
    return_file = get_arff_csv_file(PROJECT_PATH+"/"+"Processes/IGNORE_load_data_coversion_file.csv")
    return return_file, Times_list


def convert_forcast_to_test_data(import_file_name,PROJECT_PATH, Times_list):
    print("|##---##| Importing         :   Importing forcast data for Analysis")
    if "forecast_data.xlsx" not in import_file_name:
        return None
    return_list = []
    imported_data = pd.read_excel(import_file_name)
    if ".xlsx" in import_file_name:
        new_name = None
        for char in import_file_name:
            if char != ".":
                if new_name == None:
                    new_name = char
                else:
                    new_name = new_name + str(char)
            if char == ".":
                break
    import_file_name = new_name
    date_list = []
    for insta in imported_data.iloc[:, 0]:
        date_list.append(insta)
    imported_data.to_csv(PROJECT_PATH+"/"+"Processes/"+os.path.basename(import_file_name)+".csv", index=False, na_rep='?')
    imported_data = None
    i = 0  
    imported_data = get_arff_csv_file(PROJECT_PATH+"/"+"Processes/"+os.path.basename(import_file_name)+".csv")
    first_row = True
    for row in imported_data: 
        date = date_list[i]
        indiv_date_comp_list = []
        row_list = []
        if first_row == True: 
            indiv_date_comp_list = ["Year","Month","Day"]
            for date_header_entry in indiv_date_comp_list:
                row_list.append(date_header_entry)
                first_row = False
            for time_header_entry in Times_list:
                row_list.append(time_header_entry)
        else:
            first_entry = True
            for row_entry in row:
                if first_entry == True:
                    indiv_date_comp_list = row_entry.split("-")
                    for date_header_entry in indiv_date_comp_list:
                        row_list.append(float(date_header_entry))
                        first_entry = False
                else:
                    row_list.append(float(row_entry))
        return_list.append(row_list)
        i = i+1
    with open(PROJECT_PATH+"/"+"Processes/IGNORE_forcasted_load_data_coversion_file.csv","w", newline='') as w:
        writer = csv.writer(w)
        writer.writerows(return_list)

    arff_file = get_arff_csv_file(PROJECT_PATH+"/"+"Processes/IGNORE_forcasted_load_data_coversion_file.csv")
    return arff_file