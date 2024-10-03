import tkinter as tk
import tkinter.ttk as ttk
from tkinter import scrolledtext
import customtkinter as ctk
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense , Dropout
from keras.models import model_from_json , load_model
from keras import initializers
from sklearn.model_selection import KFold
import tkinter.filedialog
from contextlib import redirect_stdout
from keras.models import Model
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

class TextScrollCombo(ttk.Frame):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    # ensure a consistent GUI size
        self.grid_propagate(False)
    # implement stretchability
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    # create a Text widget
        self.txt = tk.Text(self)
        self.txt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    # create a Scrollbar and associate it with txt
        scrollb = ttk.Scrollbar(self, command=self.txt.yview)
        scrollb.grid(row=0, column=1, sticky='nsew')
        self.txt['yscrollcommand'] = scrollb.set


def summary(model: tf.keras.Model) -> str:
  summary = []
  model.summary(print_fn=lambda x: summary.append(x))
  return '\n'.join(summary)


def show_model(mod):
    textmod = summary(mod)
    #comboTmod.delete(1.0,tk.END)
    #comboTmod.insert(tk.END, textmod)



def plot_mtx(confusion_mtx,labels):
    confusion_mtx = np.array(confusion_mtx)
    confusion_mtx = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(13, 10))
    sns.heatmap(confusion_mtx, cmap="Blues", annot=True, fmt='.4f', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()


def test_model_bin():

    
    filename = tkinter.filedialog.askopenfilename(initialdir="app-v2/dataset", title="Open File", filetypes=(("executables","*.csv"), ("allfiles","*.*")))

    #Read Dataset
    dataset = pd.read_csv(filename)

    X_predict = dataset.drop(['Label'], axis=1)
    Y_predict = dataset['Label']

    labels = ['BENIGN','DDoS']

    #Normalizing features
    ms = MinMaxScaler()
    scaler = StandardScaler()
    X_predict = scaler.fit_transform(X_predict)
    X_predict = ms.fit_transform(X_predict)

    alr1bin = 0
    alr2bin = 0


    #Making Sample Predictions
    classes = model.predict(X_predict)
    print(classes)
    y_pred = np.argmax(classes, axis=-1)

    alr1bin = (y_pred == 0.0).sum()
    alr2bin = (y_pred == 1.0).sum()

    label_alr1_bin_num.configure(text=str(alr1bin))
    label_alr2_bin_num.configure(text=str(alr2bin), text_color='#A31621')

    class_rep = classification_report(Y_predict, y_pred, target_names = labels, digits=6)
    comboTbin.insert(tk.END,class_rep)
    comboTbin.insert(tk.END,'\n')
    confusion_mtx = confusion_matrix(Y_predict, y_pred)
    plot_mtx(confusion_mtx,labels)
    

def plotts_bin():
    bin_overallacc = [0.9986,0.9986,0.9996,0.9996,0.9996,0.9996,0.9996,0.9996,0.9997,0.9997]
    bin_overallvalacc = [0.9996,0.9996,0.9996,0.9996,0.9996,0.9996,0.9997,0.9997,0.9996,0.9996]
    bin_overallloss = [0.0053,0.0048,0.0042,0.0035,0.0022,0.0020,0.0019,0.0017,0.0016,0.0014]
    bin_overallvalloss = [0.0021,0.0021,0.0018,0.0018,0.0017,0.0017,0.0016,0.0016,0.0015,0.0015]

    plt.figure(figsize=(13, 10))
    plt.plot(bin_overallacc)
    plt.plot(bin_overallvalacc)
    #plt.xlim([1, 6])
    plt.ylim([0.99, 1.001])
    plt.title('Exactitude de notre modèle - Classification binaire')
    plt.ylabel('Exactitude')
    plt.xlabel('Epoque')
    plt.legend(['Entrainement','Validation'], loc='upper left')
    plt.show()

    plt.figure(figsize=(13, 10))
    plt.plot(bin_overallloss)
    plt.plot(bin_overallvalloss)
    #plt.xlim([1, 6])
    plt.ylim([0.0, 0.02])
    plt.title('Perte de notre modèle - Classification binaire')
    plt.ylabel('Perte')
    plt.xlabel('Epoque')
    plt.legend(['Entrainement','Validation'], loc='upper left')
    plt.show()

    labels = ['BENIGN','DDoS']
    matrice = [[337244,517],[741,2885845]]
    plot_mtx(matrice,labels)


def test_model_4class():

    
    filename = tkinter.filedialog.askopenfilename(initialdir="app-v2/dataset", title="Open File", filetypes=(("executables","*.csv"), ("allfiles","*.*")))

    #Read Dataset
    dataset = pd.read_csv(filename)

    X_predict = dataset.drop(['Label'], axis=1)
    Y_predict = dataset['Label']

    labels = ['BENIGN','DNS/LDAP/SNMP/NetBios','MSSQL','NTP/SSDP/UDP/Syn']

    #Normalizing features
    ms = MinMaxScaler()
    scaler = StandardScaler()
    X_predict = scaler.fit_transform(X_predict)
    X_predict = ms.fit_transform(X_predict)

    alr14c = 0
    alr24c = 0
    alr34c = 0
    alr44c = 0

    #Making Sample Predictions
    classes = model.predict(X_predict)
    print(classes)
    y_pred = np.argmax(classes, axis=-1)

    alr14c = (y_pred == 0.0).sum()
    alr24c = (y_pred == 1.0).sum()
    alr34c = (y_pred == 2.0).sum()
    alr44c = (y_pred == 3.0).sum()

    label_alr1_4class_num.configure(text=str(alr14c))
    label_alr2_4class_num.configure(text=str(alr24c), text_color='#A31621')
    label_alr3_4class_num.configure(text=str(alr34c), text_color='#A31621')
    label_alr4_4class_num.configure(text=str(alr44c), text_color='#A31621')


    class_rep = classification_report(Y_predict, y_pred, target_names = labels, digits=6)
    comboT4class.insert(tk.END,class_rep)
    comboT4class.insert(tk.END,'\n')
    confusion_mtx = confusion_matrix(Y_predict, y_pred)
    plot_mtx(confusion_mtx,labels)
    

def plotts_4class():
    gr_overallacc = [0.9565,0.9659,0.9675,0.9687,0.9695,0.9695,0.9700,0.9701,0.9706,0.9709,0.9713,0.9714,0.9714,0.9715,0.9720,0.9719,0.9719,0.9722,0.9723,0.9722]
    gr_overallvalacc = [0.9565,0.9515,0.9710,0.9713,0.9580,0.9743,0.9749,0.9750,0.9720,0.9746,0.9690,0.9744,0.9536,0.9640,0.9715,0.9717,0.9739,0.9708,0.9724,0.9725]
    gr_overallloss = [0.1399,0.1140,0.1095,0.1071,0.1048,0.1049,0.1043,0.1032,0.1021,0.1015,0.1006,0.1004,0.1003,0.1001,0.0989,0.0990,0.0990,0.0981,0.0980,0.0981]
    gr_overallvalloss = [0.1269,0.1305,0.1060,0.1022,0.1095,0.0977,0.0975,0.0967,0.1011,0.0988,0.1013,0.0926,0.1473,0.1138,0.1009,0.0976,0.0943,0.1035,0.0984,0.0984]

    plt.figure(figsize=(13, 10))
    plt.plot(gr_overallacc)
    plt.plot(gr_overallvalacc)
    #plt.xlim([1, 6])
    plt.ylim([0.80, 1.01])
    plt.title('Exactitude de notre modèle - Classification 4 classes')
    plt.ylabel('Exactitude')
    plt.xlabel('Epoque')
    plt.legend(['Entrainement','Validation'], loc='upper left')
    plt.show()

    plt.figure(figsize=(13, 10))
    plt.plot(gr_overallloss)
    plt.plot(gr_overallvalloss)
    #plt.xlim([1, 6])
    plt.ylim([0.0, 1.0])
    plt.title('Perte de notre modèle - Classification 4 classes')
    plt.ylabel('Perte')
    plt.xlabel('Epoque')
    plt.legend(['Entrainement','Validation'], loc='upper left')
    plt.show()

    labels = ['BENIGN','DNS/LDAP/SNMP/NetBios','MSSQL','NTP/SSDP/UDP/Syn']
    matrice = [[337685,30,0,46],[182,1532398,24081,4591],[31,18605,394125,26429],[425,6023,15413,864283]]
    plot_mtx(matrice,labels)



def test_model_8class():

    
    filename = tkinter.filedialog.askopenfilename(initialdir="app-v2/dataset", title="Open File", filetypes=(("executables","*.csv"), ("allfiles","*.*")))

    #Read Dataset
    dataset = pd.read_csv(filename)

    X_predict = dataset.drop(['Label'], axis=1)
    Y_predict = dataset['Label']

    labels = ['BENIGN','DNS/LDAP','MSSQL','NTP', 'NetBios', 'SNMP', 'SSDP/UDP', 'Syn']

    #Normalizing features
    ms = MinMaxScaler()
    scaler = StandardScaler()
    X_predict = scaler.fit_transform(X_predict)
    X_predict = ms.fit_transform(X_predict)

    alr18c = 0
    alr28c = 0
    alr38c = 0
    alr48c = 0
    alr58c = 0
    alr68c = 0
    alr78c = 0
    alr88c = 0

    #Making Sample Predictions
    classes = model.predict(X_predict)
    print(classes)
    y_pred = np.argmax(classes, axis=-1)

    alr18c = (y_pred == 0.0).sum()
    alr28c = (y_pred == 1.0).sum()
    alr38c = (y_pred == 2.0).sum()
    alr48c = (y_pred == 3.0).sum()
    alr58c = (y_pred == 4.0).sum()
    alr68c = (y_pred == 5.0).sum()
    alr78c = (y_pred == 6.0).sum()
    alr88c = (y_pred == 7.0).sum()

    label_alr1_8class_num.configure(text=str(alr18c))
    label_alr2_8class_num.configure(text=str(alr28c), text_color='#A31621')
    label_alr3_8class_num.configure(text=str(alr38c), text_color='#A31621')
    label_alr4_8class_num.configure(text=str(alr48c), text_color='#A31621')
    label_alr5_8class_num.configure(text=str(alr58c), text_color='#A31621')
    label_alr6_8class_num.configure(text=str(alr68c), text_color='#A31621')
    label_alr7_8class_num.configure(text=str(alr78c), text_color='#A31621')
    label_alr8_8class_num.configure(text=str(alr88c), text_color='#A31621')


    class_rep = classification_report(Y_predict, y_pred, target_names = labels, digits=6)
    comboT8class.insert(tk.END,class_rep)
    comboT8class.insert(tk.END,'\n')
    confusion_mtx = confusion_matrix(Y_predict, y_pred)
    plot_mtx(confusion_mtx,labels)
    

def plotts_8class():
    multi_overallacc = [0.8943,0.9063,0.9086,0.9099,0.9107,0.9113,0.9119,0.9120,0.9123,0.9126,0.9129,0.9129,0.9130,0.9132,0.9140,0.9143,0.9144,0.9140,0.9143,0.9145,0.9147,0.9147,0.9151,0.9152,0.9153,0.9154,0.9155,0.9162,0.9159]
    multi_overallvalacc = [0.8930,0.9036,0.8742,0.9091,0.9138,0.9131,0.9153,0.9073,0.9156,0.9134,0.9110,0.9141,0.9047,0.9124,0.9128,0.9157,0.9155,0.9112,0.9184,0.9121,0.9037,0.8977,0.9146,0.9052,0.9129,0.9098,0.9086,0.9081,0.9084]
    multi_overallloss = [0.3250,0.2875,0.2813,0.2783,0.2757,0.2742,0.2725,0.2722,0.2711,0.2703,0.2688,0.2689,0.2684,0.2676,0.2656,0.2649,0.2650,0.2658,0.2644,0.2641,0.2637,0.2635,0.2628,0.2619,0.2627,0.2623,0.2619,0.2601,0.2605]
    multi_overallvalloss = [0.3235,0.2940,0.3901,0.2765,0.2702,0.2681,0.2664,0.2944,0.2676,0.2691,0.2803,0.2637,0.2914,0.2672,0.2652,0.2581,0.2614,0.2744,0.2560,0.2693,0.2855,0.2973,0.2635,0.2889,0.2717,0.2770,0.2834,0.2574,0.2564]

    plt.figure(figsize=(13, 10))
    plt.plot(multi_overallacc)
    plt.plot(multi_overallvalacc)
    #plt.xlim([1, 6])
    plt.ylim([0.0, 1.00])
    plt.title('Exactitude de notre modèle - Classification 8 classes')
    plt.ylabel('Exactitude')
    plt.xlabel('Epoque')
    plt.legend(['Entrainement','Validation'], loc='upper left')
    plt.show()

    plt.figure(figsize=(13, 10))
    plt.plot(multi_overallloss)
    plt.plot(multi_overallvalloss)
    #plt.xlim([1, 6])
    plt.ylim([0.0, 1.0])
    plt.title('Perte de notre modèle - Classification 8 classes')
    plt.ylabel('Perte')
    plt.xlabel('Epoque')
    plt.legend(['Entrainement','Validation'], loc='upper left')
    plt.show()

    labels = ['BENIGN','DNS/LDAP','MSSQL','NTP', 'NetBios', 'SNMP', 'SSDP/UDP', 'Syn']
    matrice = [[337703,16,0,11,9,0,5,17],[154,540863,20417,992,3102,108067,3698,15],[38,8302,420870,118,0,9081,772,9],[272,788,12187,98358,71,10,7663,13],[25,1194,173,1,367410,1,471,28],[74,59778,3466,31,22779,428409,100,4],[104,899,94657,1108,27,4093,465961,6],[113,30,228,5,11,12,24,199504]]
    plot_mtx(matrice,labels)



def model_choice():
    global model
    
    choice = tabview.get()

    if choice == "Binaire":
        model = load_model('app-v2/model/model-cnn-lstm-full-bin-pr-99.keras')
    elif choice == "4-Classes":
        model = load_model('app-v2/model/model-cnn-lstm-full-gr-pr-97.keras')
    elif choice == "8-Classes":
        model = load_model('app-v2/model/model-cnn-lstm-full-multi-pr-88.keras')
    
def refresh(tab):

    if tab == "Binaire":
        label_alr1_bin_num.configure(text=str(0), text_color='white')
        label_alr2_bin_num.configure(text=str(0), text_color='white')
        comboTbin.delete(1.0,tk.END) 

    elif tab == "4-Classes":
        label_alr1_4class_num.configure(text=str(0), text_color='white')
        label_alr2_4class_num.configure(text=str(0), text_color='white')
        label_alr3_4class_num.configure(text=str(0), text_color='white')
        label_alr4_4class_num.configure(text=str(0), text_color='white')
        comboT4class.delete(1.0,tk.END)
    elif tab == "8-Classes":
        label_alr1_8class_num.configure(text=str(0), text_color='white')
        label_alr2_8class_num.configure(text=str(0), text_color='white')
        label_alr3_8class_num.configure(text=str(0), text_color='white')
        label_alr4_8class_num.configure(text=str(0), text_color='white')
        label_alr5_8class_num.configure(text=str(0), text_color='white')
        label_alr6_8class_num.configure(text=str(0), text_color='white')
        label_alr7_8class_num.configure(text=str(0), text_color='white')
        label_alr8_8class_num.configure(text=str(0), text_color='white')
        comboT8class.delete(1.0,tk.END)


       


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

intro = ctk.CTk()
intro.title("PFE")
intro.state('normal')
intro.geometry("2400x1000")

model = load_model('app-v2/model/model-cnn-lstm-full-bin-pr-99.keras')

tabview = ctk.CTkTabview(intro, command=model_choice)
tabview.place(relheight=0.95, relwidth=0.95, relx=0.025, rely=0.025)
tab_bin = tabview.add("Binaire")
tab_4class = tabview.add("4-Classes")
tab_8class = tabview.add("8-Classes")

#####BIN

frame_btn_bin = ctk.CTkFrame(tab_bin)
frame_btn_bin.place(relheight=0.2, relwidth=0.95, relx=0.025, rely=0.1)

frame_alrt_bin = ctk.CTkFrame(tab_bin)
frame_alrt_bin.place(relheight=0.6, relwidth=0.3, relx=0.025, rely=0.3)

frame_aff_bin = ctk.CTkFrame(tab_bin)
frame_aff_bin.place(relheight=0.6, relwidth=0.6, relx=0.35, rely=0.3)

comboTbin = scrolledtext.ScrolledText(frame_aff_bin, background='#292929', foreground='white')
comboTbin.pack(fill="both", expand=True)
comboTbin.configure(width=10, height=200)

testbinB = ctk.CTkButton(frame_btn_bin, text="Evaluer", command=test_model_bin)
testbinB.place(relheight=0.3, relwidth=0.2, relx=0.1, rely=0.4)

refreshbinB = ctk.CTkButton(frame_btn_bin, text="Effacer", command=lambda: refresh(tabview.get()))
refreshbinB.place(relheight=0.3, relwidth=0.2, relx=0.7, rely=0.4)

graphbinB = ctk.CTkButton(frame_btn_bin, text="Graphes", command=plotts_bin)
graphbinB.place(relheight=0.3, relwidth=0.2, relx=0.4, rely=0.4)

frame_alr1_bin = ctk.CTkFrame(frame_alrt_bin)
frame_alr1_bin.place(relheight=0.3, relwidth=0.6, relx=0.1, rely=0.1)
label_alr1_bin_txt = ctk.CTkLabel(frame_alr1_bin, text="Flux\nLégitimes", fg_color="transparent", font=('Minecraft',15))
label_alr1_bin_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr1_bin_sep = ctk.CTkLabel(frame_alr1_bin, text="", fg_color="#232323")
label_alr1_bin_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr1_bin_num = ctk.CTkLabel(frame_alr1_bin, text="0", fg_color="transparent", font=('Minecraft',20))
label_alr1_bin_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr2_bin = ctk.CTkFrame(frame_alrt_bin)
frame_alr2_bin.place(relheight=0.3, relwidth=0.6, relx=0.1, rely=0.6)
label_alr2_bin_txt = ctk.CTkLabel(frame_alr2_bin, text="Attaques\nDoS", fg_color="transparent", font=('Minecraft',15))
label_alr2_bin_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr2_bin_sep = ctk.CTkLabel(frame_alr2_bin, text="", fg_color="#232323")
label_alr2_bin_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr2_bin_num = ctk.CTkLabel(frame_alr2_bin, text="0", fg_color="transparent", font=('Minecraft',20))
label_alr2_bin_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)


####4-Class

frame_btn_4class = ctk.CTkFrame(tab_4class)
frame_btn_4class.place(relheight=0.2, relwidth=0.95, relx=0.025, rely=0.1)

frame_alrt_4class = ctk.CTkFrame(tab_4class)
frame_alrt_4class.place(relheight=0.6, relwidth=0.3, relx=0.025, rely=0.3)

frame_aff_4class = ctk.CTkFrame(tab_4class)
frame_aff_4class.place(relheight=0.6, relwidth=0.6, relx=0.35, rely=0.3)

comboT4class = scrolledtext.ScrolledText(frame_aff_4class, background='#292929', foreground='white')
comboT4class.pack(fill="both", expand=True)
comboT4class.configure(width=10, height=200)

test4classB = ctk.CTkButton(frame_btn_4class, text="Evaluer", command=test_model_4class)
test4classB.place(relheight=0.3, relwidth=0.2, relx=0.1, rely=0.4)

refresh4classB = ctk.CTkButton(frame_btn_4class, text="Effacer", command=lambda: refresh(tabview.get()))
refresh4classB.place(relheight=0.3, relwidth=0.2, relx=0.7, rely=0.4)

graph4classB = ctk.CTkButton(frame_btn_4class, text="Graphes", command=plotts_4class)
graph4classB.place(relheight=0.3, relwidth=0.2, relx=0.4, rely=0.4)

frame_alr1_4class = ctk.CTkFrame(frame_alrt_4class)
frame_alr1_4class.place(relheight=0.2, relwidth=0.6, relx=0.1, rely=0.1)
label_alr1_4class_txt = ctk.CTkLabel(frame_alr1_4class, text="Flux\nLégitimes", fg_color="transparent", font=('Minecraft',12))
label_alr1_4class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr1_4class_sep = ctk.CTkLabel(frame_alr1_4class, text="", fg_color="#232323")
label_alr1_4class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr1_4class_num = ctk.CTkLabel(frame_alr1_4class, text="0", fg_color="transparent", font=('Minecraft',20))
label_alr1_4class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr2_4class = ctk.CTkFrame(frame_alrt_4class)
frame_alr2_4class.place(relheight=0.2, relwidth=0.6, relx=0.1, rely=0.3)
label_alr2_4class_txt = ctk.CTkLabel(frame_alr2_4class, text="DNS/LDAP\nSNMP/NetBios", fg_color="transparent", font=('Minecraft',12))
label_alr2_4class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr2_4class_sep = ctk.CTkLabel(frame_alr2_4class, text="", fg_color="#232323")
label_alr2_4class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr2_4class_num = ctk.CTkLabel(frame_alr2_4class, text="0", fg_color="transparent", font=('Minecraft',20))
label_alr2_4class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr3_4class = ctk.CTkFrame(frame_alrt_4class)
frame_alr3_4class.place(relheight=0.2, relwidth=0.6, relx=0.1, rely=0.5)
label_alr3_4class_txt = ctk.CTkLabel(frame_alr3_4class, text="MSSQL", fg_color="transparent", font=('Minecraft',12))
label_alr3_4class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr3_4class_sep = ctk.CTkLabel(frame_alr3_4class, text="", fg_color="#232323")
label_alr3_4class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr3_4class_num = ctk.CTkLabel(frame_alr3_4class, text="0", fg_color="transparent", font=('Minecraft',20))
label_alr3_4class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr4_4class = ctk.CTkFrame(frame_alrt_4class)
frame_alr4_4class.place(relheight=0.2, relwidth=0.6, relx=0.1, rely=0.7)
label_alr4_4class_txt = ctk.CTkLabel(frame_alr4_4class, text="NTP/SSDP\nUDP/Syn", fg_color="transparent", font=('Minecraft',12))
label_alr4_4class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr4_4class_sep = ctk.CTkLabel(frame_alr4_4class, text="", fg_color="#232323")
label_alr4_4class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr4_4class_num = ctk.CTkLabel(frame_alr4_4class, text="0", fg_color="transparent", font=('Minecraft',20))
label_alr4_4class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)


####8-Class

frame_btn_8class = ctk.CTkFrame(tab_8class)
frame_btn_8class.place(relheight=0.2, relwidth=0.95, relx=0.025, rely=0.1)

frame_alrt_8class = ctk.CTkFrame(tab_8class)
frame_alrt_8class.place(relheight=0.6, relwidth=0.3, relx=0.025, rely=0.3)

frame_aff_8class = ctk.CTkFrame(tab_8class)
frame_aff_8class.place(relheight=0.6, relwidth=0.6, relx=0.35, rely=0.3)

comboT8class = scrolledtext.ScrolledText(frame_aff_8class, background='#292929', foreground='white')
comboT8class.pack(fill="both", expand=True)
comboT8class.configure(width=10, height=200)

test8classB = ctk.CTkButton(frame_btn_8class, text="Evaluer", command=test_model_8class)
test8classB.place(relheight=0.3, relwidth=0.2, relx=0.1, rely=0.4)

refresh8classB = ctk.CTkButton(frame_btn_8class, text="Effacer", command=lambda: refresh(tabview.get()))
refresh8classB.place(relheight=0.3, relwidth=0.2, relx=0.7, rely=0.4)

graph8classB = ctk.CTkButton(frame_btn_8class, text="Graphes", command=plotts_8class)
graph8classB.place(relheight=0.3, relwidth=0.2, relx=0.4, rely=0.4)

frame_alr1_8class = ctk.CTkFrame(frame_alrt_8class)
frame_alr1_8class.place(relheight=0.2, relwidth=0.4, relx=0.1, rely=0.1)
label_alr1_8class_txt = ctk.CTkLabel(frame_alr1_8class, text="Flux\nLégitimes", fg_color="transparent", font=('Minecraft',11))
label_alr1_8class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr1_8class_sep = ctk.CTkLabel(frame_alr1_8class, text="", fg_color="#232323")
label_alr1_8class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr1_8class_num = ctk.CTkLabel(frame_alr1_8class, text="0", fg_color="transparent", font=('Minecraft',14))
label_alr1_8class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr2_8class = ctk.CTkFrame(frame_alrt_8class)
frame_alr2_8class.place(relheight=0.2, relwidth=0.4, relx=0.1, rely=0.3)
label_alr2_8class_txt = ctk.CTkLabel(frame_alr2_8class, text="DNS/LDAP", fg_color="transparent", font=('Minecraft',10))
label_alr2_8class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr2_8class_sep = ctk.CTkLabel(frame_alr2_8class, text="", fg_color="#232323")
label_alr2_8class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr2_8class_num = ctk.CTkLabel(frame_alr2_8class, text="0", fg_color="transparent", font=('Minecraft',14))
label_alr2_8class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr3_8class = ctk.CTkFrame(frame_alrt_8class)
frame_alr3_8class.place(relheight=0.2, relwidth=0.4, relx=0.1, rely=0.5)
label_alr3_8class_txt = ctk.CTkLabel(frame_alr3_8class, text="MSSQL", fg_color="transparent", font=('Minecraft',12))
label_alr3_8class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr3_8class_sep = ctk.CTkLabel(frame_alr3_8class, text="", fg_color="#232323")
label_alr3_8class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr3_8class_num = ctk.CTkLabel(frame_alr3_8class, text="0", fg_color="transparent", font=('Minecraft',14))
label_alr3_8class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr4_8class = ctk.CTkFrame(frame_alrt_8class)
frame_alr4_8class.place(relheight=0.2, relwidth=0.4, relx=0.1, rely=0.7)
label_alr4_8class_txt = ctk.CTkLabel(frame_alr4_8class, text="NTP", fg_color="transparent", font=('Minecraft',12))
label_alr4_8class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr4_8class_sep = ctk.CTkLabel(frame_alr4_8class, text="", fg_color="#232323")
label_alr4_8class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr4_8class_num = ctk.CTkLabel(frame_alr4_8class, text="0", fg_color="transparent", font=('Minecraft',14))
label_alr4_8class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr5_8class = ctk.CTkFrame(frame_alrt_8class)
frame_alr5_8class.place(relheight=0.2, relwidth=0.4, relx=0.5, rely=0.1)
label_alr5_8class_txt = ctk.CTkLabel(frame_alr5_8class, text="NetBios", fg_color="transparent", font=('Minecraft',12))
label_alr5_8class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr5_8class_sep = ctk.CTkLabel(frame_alr5_8class, text="", fg_color="#232323")
label_alr5_8class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr5_8class_num = ctk.CTkLabel(frame_alr5_8class, text="0", fg_color="transparent", font=('Minecraft',14))
label_alr5_8class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr6_8class = ctk.CTkFrame(frame_alrt_8class)
frame_alr6_8class.place(relheight=0.2, relwidth=0.4, relx=0.5, rely=0.3)
label_alr6_8class_txt = ctk.CTkLabel(frame_alr6_8class, text="SNMP", fg_color="transparent", font=('Minecraft',12))
label_alr6_8class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr6_8class_sep = ctk.CTkLabel(frame_alr6_8class, text="", fg_color="#232323")
label_alr6_8class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr6_8class_num = ctk.CTkLabel(frame_alr6_8class, text="0", fg_color="transparent", font=('Minecraft',14))
label_alr6_8class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr7_8class = ctk.CTkFrame(frame_alrt_8class)
frame_alr7_8class.place(relheight=0.2, relwidth=0.4, relx=0.5, rely=0.5)
label_alr7_8class_txt = ctk.CTkLabel(frame_alr7_8class, text="SSDP/UDP", fg_color="transparent", font=('Minecraft',10))
label_alr7_8class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr7_8class_sep = ctk.CTkLabel(frame_alr7_8class, text="", fg_color="#232323")
label_alr7_8class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr7_8class_num = ctk.CTkLabel(frame_alr7_8class, text="0", fg_color="transparent", font=('Minecraft',14))
label_alr7_8class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)

frame_alr8_8class = ctk.CTkFrame(frame_alrt_8class)
frame_alr8_8class.place(relheight=0.2, relwidth=0.4, relx=0.5, rely=0.7)
label_alr8_8class_txt = ctk.CTkLabel(frame_alr8_8class, text="Syn", fg_color="transparent", font=('Minecraft',12))
label_alr8_8class_txt.place(relheight=0.9, relwidth=0.4, relx=0.05, rely=0.05)
label_alr8_8class_sep = ctk.CTkLabel(frame_alr8_8class, text="", fg_color="#232323")
label_alr8_8class_sep.place(relheight=0.9, relwidth=0.01, relx=0.5, rely=0.05)
label_alr8_8class_num = ctk.CTkLabel(frame_alr8_8class, text="0", fg_color="transparent", font=('Minecraft',14))
label_alr8_8class_num.place(relheight=0.9, relwidth=0.4, relx=0.55, rely=0.05)



intro.mainloop()