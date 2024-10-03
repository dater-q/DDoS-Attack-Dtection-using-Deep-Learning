import pandas as pd
from datetime import datetime
import tensorflow as tf
import random
import sklearn
import numpy as np

start=datetime.now()

df = pd.read_csv('New folder/processed-dataset/dns.csv')


df = df.drop('Unnamed: 0', axis=1)
df = df.drop('Flow ID', axis=1)
df = df.drop('Source IP', axis=1)
df = df.drop('Source Port', axis=1)
df = df.drop('Destination IP', axis=1)
df = df.drop('Timestamp', axis=1)
df = df.drop('Fwd Header Length.1', axis=1)
df = df.drop('Fwd Avg Bytes/Bulk', axis=1)
df = df.drop('Fwd Avg Packets/Bulk', axis=1)
df = df.drop('Fwd Avg Bulk Rate', axis=1)
df = df.drop('Bwd Avg Bytes/Bulk', axis=1)
df = df.drop('Bwd Avg Packets/Bulk', axis=1)
df = df.drop('Bwd Avg Bulk Rate', axis=1)
df = df.drop('Subflow Fwd Packets', axis=1)
df = df.drop('Subflow Bwd Packets', axis=1)
df = df.drop('Subflow Bwd Bytes', axis=1)
df = df.drop('SimillarHTTP', axis=1)
df = df.drop('Inbound', axis=1)

df['Subflow Fwd Bytes'] = df['Subflow Fwd Bytes'].astype(float)
df['Min Packet Length'] = df['Min Packet Length'].astype(float)
df['Max Packet Length'] = df['Max Packet Length'].astype(float)
df['Fwd Packets/s'] = df['Fwd Packets/s'].astype(float)
df['Fwd Header Length'] = df['Fwd Header Length'].astype(float)
df['Flow IAT Max'] = df['Flow IAT Max'].astype(float)
df['Fwd IAT Mean'] = df['Fwd IAT Mean'].astype(float)
df['Fwd IAT Max'] = df['Fwd IAT Max'].astype(float)
df['Destination Port'] = df['Destination Port'].astype(float)

df['Protocol'] = df['Protocol'].astype(float)
df['Packet Length Std'] = df['Packet Length Std'].astype(float)
df['Fwd Packet Length Std'] = df['Fwd Packet Length Std'].astype(float)
df['Flow Duration'] = df['Flow Duration'].astype(float)
df['Total Backward Packets'] = df['Total Backward Packets'].astype(float)
df['Total Length of Fwd Packets'] = df['Total Length of Fwd Packets'].astype(float)
df['Fwd Packet Length Min'] = df['Fwd Packet Length Min'].astype(float)
df['Fwd Packet Length Max'] = df['Fwd Packet Length Max'].astype(float)
df['Bwd Packet Length Min'] = df['Bwd Packet Length Min'].astype(float)
df['Bwd Packet Length Max'] = df['Bwd Packet Length Max'].astype(float)
df['Flow Packets/s'] = df['Flow Packets/s'].astype(float)
df['Flow IAT Mean'] = df['Flow IAT Mean'].astype(float)
df['Flow IAT Min'] = df['Flow IAT Min'].astype(float)
df['Fwd IAT Total'] = df['Fwd IAT Total'].astype(float)
df['Fwd IAT Min'] = df['Fwd IAT Min'].astype(float)
df['Bwd IAT Total'] = df['Bwd IAT Total'].astype(float)
df['Bwd IAT Min'] = df['Bwd IAT Min'].astype(float)
df['Bwd Packets/s'] = df['Bwd Packets/s'].astype(float)
df['ACK Flag Count'] = df['ACK Flag Count'].astype(float)
df['Down/Up Ratio'] = df['Down/Up Ratio'].astype(float)
df['Init_Win_bytes_forward'] = df['Init_Win_bytes_forward'].astype(float)
df['Init_Win_bytes_backward'] = df['Init_Win_bytes_backward'].astype(float)
df['act_data_pkt_fwd'] = df['act_data_pkt_fwd'].astype(float)
df['min_seg_size_forward'] = df['min_seg_size_forward'].astype(float)
df['Active Mean'] = df['Active Mean'].astype(float)
df['Active Min'] = df['Active Min'].astype(float)

df['Total Length of Bwd Packets'] = df['Total Length of Bwd Packets'].astype(float)
df['Total Fwd Packets'] = df['Total Fwd Packets'].astype(float)
df['Fwd Packet Length Mean'] = df['Fwd Packet Length Mean'].astype(float)
df['Bwd Packet Length Mean'] = df['Bwd Packet Length Mean'].astype(float)
df['Bwd Packet Length Std'] = df['Bwd Packet Length Std'].astype(float)
df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(float)
df['Flow IAT Std'] = df['Flow IAT Std'].astype(float)
df['Fwd IAT Std'] = df['Fwd IAT Std'].astype(float)
df['Bwd IAT Mean'] = df['Bwd IAT Mean'].astype(float)
df['Bwd IAT Std'] = df['Bwd IAT Std'].astype(float)
df['Bwd IAT Max'] = df['Bwd IAT Max'].astype(float)
df['Fwd PSH Flags'] = df['Fwd PSH Flags'].astype(float)
df['Bwd PSH Flags'] = df['Bwd PSH Flags'].astype(float)
df['Fwd URG Flags'] = df['Fwd URG Flags'].astype(float)
df['Bwd URG Flags'] = df['Bwd URG Flags'].astype(float)
df['Bwd Header Length'] = df['Bwd Header Length'].astype(float)
df['Packet Length Mean'] = df['Packet Length Mean'].astype(float)
df['Packet Length Variance'] = df['Packet Length Variance'].astype(float)
df['FIN Flag Count'] = df['FIN Flag Count'].astype(float)
df['SYN Flag Count'] = df['SYN Flag Count'].astype(float)
df['RST Flag Count'] = df['RST Flag Count'].astype(float)
df['PSH Flag Count'] = df['PSH Flag Count'].astype(float)
df['URG Flag Count'] = df['URG Flag Count'].astype(float)
df['CWE Flag Count'] = df['CWE Flag Count'].astype(float)
df['ECE Flag Count'] = df['ECE Flag Count'].astype(float)
df['Average Packet Size'] = df['Average Packet Size'].astype(float)
df['Avg Fwd Segment Size'] = df['Avg Fwd Segment Size'].astype(float)
df['Avg Bwd Segment Size'] = df['Avg Bwd Segment Size'].astype(float)
df['Active Std'] = df['Active Std'].astype(float)
df['Active Max'] = df['Active Max'].astype(float)
df['Idle Mean'] = df['Idle Mean'].astype(float)
df['Idle Std'] = df['Idle Std'].astype(float)
df['Idle Max'] = df['Idle Max'].astype(float)
df['Idle Min'] = df['Idle Min'].astype(float)


df.dropna(inplace=True)
df.columns
df.Label.unique()
df['Label'] = df['Label'].replace('BENIGN', '0')
df['Label'] = df['Label'].replace('DrDoS_DNS', '1')
df['Label'] = df['Label'].replace('LDAP', '1')
df['Label'] = df['Label'].replace('DrDoS_MSSQL', '2')
df['Label'] = df['Label'].replace('MSSQL', '2')
df['Label'] = df['Label'].replace('DrDoS_NTP', '3')
df['Label'] = df['Label'].replace('NetBIOS', '4')
df['Label'] = df['Label'].replace('DrDoS_SNMP', '5')
df['Label'] = df['Label'].replace('DrDoS_SSDP', '6')
df['Label'] = df['Label'].replace('DrDoS_UDP', '6')
df['Label'] = df['Label'].replace('UDP', '6')
df['Label'] = df['Label'].replace('Syn', '7')

df['Label'] = df['Label'].astype(float)
df.Label.unique()


df.dropna(inplace=True)
df = df[np.isfinite(df).all(1)]


dns = pd.read_csv('New folder/processed-dataset/dns.csv')
ldap = pd.read_csv('New folder/processed-dataset/ldap.csv')
mssql = pd.read_csv('New folder/processed-dataset/mssql.csv')
netbios = pd.read_csv('New folder/processed-dataset/netbios.csv')
ntp = pd.read_csv('New folder/processed-dataset/ntp.csv')
snmp = pd.read_csv('New folder/processed-dataset/snmp.csv')
ssdp = pd.read_csv('New folder/processed-dataset/ssdp.csv')
syn = pd.read_csv('New folder/processed-dataset/syn.csv')
udp = pd.read_csv('New folder/processed-dataset/udp.csv')

ds = pd.concat([dns,ldap, mssql, netbios, ntp, snmp, ssdp, syn, udp])
ds = sklearn.utils.shuffle(ds)
print(ds.Label.value_counts())
ds.to_csv('New folder/processed-dataset/cicdos2019-multi.csv', index=False)


ds = pd.read_csv('New folder/processed-dataset/cicdos2019-multi.csv')

ds.dropna(inplace=True)
ds = ds[np.isfinite(ds).all(1)]

part_80 = ds.sample(frac = 0.8)
rest_part_20 = ds.drop(part_80.index)

part_80.to_csv('New folder/processed-dataset/cicdos2019-multi-train.csv', index=False)
rest_part_20.to_csv('New folder/processed-dataset/cicdos2019-multi-test.csv', index=False)

df.to_csv('New folder/processed-dataset/dns.csv', index=False)

print(datetime.now()-start)