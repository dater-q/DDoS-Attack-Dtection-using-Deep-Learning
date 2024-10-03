from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

bin_overallacc = [0.9986,0.9986,0.9996,0.9996,0.9996,0.9996,0.9996,0.9996,0.9997,0.9997]
bin_overallvalacc = [0.9996,0.9996,0.9996,0.9996,0.9996,0.9996,0.9997,0.9997,0.9996,0.9996]
bin_overallloss = [0.0053,0.0048,0.0042,0.0035,0.0022,0.0020,0.0019,0.0017,0.0016,0.0014]
bin_overallvalloss = [0.0021,0.0021,0.0018,0.0018,0.0017,0.0017,0.0016,0.0016,0.0015,0.0015]

multi_overallacc = [0.8943,0.9063,0.9086,0.9099,0.9107,0.9113,0.9119,0.9120,0.9123,0.9126,0.9129,0.9129,0.9130,0.9132,0.9140,0.9143,0.9144,0.9140,0.9143,0.9145,0.9147,0.9147,0.9151,0.9152,0.9153,0.9154,0.9155,0.9162,0.9159]
multi_overallvalacc = [0.8930,0.9036,0.8742,0.9091,0.9138,0.9131,0.9153,0.9073,0.9156,0.9134,0.9110,0.9141,0.9047,0.9124,0.9128,0.9157,0.9155,0.9112,0.9184,0.9121,0.9037,0.8977,0.9146,0.9052,0.9129,0.9098,0.9086,0.9081,0.9084]
multi_overallloss = [0.3250,0.2875,0.2813,0.2783,0.2757,0.2742,0.2725,0.2722,0.2711,0.2703,0.2688,0.2689,0.2684,0.2676,0.2656,0.2649,0.2650,0.2658,0.2644,0.2641,0.2637,0.2635,0.2628,0.2619,0.2627,0.2623,0.2619,0.2601,0.2605]
multi_overallvalloss = [0.3235,0.2940,0.3901,0.2765,0.2702,0.2681,0.2664,0.2944,0.2676,0.2691,0.2803,0.2637,0.2914,0.2672,0.2652,0.2581,0.2614,0.2744,0.2560,0.2693,0.2855,0.2973,0.2635,0.2889,0.2717,0.2770,0.2834,0.2574,0.2564]

gr_overallacc = [0.9565,0.9659,0.9675,0.9687,0.9695,0.9695,0.9700,0.9701,0.9706,0.9709,0.9713,0.9714,0.9714,0.9715,0.9720,0.9719,0.9719,0.9722,0.9723,0.9722]
gr_overallvalacc = [0.9565,0.9515,0.9710,0.9713,0.9580,0.9743,0.9749,0.9750,0.9720,0.9746,0.9690,0.9744,0.9536,0.9640,0.9715,0.9717,0.9739,0.9708,0.9724,0.9725]
gr_overallloss = [0.1399,0.1140,0.1095,0.1071,0.1048,0.1049,0.1043,0.1032,0.1021,0.1015,0.1006,0.1004,0.1003,0.1001,0.0989,0.0990,0.0990,0.0981,0.0980,0.0981]
gr_overallvalloss = [0.1269,0.1305,0.1060,0.1022,0.1095,0.0977,0.0975,0.0967,0.1011,0.0988,0.1013,0.0926,0.1473,0.1138,0.1009,0.0976,0.0943,0.1035,0.0984,0.0984]


plt.plot(bin_overallacc)
plt.plot(bin_overallvalacc)
plt.ylim([0.0, 1.00])
plt.title('Exactitude du model - Classification binaire')
plt.ylabel('Exactitude')
plt.xlabel('Epoque')
plt.legend(['Entrainement','Validation'], loc='upper left')
plt.show()

plt.plot(bin_overallloss)
plt.plot(bin_overallvalloss)
plt.ylim([0.0, 1.0])
plt.title('Perte du model - Classification binaire')
plt.ylabel('Perte')
plt.xlabel('Epoque')
plt.legend(['Entrainement','Validation'], loc='upper left')
plt.show()

plt.plot(multi_overallacc)
plt.plot(multi_overallvalacc)
plt.ylim([0.0, 1.00])
plt.title('Exactitude du model - Classification 8 classes')
plt.ylabel('Exactitude')
plt.xlabel('Epoque')
plt.legend(['Entrainement','Validation'], loc='upper left')
plt.show()

plt.plot(multi_overallloss)
plt.plot(multi_overallvalloss)
plt.ylim([0.0, 1.0])
plt.title('Perte du model - Classification 8 classes')
plt.ylabel('Perte')
plt.xlabel('Epoque')
plt.legend(['Entrainement','Validation'], loc='upper left')
plt.show()

plt.plot(gr_overallacc)
plt.plot(gr_overallvalacc)
plt.ylim([0.0, 1.00])
plt.title('Exactitude du model - Classification 4 classes')
plt.ylabel('Exactitude')
plt.xlabel('Epoque')
plt.legend(['Entrainement','Validation'], loc='upper left')
plt.show()

plt.plot(gr_overallloss)
plt.plot(gr_overallvalloss)
plt.ylim([0.0, 1.0])
plt.title('Perte du model - Classification 4 classes')
plt.ylabel('Perte')
plt.xlabel('Epoque')
plt.legend(['Entrainement','Validation'], loc='upper left')
plt.show()