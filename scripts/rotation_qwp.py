import numpy as np
import matplotlib.pyplot as plt

# freq_from_above = [873., 910., 902., 836., 777., 702., 654., 608., 592., 569., 552., 534., 502., 490., 460., 429., 0., 0., 0., 0., 0.]

# angle_from_above = [22., 27., 32., 38., 42., 46., 48., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 62., 72., 82.]


# freq_from_below = [874., 906., 899., 839., 775., 696., 656., 612., 584., 563., 0.,0.,0.,0.,0.,0.]

# angle_from_below = [22., 27., 32., 38., 42., 46., 48., 50., 51., 52., 54., 56., 58., 62., 72., 82., ]


freq_from_below3 = [186, 190, 190, 186, 179, 168, 156, 147, 141, 0, 0, 0, 0, 0, 0, 0, 0]

angle_from_below3 = [30, 34, 38, 42, 46, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 82]


freq_from_above3 = [185, 190, 190, 185, 177, 166, 151, 143, 135, 126, 0, 0, 0, 0, 0, 0, 0]

angle_from_above3 = [30, 34, 38, 42, 46, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 82]


freq_from_below1 = [1067, 1074, 1074, 1067, 1056, 1020, 962, 897, 879, 858, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0]

angle_from_below1 = [28, 30, 32, 34, 36, 40, 44, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 82]


freq_from_above1 = [1069, 1074, 1074, 1064, 1054, 1018, 962, 897, 878, 860, 838, 820, 798, 769, 750, 709, 676, 661, 0, 0, 0, 0, 0]

angle_from_above1 = [28, 30, 32, 34, 36, 40, 44, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 72, 82]



freq_from_above06 = [1839, 1845, 1810, 1730, 1637, 1579, 1545, 1509, 1478, 1441, 1395, 1361, 1325, 1283, 1238, 1202, 1172, 1133, 1098, 1052, 1013, 968, 909, 866, 0, 0, 0, 0, 0, 0]

angle_from_above06 = [28, 32, 36, 40, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 70, 72, 82]


freq_from_below06 = [1831, 1851, 1816, 1739, 1629, 1572, 1538, 1497, 1462, 1418, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

angle_from_below06 = [28, 32, 36, 40, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 82]



print len(angle_from_below06)
print len(freq_from_below06)




plt.figure()
plt.plot(angle_from_above3, freq_from_above3, "ro", label = "3mbar high rotation as start")
plt.plot(angle_from_below3, freq_from_below3, "bo", label = "3mbar high rotation as end")

plt.plot(angle_from_above1, freq_from_above1, "r+", label = "1mbar high rotation as start")
plt.plot(angle_from_below1, freq_from_below1, "b+", label = "1mbar high rotation as end")

plt.plot(angle_from_above06, freq_from_above06, "rx", label = "06mbar high rotation as start")
plt.plot(angle_from_below06, freq_from_below06, "bx", label = "06mbar high rotation as end")


plt.ylabel("Rotation [Hz]")
plt.xlabel("Angle [degrees]")
plt.legend(loc="upper right", frameon = False)
plt.grid()
plt.show()
