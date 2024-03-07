# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:09:44 2020

@author: User
"""

import math
import numpy 
import pandas 

displaced_v = [0.0625,0.125,0.25,0.5]
length = [1,2.125,2.375,3.5]
draft = [.2,.1,.3]
offset_s1 = [0,.2,.4,.6,.8,1]

count = 0


for i1 in displaced_v:
    for i2 in length:
        for i3 in draft:
            for y0 in offset_s1:
                for y1 in offset_s1:
                    for y2 in offset_s1:
                        
                        count = count + 1                        
                        filename = ("old_data\T1v1/inputs/input%d.mlt" % count)
                        f = open(filename, "x")
                        f = open(filename, "w")
                        f.write("""
# =================== INPUT FILE TYPE AND SUBTYPE ======================
# Input File Type (0=Standard)
0
# Input File Subtype (0=Standard)
0
# =================== OUTPUT FILE TYPE AND SUBTYPE =====================
# Output File Type (0=Standard)
0
# Output File Subtype (0=Standard)
0
# ====================== COURSE AND VESSEL TYPE ========================
# Course Particulars (0=None)
0
# Number of Hulls (1, 2,..., or 5)
1
# ======================== PHYSICAL QUANTITIES =========================
# Gravitational Acceleration (m/sec/sec) (min 9.6, max 9.9)
9.80665
# ========================= WATER PROPERTIES ===========================
# Water Density (kg/cubic metre) (min 995.0, max 1030.0)
1025.9
# Water Kin. Viscosity (sq. m/sec * 10^-6) (min 0.5, max 2.0)
1.18831
# Base Eddy Kin. Viscosity (non-dimensional, min 1.0)
10.0
# Water Depth (metres) (max=10000.0)
10000.0
# ========================== AIR PROPERTIES ============================
# Air Density (kg/cubic metre) (min 0.9, max 2.0)
1.226
# Air Kin. Viscosity (sq. m/sec * 10^-6) (min 10.0, max 20.0)
14.4
# Wind Speed (m/sec)
0.0
# Wind Direction (degrees)
0.0
# ======================= CALCULATION PARAMETERS =======================
# Minimum Speed (m/sec) (min 0.01, max 51.9)
1.0
# Maximum Speed (m/sec) (max 52.0)
25
# Number of Speeds (min 2, max 101)
25
# Leeway Parameters (0=None)
0
# Wave Drag Ntheta
512
# Skin Friction Method (0=None, 1=ITTC1957, 2=Grigson)
1
# Viscous Form Factor Method (0=None, 3=Dual)
3
# Viscous Drag Form Factor
1.0
# Wave Drag Form Factor
1.0
# Pressure Signature Method (0=None,1=Slender body)
1
# ==================== SHIP CALCULATION PARAMETERS =====================
# Number of Offset Stations (rows) (odd integer: min 5, max 81)
81
# Number of Offset Waterlines (columns) (odd integer: min 5, max 81)
81
# Ship Loading Type
3
# Ship Loading Formula Parameters
1.0,0.0,0.0
# ===================== WAVE ELEVATION PARAMETERS ======================
# Sectorial Cuts and Patches
# R0
5.0
# R1
20.0
# Beta
22.5
# Nr
101
# Nbeta
101
# Rectangular Cuts and Patches
# x0
5.0
# x1
20.0
# y0
-7.5
# y1
7.5
# Nwx
101
# Nwy
101
# Beaches and Walls
# x0
5.0
# x1
20.0
# y0
7.5
# z0
-5.0
# z1
0.0
# Slope
90.0
# Nbx
2
# Nbz
2
# ============================ FIRST HULL ==============================
# Offsets
1, %f, %f, %f
# Displacement Volume (cubic metres)
%f
# Length (metres)
%f
# Draft (metres)
%f
# Longitudinal Separation (metres) (0.0 for a monohull)
0.0
# Lateral Separation Distance (metres) (0.0 for a monohull)
0.0
# Loading Type for this hull
3
# Loading Formula Parameters
1.0,0.0,0.0
# Trim Method
0
# Trim: Number of speeds ( >= 2)
2
# Trim: speed, angle
0.0,0.0
40.0,0.0
# Sinkage Method
0
# Sinkage: Number of speeds ( >= 2)
2
# Sinkage: speed, amount
0.0,0.0
40.0,0.0
# Heel Method
0
# Heel: Number of speeds ( >= 2)
2
# Heel: speed, angle
0.0,0.0
40.0,0.0
# Appendages (0=None)
0
# Other Particulars (0=None)
0
#end
                    """ % (y0,y1,y2,i1, i2, i3))
                    f.close()
print(count)               
    