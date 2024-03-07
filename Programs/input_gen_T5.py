# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:45:40 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:59:34 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:50:31 2020

@author: User
"""


import math
import numpy 
import pandas 


#[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,.05,.15,.25,.35,.45,.55,.65,.75,.85,.95]
#[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
#[0,.2,.4,.6,.8,1]

#[0,1,.175,.35,.525,.7,.875]


count = 0

"""

Shape = [0, .5, 1]
Hull_beam = [.85,.9,.95]
Keeline_depth = [1,.75,.5]
Section = [.75,.5,.25]
"""
Shape = [0, .5, 1]
Hull_beam = [.85,.9,.95]
Keeline_depth = [.75]
Section = [.75,.5,.25]

for S_1 in Shape:
    for f8 in Hull_beam:
            for f9 in Hull_beam:
                for f10 in Hull_beam:
                    for f11 in Hull_beam:
                        for f0 in Section:
                            for f1 in Section:
                                for f2 in Section:
                                    for d_in in Keeline_depth:
                                        for d_out in Keeline_depth:
                                        
                                            count = count + 1                        
                                            filename = ("old_data\T5v2\inputs/input%d.mlt" % (count))
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
20
# Number of Speeds (min 2, max 101)
20
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
20,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,0.00,0.00,0.00,0.00  
# Displacement Volume (cubic metres)
5500
# Length (metres)
135
# Draft (metres)
5.22
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
                    """ % (f0,f1,f2,d_in,d_out,d_out,d_out,d_in,f8,f9,f10,f11,S_1,S_1,S_1,S_1))
                                            f.close()
                         
print(count) 