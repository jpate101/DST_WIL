# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:26:26 2020

@author: User
"""


if __name__=="__main__":
    for i in range(1, 60000):
        output_file = open("old_data\T5v2\outputs\out"+str(i)+".mlt","r") 
        
        output = output_file.readline()
        #print(output)
        if "BLANK" in output:
            print(i)
        
        ## BLANK
        
        output_file.close() 
        pass
    