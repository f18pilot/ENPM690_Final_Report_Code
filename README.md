# ENPM690_Final_Report_Code

This code is for RL path determination for the Pirate model robot with a raspberry Pi4.  The playing field is a 10 x 10 grid 
with obstacles at 4,5 and 8,3 and a start position at [9,9].  The desired object is at [0,7], and the robot task is to pick it up and deliver it to [0,0].  This code finds the path from the starting position to the object.  This code is stand alone code that is not integrated into the robot control code so that it can be run without a raspberry pi computer and all its dependencies.  The actual code is written as a function in the control code. 

Three RL methods are included: basic, MC, and TD.

This program is written on python 3.7 and requires only the imported modules numpy, matplotlib, tqdm, and random.


