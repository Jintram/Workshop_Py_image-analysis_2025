

I wanted to quickly revisit what was done in the original Kiviet et al. paper
to segment the E coli. This is a bit tricky since that was written in Matlab.

I had Claude translate the Matlab code to Python to be able to execute it 
easily in Python. Didn't work completely, as the matlab `edge(..)` function
with `'log'` as keyword for the method argument is hard to reproduce.