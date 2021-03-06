# Code to generate Figure 4

## Plotting buildup and offset of the urgency signal

Figures 4a-b of the paper correspond to the buildup and offset (initial point, u_0) of the urgency signal.  To plot these, use
```
urgency_buildup_offset
```
at the MATLAB command line. Note that the data parts of Figures 4a-b (top) have been taken from Supplementary Figure 6b of Churchland et al., Decision-making with multiple alternatives, *Nat. Neurosci.* (2008)

## Plotting Hick's law

A cached version of the performance data is stored in the MultiAlternativeDecisions/shared/optimParams_paper folder, which is used to compute the reaction times for Figure 4c (Hick's law). If you change the folder structure on your system, make sure that you change the path in the file below.

To plot Hick's law for value-based and perceptual decisions, use
```
reactionTime
```
at the MATLAB command line. 
