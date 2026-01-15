README #### Task 2 

– NN surrogate with experimental validation

This version improves the baseline Task-2 model by:

- Increasing network depth from 4 → 5,6,7,8 layers
- Adding experimental comparison using Ishihara data
- Plotting NN-predicted displacement on the deformed mesh
- Computing NN – EXP error in the deformed configuration
- Using KD-Tree matching between experimental and NN nodes
- Automatically saving all result figures

## Files
- `nn_group20.py` – training + evaluation + plotting
- `results_nn_task2/` – generated figures

## Plots
- True vs predicted ux, uy  
- Experimental vs predicted ux, uy  
- NN ux, uy in deformed configuration  
- Error (NN − EXP) in deformed configuration"""

