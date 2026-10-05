# bargun2 and kaank3

## Establish goals
1.  For a system consisting of a single colloid in a solvent, gather data for the trajectory of the tracer colloid at various densities and temperatures.
2.  Train a NN to predict the temperature of the solvent using trajectories of the tracer.
3.  Train a NN to predict both temperature and density of the solvent using trajectories of the tracer.. 
4.  Predict the geometry of the colloid from the trajectory of the denter of mass of the tracer.
5.  Predict diffusion constant of the tracer colloid from the size of the colloid + solvent properties(temperature,density)

## Identify your Data
1. Above mentioned training data (fluid properties, trajectories etc.) will be generated using Molecular Dynamics (MD) simulations on Hoomd-Blue. For most of the goals only the trajectory of the tracer colloid will be enough. How long the trajectories should be when they are fed to NN is not clear. 

## Anticipate storage needs
1. A trajectory of 10k timesteps is long enough. 10k * 4bytes * cooridnates = 1,2MB per trajectory. We don't know how many trajectories are going to be needed for each goal but for 1K trajectory ~ 1,2 GB is needed. So the personal computers + box drive should be enough. 

## Commit to a naming scheme
1.  Have a directory for each goal. Have a separate python script for each different simulation script. -> diffusion_v02.hoomd etc. 
2.  For each trajectory have the "simulation script version", "density", "temperature" and "run_id" (we will have more than one trajectory at the same solvent properties) written on the filename. -> version02_rho0.30_kT1.50_runid15.gsd

## Outline your verfication plan
1.  Predictions attained by NN will be compared against more results from MD simulations. We will try state points that are not fed into NN such as T=1.1143 or density=0.3434

## Map out your computing resources
1.  Personal computers should suffice for getting the trajectory data. 
2.  HAL clusters gpus will probably be needed for training. 

## References

1. Leverant, C. J., Harvey, J. A., & Alam, T. M. (2020). Machine Learning-Based Upscaling of Finite-Size Molecular Dynamics Diffusion Simulations for Binary Fluids. Journal of Physical Chemistry Letters, 11(24), 10375–10381. https://doi.org/10.1021/acs.jpclett.0c03108
2. Allers, J. P., Harvey, J. A., Garzon, F. H., & Alam, T. M. (2020). Machine learning prediction of self-diffusion in Lennard-Jones fluids. Journal of Chemical Physics, 153(3). https://doi.org/10.1063/5.0011512
3. Leverant, C. J., Harvey, J. A., & Alam, T. M. (2020). Machine Learning-Based Upscaling of Finite-Size Molecular Dynamics Diffusion Simulations for Binary Fluids. Journal of Physical Chemistry Letters, 11(24), 10375–10381. https://doi.org/10.1021/acs.jpclett.0c03108
