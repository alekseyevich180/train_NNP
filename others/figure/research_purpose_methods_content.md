# Research Objective, Methods, and Content

## Research Objective

This study aims to develop an active-learning workflow for selecting high-value atomic configurations from AIMD trajectories of a ZnO(10-10) surface interacting with C9 ketone isomers, water, and molecular oxygen. Instead of transferring all trajectory frames directly into the neural network potential training pool, the workflow identifies chemically informative structures associated with bond rearrangement, Zn-O coordination changes, interfacial interactions, and feature-space novelty. The final objective is to construct a compact, diverse, and physically meaningful dataset for subsequent neural network potential training.

## Research Methods

The workflow begins with AIMD trajectory generation and representative frame sampling for the ZnO surface reaction system. Consecutive trajectory frames are then compared to detect bond formation and bond breaking events, which provide rule-based indicators of reactive configurations. In parallel, Zn-O coordination numbers are tracked along the trajectory to identify adsorption, desorption, and coordination-shift events at the interface.

To represent each structure in a machine-readable form, SOAP descriptors are used to encode local atomic environments. Additional vdW-related scoring terms are introduced to characterize nonbonded interactions among the organic molecule, water, and the surface. These chemical event signals, coordination-change indicators, SOAP features, and vdW interaction cues are then integrated into a neural-network structure selector. The selector is formulated as an MLP-based scoring model that ranks configurations according to their expected value for dataset construction. It is a structure-selection model, not a direct energy or force predictor and not a committee-uncertainty estimator.

## Research Content

The research content is organized into three connected layers. First, the workflow extracts chemically relevant events from AIMD trajectories by monitoring bond changes and Zn-O coordination changes. This layer captures the local reaction behavior of C9 ketone isomers, water, and oxygen near the ZnO(10-10) surface.

Second, the workflow converts trajectory frames into quantitative descriptors and ranking cues. SOAP descriptors provide a structural representation of local atomic environments, while vdW scoring highlights interfacial configurations involving organic-surface, water-surface, and organic-water interactions.

Third, the workflow uses an MLP-based structure selector to combine rule-derived labels and descriptor-based features into a unified selection score. High-value candidate structures are aggregated, deduplicated, and tracked by provenance before being organized into a final training dataset. This dataset is intended to support efficient and chemically informed neural network potential development for reactive ZnO surface systems.

