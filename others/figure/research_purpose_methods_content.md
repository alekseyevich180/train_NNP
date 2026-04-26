# Research Objective, Methods, and Content

## Research Objective

This study aims to establish an active-learning workflow for selecting high-value atomic configurations from AIMD trajectories of a ZnO(10-10) surface interacting with C9 ketone isomers, water, and molecular oxygen. Rather than transferring all trajectory frames directly into the neural network potential training pool, the proposed workflow identifies chemically informative configurations associated with bond rearrangement, Zn-O coordination changes, interfacial interactions, and structural novelty. The overall objective is to construct a compact, diverse, and physically meaningful dataset for the subsequent development of a neural network potential.

## Research Methods

The workflow begins with AIMD trajectory generation and representative frame sampling for the ZnO surface reaction system. Consecutive trajectory frames are compared to detect bond formation and bond breaking events, which serve as rule-based indicators of reactive configurations. In parallel, Zn-O coordination numbers are monitored throughout the trajectory to identify adsorption, desorption, and coordination-shift events at the interface.

To represent each configuration in a machine-readable form, SOAP descriptors are used to encode local atomic environments. Additional vdW-related scoring terms are introduced to characterize nonbonded interactions among the organic molecule, water, and the surface. Chemical event signals, coordination-change indicators, SOAP features, and vdW interaction cues are then integrated into a neural-network-based structure selector. This selector is formulated as an MLP scoring model that ranks configurations according to their expected value for dataset construction.

## Research Content

The research content is organized into three connected components. First, chemically relevant events are extracted from AIMD trajectories by monitoring bond changes and Zn-O coordination changes. This component captures local reaction behavior involving C9 ketone isomers, water, and oxygen near the ZnO(10-10) surface.

Second, trajectory frames are converted into quantitative descriptors and ranking cues. SOAP descriptors provide a structural representation of local atomic environments, while vdW scoring highlights interfacial configurations involving organic-surface, water-surface, and organic-water interactions.

Third, an MLP-based structure selector combines rule-derived labels and descriptor-based features into a unified selection score. High-value candidate structures are aggregated, deduplicated, and tracked by provenance before being organized into a final training dataset. The resulting dataset is designed to support efficient and chemically informed neural network potential development for reactive ZnO surface systems.

## Two-Year Research Plan

In Year 1, AIMD trajectories of the ZnO surface system will be generated and sampled, followed by chemical event detection and preliminary structure scoring based on SOAP descriptors and vdW interaction features.

In Year 2, the event signals, SOAP features, and vdW scores will be used to train an MLP-based structure selector, after which the selected structures will be deduplicated, documented, and assembled into the final NNP training dataset.

## Research Features and Originality

A key feature of this research is that it treats active learning as a chemically informed structure-selection problem rather than a simple frame-sampling procedure. By combining bond-change detection, Zn-O coordination analysis, SOAP-based structural descriptors, and vdW interaction cues, the workflow is designed to identify configurations that are both statistically diverse and chemically meaningful.

The originality of the study lies in integrating rule-based chemical event detection with an MLP-based structure selector for constructing neural network potential training data. This approach enables the training dataset to focus on reactive and interfacial configurations that are likely to be important for ZnO surface chemistry, while avoiding unnecessary inclusion of redundant AIMD frames.
