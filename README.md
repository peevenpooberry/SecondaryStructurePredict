# Protein Sequence to Secondary Structure Prediction

## Task Description

In structural biology the task of predicting protein structure was hailed as one of the legendary problems of the millenium. Other structures like nucleic acid, with only 4 possible bases, have very simple rules to their structure, but proteins have 20 amino acids each with their own very unique properties. This space has proven to be very hard to explore with analytical methods.

For instance a length 10 nucleic acid has $10^4$ combinations but a protein would have $10^{20}$ combinations, 16 orders of magnitude larger.

The 2024 Nobel Prize in Chemistry was awarded for deep learning methods in protein structural prediction. This new field aims to predict complex 3d teriary structure and multisystem quaternary structure. This requires very complex model architectures and legendary compute that would take forever for an undergraduate student to achieve at the TACC. I aim to do a small more managable task: predict local interactions (secondary structure) from sequence alone.

## Dataset Creation

In biology perhaps the most comprehensive database is the Protein Data Bank (PDB) containing roughly 250,000 resolved structures. Many of these structures however are low quality and as a training set are too evolutionarily related. 

Fortunately the PDB has their own reduced dataset for training that only has 25% similarity.

Additionally, the Dunbrack lab at the Fox Chase Cancer Center has created an amazing compiling tool called PISCES. That allows me to filter this dataset for the highest quality structures.

I specifically chose:
- 25 maximum pairwise percent sequence identity
- Maximum Resolution of 2.0 Å
- Maximum R value 0.25
- Minimum chain length: 40
- Maximum chain length: 500
- Only include entries from X-ray

This dataset contained 8,704 PDB IDs and Chain IDs of the structures in this repository as `selected_chains.txt`.However, I had to then download these structures from the PDB, which after 1 hour on my laptop, I had 8,586 structures.

These structure files come in a CIF format, which details the 3D coordinates of each atom in the structure and their relationships (bonds) to each other. To get secondary structure labels I had to use the program DSSP, which provides 8 labels for secondary structure.
For this project I decided to reduce this to the 3 simple labels: 
- Helix (H)
- Sheet (E)
- Loop (C)

This dataset can be found in `selected_proteins.csv`.

## Model

Classifying the secondary structure of each residue position in a sequence is a complicated task. Local neighborhoods of residues effect each other's probabilities of the 3 structural states.

For instance in a helix, every turn should be 3.6 residues, sheets are around 5-10 residues in length, and loops are around 3-10 residues long.

To solve this problem, I hot encoded each of the 20 amino acids plus a general non-conventional residue, and use a sliding window of 15 positions, which is passed into a 1D Convolutional Nerual Network with a softmax classifier, all coded in pytorch.

There remains a problem with this challenge as the class distribution is not similar. There are far more loops then helicies and sheets, so in evaluation I look at the Recall, Precision, and F1 per class.

## Files

The dataset was created using the `protein_dataset.ipynb` notebook. This uses the `selected_chains.txt` for the downloaded PDB `.cif` files contained in the `Structures/` directory I decided not to upload for size. I then use the `dssp` github repository and compiled the program using CMake and feed the path in the notebook. The final dataframe is saved in `selected_proteins.csv`.

This data is then explored in the `structure_classifier.ipynb` notebook. First hot encoding the sequence and then encoding the structure labels. Finally this is split into training and testing data. 

I then train the softmax classifier using the `softmax_ss.py` module and finally train the 1D CNN using the `CNN_ss.py` module.