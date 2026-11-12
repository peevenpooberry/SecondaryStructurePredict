# Recreating the Dataset

To run the `protein_dataset.ipynb` notebook:
1. create a new conda environment:
```
cd Dataset_Creation
conda env create -f protein_dataset_env.yml
conda activate protein_dataset_env
```
2. In the notebook choose `protein_dataset_env` as the selected kernel
3. Run the cells of the notebook either with the `selected_chains.txt` or provide your own text document with PDB IDs.