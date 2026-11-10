Requirements for protein chains in dataset

choose from PISCES from dunbrack lab:
- 25 maximum pairwise percent sequence identity
- maximum resolution 2.0 A
- Maximum R value 0.25
- Minimum chain length: 40
- Maximum chain length: 500
- Only include entries from X-ray

1. labels turned 8 structures to 3 structures
2. class imbalance in class
3. Post processing CRF/HMM step


Path:
1. Create dataframe with pbd_id, chain_id, sequence, structure
2. Create onehot encoding of sequence with each of the 20 and any noncanonical being X
3. train test split
4. 

use pytorch as its the only one with CNN

1. window of 15 one-hot softmax regression in scikit-learn
2. then do a MLP
3. then do a 1d CNN
4. post processing transition matrix