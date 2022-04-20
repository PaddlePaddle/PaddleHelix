# Protein Data Bank (PDB) Dataset - Preprocessing


## Implementation Setup
* Python==3.7
* [Biopython](https://biopython.org/wiki/Download)==1.79
* [Scipy](https://scipy.org/install/)==1.7.3

## Download
The PDB dataset can be download by listing the proteins in the file [proteins_all.txt](./proteins_all.txt) and running the python script [download_pdb.py)](./download_pdb.py).  
```
python download_pdb.py [--save_dir SAVE_DIR] [--prot_list PROT_LIST]
```
Where the arguments are:
```
    --save_dir    Where to save downloaded PDB files. (default=./PDB_files)
    --prot_list  Text file with list of proteins to be downloaded (default=./proteins_all.txt)
```


## Retreive Residue Sequences and Coordinates
```
python pdb_to_seq_coords.py [-h] [--pdb_dir PDB_DIR] [--save_dir SAVE_DIR]  
```
Where the arguments are given as:
```
    --pdb_dir   Protein PDB files directory.
    --save_dir  Where to save retrieved chain sequences and coordinates.
```

## Transform Protein Chains into Graphs
```
python create_chain_graphs.py [-h] [--cmap_thresh CMAP_THRESH] [--save_dir SAVE_DIR] [--input_dir INPUT_DIR]  
```
The arguments are: 
```
    --cmap_thresh  Threshold for contact map.
    --save_dir     Where to save generated protein chain graphs.
    --input_dir    Directory containing protein chain sequences and coordinates
```
## Generate Label Related-data
```
python create_labels_npz.py [--annot_file ANNOT_FILE] [--save_dir SAVE_DIR] 
```
The arguments are: 
```
 --annot_file ANNOT_FILE 
 --save_dir   SAVE_DIR
```
default annot_file =[./nrPDB-GO_2019.06.18_annot.tsv](nrPDB-GO_2019.06.18_annot.tsv) proposed by [[1]](#1).

## References
> <a id="1">[1]</a> 
Gligorijević, V., Renfrew, P.D., Kosciolek, T. et al. [Structure-based protein function prediction using graph convolutional networks](https://doi.org/10.1038/s41467-021-23303-9). Nat Commun 12, 3168 (2021).