# Assignment 1

## What the data is all about ?
 <p>In this study, we investigate how an organism’s codon usage bias levels can serve as a predictor and classifier of various genomic and evolutionary features across the three kingdoms of life (archaea, bacteria, eukarya).</p>

 We build several machine learning models trained over an existing [dataset](https://archive.ics.uci.edu/ml/datasets/Codon+usage "codon_usage") containing about 13,000 organisms that show it is possible to accurately predict an organism’s DNA type (nuclear, mitochondrial, chloroplast) and taxonomic identity simply using its genetic code (64 codon usage frequencies).

## What benefits you might hope to get from data mining ?
 By leveraging machine learning methods to accurately identify evolutionary origins and genetic composition from codon usage patterns, our study suggests that the genetic code can be utilized to train accurate machine learning classifiers of taxonomic and phylogenetic features.

## Discussion the data quality issue 

* ### Problems with the data set
    1. Data set contains missing values
    2. Some genomic entries do not have suffient number of codon to make a reliable prediction
       1. Class distributions for Kingdom and DNAtype class are highly skewed.
           ```python
           import pandas as pd
           df = pd.read_csv('../../codon-usage-data-set/codon_usage.csv')

           df.groupby('Kingdom').size()
           df.groupby('DNAtype').size()
           ```
           | DNAtype  |   Frequency |     | Kingdom | Frequency |                     
           |-------    |-----|---------| ----- | ---- |
           | 0        |      9267  |     | arc     | 126     |
           | 1        |       2899  |     | bct     | 2920    | 
           | 2        |        816 |     | inv     | 1345    |
           | 3        |          2 |     | mam     | 572     |
           | 4        |         31 |     | phg     | 220     |
           | 5        |          2 |     | plm     | 18      |
           |6        |           1 |     | pln     | 2523    |
           |7        |           1  |     | pri     | 180     | 
           | 9        |           2  |     | rod     | 215     | 
           | 11       |          2  |     | vrl     | 2832    |
           | 12       |           5  |     |  vrt | 2077    |

           As we can clearly observe DNAtypes 3-12 are infrequent and may not produce reliable predictions

           Some of the kingdom classes for ex 'arc'(archea), 'plm'(bacterial plasmid) are relatively infrequent

* ### Appropriate responses to above data quality problems
    
    1. Discard all the missing value observations (only two observations had missing values)
    2. Discard the genome entries with `Ncodons` less than 1000
    3. To deal with the highly imbalanced class distribution

        1. Re-classify and harmonize genome entries from the `'Kingdom'` column with values 'xxx' (where ‘xxx’ is one of ‘pln’, ‘inv’, ‘vrt’, ‘mam’, ‘rod’, or ‘pri’) as ‘euk’ (eukaryotes) because these Kingdoms are part of 'euk' family

        1. Identify the DNA type of the eukaryotic genomes as either 0 (nuclear), 1 (mitochondrion), 2 (chloroplast), 3 (cyanelle), 4 (plastid), 5 (nucleomorph), 6 (secondary endosymbiont), 7 (chromoplast), 8 (leukoplast), 9 (NA), 10 (proplastid), 11 (apicoplast), 12 (kinetoplast). Remove any rows that are not 0, 1, or 2 (in other words, avoid any DNA types specified by the integers greater than 2).

        1. Exclude the genome entries classified as ‘plm’ (mostly to avoid imbalanced classes in our machine learning models, since there are only 18 plasmids).

# Assignment 2

## *Apriori* Algorithm Implementation (Python 3)

### 1. Apriori Algorithm
Apriori is given by R. Agrawal and R. Srikant in 1994 for frequent item set mining and association rule learning. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often.</br>

This implementation of *Apriori* Algorithm is based on **Hash-Tree Data Structure** to efficiently do support-counting of candidate itemsets

### 2. Improvement over the usual *Apriori*-Algorithm (for Frequent Itemset Generation)

We use AprioriTID algorithm as an improvement over usual apriori algorithm

In this algorithm database D is not used for counting support after the first pass. Rather a set C̅<sub>k</sub> is used for this purpose.

Each member of the set C̅<sub>k</sub> is of the form < TID, {X<sub>k</sub>} >,
where each X<sub>k</sub> is a potentially large k-itemset present in the transaction with identifier TID. 
For k = 1, C̅<sub>1</sub> corresponds to the database D, although conceptually each item i is replaced by the itemset {i}

C̅<sub>k</sub> is generated as follows 
`if (C_t != { }) then C̅_k += <t.TID, C_t>`, 
where `C_t` is set of candidate k-itemsets belonging to transaction `t`

In addition, for large values of k, each entry may be smaller than the corresponding
transaction because very few candidates may be contained in the transaction, hence a significant perfomance gain is expected.

### 3. How to use this library ?
Source code is contained in `src\rule-mining\apriori\` where `apriori` directory is a python package and you can import the following functions:

For Frequent Itemset Generation

```python
gen_freq_itemsets(transactions,min_sup=0.5,max_len=None,max_leaf_size=15,max_children=50)

gen_freq_itemsets_tid(transactions,min_sup=0.5,max_len=None)
  ```
_(see the implementation docstrings for more information about the parameters)_

For Rule Mining 

```python
gen_rules(freq_itemsets, min_conf=0.6)
```
_(see the implementation docstrings for more information about the parameters)_

## FP-Growth Algorithm Implementation (Python 3)

### 1. FP-Growth Algorithm
FP stands for frequent pattern. 
Frequent pattern discovery (or FP discovery, FP mining, or Frequent itemset mining) is part of knowledge discovery in databases, Massive Online Analysis, and data mining; it describes the task of finding the most frequent and relevant patterns in large datasets.

It is a divide-and-conquer algorithm and uses **FP-Tree Data Structure** to compress the database in a more compact tree-representation and using it to generate frequent itemsets in just two passes over the database.
This algorithm in general runs much faster than *Apriori* algorithm

### 2. Improvement over the usual FP-Growth

We use **Projected Databases** method to improve upon the usual FP-*growth* algorithm

FP-*growth* described previously is a main memory-based frequent pattern mining algorithm. However, when the database is large, or when the `min_sup` threshold is quite low, it is unrealistic to assume that FP-tree of a database can fit into main memory. The *Projected Database* algorithm that we describe here scales very well with large databases 

*Definition* (Projected Database). Let a<sub>i</sub> be a frequent item in a transaction database, DB. The a<sub>i</sub>-projected database for a<sub>i</sub>
is derived from DB by collecting all the transactions containing a<sub>i</sub> and removing from
them (1) infrequent items, (2) all frequent items after a<sub>i</sub> in the list of frequent items, and
(3) a<sub>i</sub> itself.

*Alogrithm* : Partition the DB into a set of projected DBs and then for each p-projected database, where p is a frequent item from DB, construct p-conditional FP-tree and mine frequent patterns from it.



### 3. How to use this library
Source code is contained in `src\rule-mining\fpgrowth\` where `fpgrowth` directory is a python package and you can import the following functions:

For Frequent Itemset Generation

```python
gen_freq_itemsets(transactions, null_label=None, min_sup=0.5)
  ```
_(see the implementation docstrings for more information about the parameters)_

For Frequent Itemset Generation with Projected DBs

```python
gen_freq_itemsets_projected_DB(transactions, null_label=None, min_sup=0.5)
```
_(see the implementation docstrings for more information about the parameters)_

## About the dataset used
The following dataset was donated by Tom Brijs and contains the (anonymized) retail market basket data from an anonymous Belgian retail store.
The data are provided ’as is’.

More details can be found [here](http://fimi.uantwerpen.be/data/retail.pdf).

## Remarks
Except for standard Python 3 libraries, no other libraries have been used for above implementations.
## References
1. *Introduction to Data Mining*, by P.-N. Tan, M. Steinbach, V. Kumar, Addison-Wesley.
2. *Fast Algorithms for Mining Association Rules*, by Rakesh Agrawal, Ramakrishnan Srikant, IBM Almaden Research Center
3. *Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern Tree Approach*, by Jiawei Han, Jian Pei, Yiwen Yin, Runying Mao
