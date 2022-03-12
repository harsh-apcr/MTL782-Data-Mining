# Assignment 1

## What the data is all about ?
> <p>In this study, we investigate how an organism’s codon usage bias levels can serve as a predictor and classifier of various genomic and evolutionary features across the three kingdoms of life (archaea, bacteria, eukarya).</p>

> We build several machine learning models trained over an existing [dataset](https://archive.ics.uci.edu/ml/datasets/Codon+usage "codon_usage") containing about 13,000 organisms that show it is possible to accurately predict an organism’s DNA type (nuclear, mitochondrial, chloroplast) and taxonomic identity simply using its genetic code (64 codon usage frequencies).

## What benefits you might hope to get from data mining ?
> By leveraging machine learning methods to accurately identify evolutionary origins and genetic composition from codon usage patterns, our study suggests that the genetic code can be utilized to train accurate machine learning classifiers of taxonomic and phylogenetic features.

## Discussion the data quality issue 

* ### Problems with the data set
    1. Data set contains missing values
    2. Some genomic entries do not have suffient number of codon to make a reliable prediction
    3. Class distributions for Kingdom and DNAtype class are highly skewed.
        ```python
        import pandas as pd
        df = pd.read_csv('../../codon-usage-data-set/codon_usage.csv')

        df.groupby('Kingdom').size()
        df.groupby('DNAtype').size()
        ```
        | DNAtype  |   Frequency | Kingdom | Frequency |                     
        |-------    |   --------- | ----- | ---- |
        | 0        |      9267  |arc  |   126 |
        | 1        |       2899  |bct   | 2920 | 
        | 2        |        816 |inv   | 1345 |
        | 3        |          2 |mam   | 572 |
        | 4        |         31 |phg   |220 |
        | 5        |          2 |plm   |18 |
        |6        |           1 |pln   | 2523 |
        |7        |           1  |pri   |  180 | 
        | 9        |           2  |rod   |  215 | 
        | 11       |          2  |vrl   | 2832 |
        | 12       |           5  |vrt   | 2077 |

        As we can clearly observe DNAtypes 3-12 are infrequent and may not produce reliable predictions

        Some of the kingdom classes for ex 'arc'(archea), 'plm'(bacterial plasmid) are relatively infrequent

* ### Appropriate responses to above data quality problems
    
    1. Discard all the missing value observations (only two observations had missing values)
    2. Discard the genome entries with `Ncodons` less than 1000
    3. To deal with the highly imbalanced class distribution

        1. Re-classify and harmonize genome entries from the `'Kingdom'` column with values 'xxx' (where ‘xxx’ is one of ‘pln’, ‘inv’, ‘vrt’, ‘mam’, ‘rod’, or ‘pri’) as ‘euk’ (eukaryotes) because these `'Kingdoms'` are part of 'euk' family

        1. Identify the DNA type of the eukaryotic genomes as either 0 (nuclear), 1 (mitochondrion), 2 (chloroplast), 3 (cyanelle), 4 (plastid), 5 (nucleomorph), 6 (secondary endosymbiont), 7 (chromoplast), 8 (leukoplast), 9 (NA), 10 (proplastid), 11 (apicoplast), 12 (kinetoplast). Remove any rows that are not 0, 1, or 2 (in other words, avoid any DNA types specified by the integers greater than 2).

        1. Exclude the genome entries classified as ‘plm’ (mostly to avoid imbalanced classes in our machine learning models, since there are only 18 plasmids).

    
    



#### Task List
* [x] Decide on which data-set to work on  (Codon Usage Data Set)
* [x] Report on the data-set chosen
* [ ] Decision Tree 
* [ ] Random Forest
* [ ] Naïve Bayes Classifier
* [ ] KNN Classifier



