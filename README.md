# GARI
This work introduces a simple project called GARI (Genetic Algorithm for Reproducing Images).
GARI reproduces a single image using Genetic Algorithm (GA) by evolving pixel values.

This project works with both color and gray images without any modifications.
Just give the image path.
Using three parameters, we can customize it to statisfy our need. 
The parameters are:
    1) Population size. I.e. number of individuals pepr population.
    2) Mating pool size. I.e. Number of selected parents in the mating pool.
    3) Mutation percentage. I.e. number of genes to change their values.

Value encoding used for representing the input.
Crossover is applied by exchanging half of genes from two parents.
Mutation is applied by randomly changing the values of randomly selected 
predefined percent of genes from the parents chromosome.

This project is implemented using Python 3.5 by Ahmed F. Gad.
Contact info:
ahmed.fawzy@ci.menofia.edu.eg
https://www.linkedin.com/in/ahmedfgad/
