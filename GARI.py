import numpy
import matplotlib.pyplot
import itertools
import functools
import operator
import random

"""
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
"""

def img2chromosome(img_arr):
    """
    First step in GA is to represent/encode the input as a sequence of characters.
    The encoding used is value encoding by giving each gene in the 
    chromosome its actual value in the image.
    Image is converted into a chromosome by reshaping it as a single row vector.
    """
    chromosome = numpy.reshape(a=img_arr, 
                               newshape=(functools.reduce(operator.mul, 
                                                          img_arr.shape)))
    return chromosome

def initial_population(img_shape, n_individuals=8):
    """
    Creating an initial population randomly.
    """
    # Empty population of chromosomes accoridng to the population size specified.
    init_population = numpy.empty(shape=(n_individuals, 
                                  functools.reduce(operator.mul, img_shape)),
                                  dtype=numpy.uint8)
    for indv_num in range(n_individuals):
        # Randomly generating initial population chromosomes genes values.
        init_population[indv_num, :] = numpy.random.random(
                                functools.reduce(operator.mul, img_shape))*256
    return init_population

def chromosome2img(chromosome, img_shape):
    """
    First step in GA is to represent the input in a sequence of characters.
    The encoding used is value encoding by giving each gene in the chromosome 
    its actual value.
    """
    img_arr = numpy.reshape(a=chromosome, newshape=img_shape)
    return img_arr

def fitness_fun(target_chrom, indiv_chrom):
    """
    Calculating the fitness of a single solution.
    The fitness is basicly calculated using the sum of absolute difference 
    between genes values in the original and reproduced chromosomes.
    """
    quality = numpy.mean(numpy.abs(target_chrom-indiv_chrom))
    """
    Negating the fitness value to make it increasing rather than decreasing.
    Actually the next line adds nothing but it exists just because it is known 
    that the fitness values are increasing not decreasing.
    """
    quality = numpy.sum(target_chrom) - quality
    return quality

def cal_pop_fitness(target_chrom, pop):
    """
    This method calculates the fitness of all solutions in the population.
    """
    qualities = numpy.zeros(pop.shape[0])
    for indv_num in range(pop.shape[0]):
        # Calling fitness_fun(...) to get the fitness of the current solution.
        qualities[indv_num] = fitness_fun(target_chrom, pop[indv_num, :])
    return qualities

def select_mating_pool(pop, qualities, num_parents):
    """
    Selects the best individuals in the current generation, according to the 
    number of parents specified, for mating and generating a new better population.
    """
    parents = numpy.empty((num_parents, pop.shape[1]), dtype=numpy.uint8)
    for parent_num in range(num_parents):
        # Retrieving the best unselected solution.
        max_qual_idx = numpy.where(qualities == numpy.max(qualities))
        max_qual_idx = max_qual_idx[0][0]
        # Appending the currently selected 
        parents[parent_num, :] = pop[max_qual_idx, :]
        """
        Set quality of selected individual to a negative value to not get 
        selected again. Algorithm calcululations will just make qualities >= 0.
        """
        qualities[max_qual_idx] = -1
    return parents

def crossover(parents, img_shape, n_individuals=8):
    """
    Applying crossover operation to the set of currently selected parents to 
    create a new generation.
    """
    new_population = numpy.empty(shape=(n_individuals, 
                                        functools.reduce(operator.mul, img_shape)),
                                        dtype=numpy.uint8)
    
    """
    Selecting the best previous parents to be individuals in the new generation.

    **Question** Why using the previous parents in the new population?
    It is recommened to use the previous best solutions (parents) in the new 
    generation in addition to the offspring generated from these parents and 
    not use just the offspring.
    The reason is that the offspring may not produce the same fitness values 
    generated by their parents. Offspring may be worse than their parents.
    As a result, if none of the offspring are better, the previous generations 
    winners will be reselected until getting a better offspring.
    """
    #Previous parents (best elements).
    new_population[0:parents.shape[0], :] = parents


    # Getting how many offspring to be generated. If the population size is 8 and number of parents mating is 4, then number of offspring to be generated is 4.
    num_newly_generated = n_individuals-parents.shape[0]
    # Getting all possible permutations of the selected parents.
    parents_permutations = list(itertools.permutations(iterable=numpy.arange(0, parents.shape[0]), r=2))
    # Randomly selecting the parents permutations to generate the offspring.
    selected_permutations = random.sample(range(len(parents_permutations)), 
                                          num_newly_generated)
    
    comb_idx = parents.shape[0]
    for comb in range(len(selected_permutations)):
        # Generating the offspring using the permutations previously selected randmly.
        selected_comb_idx = selected_permutations[comb]
        selected_comb = parents_permutations[selected_comb_idx]
        
        # Applying crossover by exchanging half of the genes between two parents.
        half_size = numpy.int32(new_population.shape[1]/2)
        new_population[comb_idx+comb, 0:half_size] = parents[selected_comb[0], 
                                                             0:half_size]
        new_population[comb_idx+comb, half_size:] =  parents[selected_comb[1], 
                                                             half_size:]
    
    return new_population

def mutation(population, mut_percent):
    """
    Applying mutation by selecting a predefined percent of genes randomly.
    Values of the randomly selected genes are changed randmly.
    """
    for idx in range(num_parents_mating, population.shape[0]):
        # A predefined percent of genes are selected randomly.
        rand_idx = numpy.uint32(numpy.random.random(size=numpy.uint32(mut_percent/100*population.shape[1]))
                                                    *population.shape[1])
        # Changing the values of the selected genes randomly.
        new_values = numpy.uint8(numpy.random.random(size=rand_idx.shape[0])*256)
        # Updating population after mutation.
        population[idx, rand_idx] = new_values
    return population

def save_images(curr_iteration, qualities, new_population, im_shape, 
                save_point, save_dir):
    """
    Saving best solution in a given generation as an image in the specified directory.
    Images are saved accoirding to stop points to avoid saving images from 
    all generations as saving mang images will make the algorithm slow.
    """
    if(numpy.mod(curr_iteration, save_point)==0):
        # Selecting best solution (chromosome) in the generation.
        best_solution_chrom = new_population[numpy.where(qualities == 
                                                         numpy.max(qualities))[0][0], :]
        # Decoding the selected chromosome to return it back as an image.
        best_solution_img = chromosome2img(best_solution_chrom, im_shape)
        # Saving the image in the specified directory.
        matplotlib.pyplot.imsave(save_dir+'solution_'+str(curr_iteration)+'.png', best_solution_img)

def show_indivs(individuals, im_shape):
    """
    Show all individuals as image in a single graph.
    """
    num_ind = individuals.shape[0]
    fig_row_col = 1
    for k in range(1, numpy.uint16(individuals.shape[0]/2)):
        if numpy.floor(numpy.power(k, 2)/num_ind) == 1:
            fig_row_col = k
            break
    fig1, axis1 = matplotlib.pyplot.subplots(fig_row_col, fig_row_col)

    curr_ind = 0
    for idx_r in range(fig_row_col):
        for idx_c in range(fig_row_col):
            if(curr_ind>=individuals.shape[0]):
                break
            else:
                curr_img = chromosome2img(individuals[curr_ind, :], im_shape)
                axis1[idx_r, idx_c].imshow(curr_img)
                curr_ind = curr_ind + 1
