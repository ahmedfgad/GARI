# GARI
GARI (Genetic Algorithm for Reproducing Images) is a Python project that uses the PyGAD library for reproducing images using the genetic algorithm. GARI reproduces a single image using Genetic Algorithm (GA) by evolving pixel values. This project works with both color and gray images.

For implementing the genetic algorithm, the PyGAD library is used. Check its documentation here: https://pygad.readthedocs.io

**IMPORTANT** *If you are coming for the code of the tutorial [Reproducing Images using a Genetic Algorithm with Python](https://heartbeat.fritz.ai/reproducing-images-using-a-genetic-algorithm-with-python-91fc701ff84), then it has been moved to the [TutorialProject](https://github.com/ahmedfgad/GARI/tree/master/TutorialProject) directory on 18 May 2020.*

# PyGAD Installation

To install [PyGAD](https://pypi.org/project/pygad), simply use pip to download and install the library from [PyPI](https://pypi.org/project/pygad) (Python Package Index). The library lives a PyPI at this page https://pypi.org/project/pygad.

For Windows, issue the following command:

```python
pip install pygad
```

For Linux and Mac, replace `pip` by use `pip3` because the library only supports Python 3.

```python
pip3 install pygad
```

PyGAD is developed in Python 3.7.3 and depends on NumPy for creating and manipulating arrays and Matplotlib for creating figures. The exact NumPy version used in developing PyGAD is 1.16.4. For Matplotlib, the version is 3.1.0.

# ImageIO Installation

To install [ImageIO](https://pypi.org/project/imageio/), simply use pip to download and install the library from [ImageIO](https://pypi.org/project/imageio/) (Python Package Index). The library lives a PyPI at this page https://pypi.org/project/imageio/.

For Windows, issue the following command:

```python
pip install imageio
```

For Linux and Mac, replace `pip` by use `pip3` because the library only supports Python 3.

```python
pip3 install imageio
```

Imageio is a Python library that provides an easy interface to read and write a wide range of image data, including animated images, volumetric data, and scientific formats. It is cross-platform, runs on Python 3.5+, and is easy to install.


# Project Steps

The steps to follow in order to reproduce an image are as follows:

- Read an image
- Prepare the fitness function
- Create an instance of the pygad.GA class with the appropriate parameters
- Run PyGAD
- Plot results
- Calculate some statistics

The next sections discusses the code of each of these steps.

## Read an Image

There is an image named `fruit.jpg` in the project which is read according to the next code.

```python
import imageio
import numpy

target_im = imageio.imread('fruit.jpg')
target_im = numpy.asarray(target_im/255, dtype=numpy.float)
```

Here is the read image.

![fruit](https://user-images.githubusercontent.com/16560492/36948808-f0ac882e-1fe8-11e8-8d07-1307e3477fd0.jpg)

Based on the chromosome representation used in the example, the pixel values can be either in the 0-255, 0-1, or any other ranges. 

Note that the range of pixel values affect other parameters like the range from which the random values are selected during mutation and also the range of the values used in the initial population. So, be consistent.

## Prepare the Fitness Function

The next code creates a function that will be used as a fitness function for calculating the fitness value for each solution in the population. This function must be a maximization function that accepts 2 parameters representing a solution and its index. It returns a value representing the fitness value.

The fitness value is calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes. The `gari.img2chromosome()` function is called before the fitness function to represent the image as a vector because the genetic algorithm can work with 1D chromosomes. 

For more information about preparing the fitness function in PyGAD, please read the [PyGAD's documentation](https://pygad.readthedocs.io).

```python
target_chromosome = gari.img2chromosome(target_im)

def fitness_fun(solution, solution_idx):
    fitness = numpy.sum(numpy.abs(target_chromosome-solution))

    # Negating the fitness value to make it increasing rather than decreasing.
    fitness = numpy.sum(target_chromosome) - fitness
    return fitness
```

## Create an Instance of the `pygad.GA` Class

It is very important to use random mutation and set the `mutation_by_replacement` to `True`. Based on the range of pixel values, the values assigned to the `init_range_low`, `init_range_high`, `random_mutation_min_val`, and `random_mutation_max_val` parameters should be changed.

If the image pixel values range from 0 to 255, then set `init_range_low` and `random_mutation_min_val` to 0 as they are but change `init_range_high` and `random_mutation_max_val` to 255.

Feel free to change the other parameters or add other parameters. Please check the [PyGAD's documentation](https://pygad.readthedocs.io) for the full list of parameters. 

```python
import pygad

ga_instance = pygad.GA(num_generations=20000,
                       num_parents_mating=10,
                       fitness_func=fitness_fun,
                       sol_per_pop=20,
                       num_genes=target_im.size,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=0.01,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0)
```

## Run PyGAD

Simply, call the `run()` method to run PyGAD.

```python
ga_instance.run()
```

## Plot Results

After the `run()` method completes, the fitness values of all generations can be viewed in a plot using the `plot_result()` method.

```python
ga_instance.plot_result()
```

Here is the plot after 20,000 generations.

![Fitness Values](https://user-images.githubusercontent.com/16560492/82232124-77762c00-992e-11ea-9fc6-14a1cd7a04ff.png)

## Calculate Some Statistics

Here is some information about the best solution. 

```python
# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

result = gari.chromosome2img(solution, target_im.shape)
matplotlib.pyplot.imshow(result)
matplotlib.pyplot.title("PyGAD & GARI for Reproducing Images")
matplotlib.pyplot.show()
```

# Evolution by Generation

The solution reached after the 20,000 generations is shown below.

![solution](https://user-images.githubusercontent.com/16560492/82232405-e0f63a80-992e-11ea-984f-b6ed76465bd1.png)

After more generations, the result can be enhanced like what shown below.

![solution](https://user-images.githubusercontent.com/16560492/82232345-cf149780-992e-11ea-8390-bf1a57a19de7.png)

The results can also be enhanced by changing the parameters passed to the constructor of the `pygad.GA` class.

Here is an example of input image and how it is evolved after some iterations.

<h1>Generation 0</h1>

![solution_0](https://user-images.githubusercontent.com/16560492/36948589-b47276f0-1fe5-11e8-8efe-0cd1a225ea3a.png)

<h1>Generation 1,000</h1>

![solution_1000](https://user-images.githubusercontent.com/16560492/36948823-16f490ee-1fe9-11e8-97db-3e8905ad5440.png)

<h1>Generation 2,500</h1>

![solution_2500](https://user-images.githubusercontent.com/16560492/36948832-3f314b60-1fe9-11e8-8f4a-4d9a53b99f3d.png)

<h1>Generation 4,500</h1>

![solution_4500](https://user-images.githubusercontent.com/16560492/36948837-53d1849a-1fe9-11e8-9b36-e9e9291e347b.png)

<h1>Generation 7,000</h1>

![solution_7000](https://user-images.githubusercontent.com/16560492/36948852-66f1b176-1fe9-11e8-9f9b-460804e94004.png)

<h1>Generation 8,500</h1>

![solution_8500](https://user-images.githubusercontent.com/16560492/36948865-7fbb5158-1fe9-11e8-8c04-8ac3c1f7b1b1.png)

<h1>Generation 20,000</h1>

![solution](https://user-images.githubusercontent.com/16560492/82232405-e0f63a80-992e-11ea-984f-b6ed76465bd1.png)

# For More Information

There are different resources that can be used to get started with the building CNN and its Python implementation. 

## Tutorial: Reproduce Images with Genetic Algorithm

In 1 May 2019, I wrote a tutorial discussing this project. The tutorial is titled [**Reproducing Images using a Genetic Algorithm with Python**](https://www.linkedin.com/pulse/reproducing-images-using-genetic-algorithm-python-ahmed-gad) which is published at Heartbeat. Check it at these links:

- [Heartbeat](https://heartbeat.fritz.ai/reproducing-images-using-a-genetic-algorithm-with-python-91fc701ff84): https://heartbeat.fritz.ai/reproducing-images-using-a-genetic-algorithm-with-python-91fc701ff84
- [LinkedIn](https://www.linkedin.com/pulse/reproducing-images-using-genetic-algorithm-python-ahmed-gad): https://www.linkedin.com/pulse/reproducing-images-using-genetic-algorithm-python-ahmed-gad

[![Tutorial Cover Image](https://miro.medium.com/max/2560/1*47K2h_Zz6SQVMHW2NL-WsQ.jpeg)](https://heartbeat.fritz.ai/reproducing-images-using-a-genetic-algorithm-with-python-91fc701ff84)

## Book: Practical Computer Vision Applications Using Deep Learning with CNNs

You can also check my book cited as [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665) which discusses neural networks, convolutional neural networks, deep learning, genetic algorithm, and more.

Find the book at these links:

- [Amazon](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665)
- [Springer](https://link.springer.com/book/10.1007/978-1-4842-4167-7)
- [Apress](https://www.apress.com/gp/book/9781484241660)
- [O'Reilly](https://www.oreilly.com/library/view/practical-computer-vision/9781484241677)
- [Google Books](https://books.google.com.eg/books?id=xLd9DwAAQBAJ)

![Fig04](https://user-images.githubusercontent.com/16560492/78830077-ae7c2800-79e7-11ea-980b-53b6bd879eeb.jpg)

# Citing PyGAD - Bibtex Formatted Citation

If you used PyGAD, please consider adding a citation to the following paper about PyGAD:

```
@misc{gad2021pygad,
      title={PyGAD: An Intuitive Genetic Algorithm Python Library}, 
      author={Ahmed Fawzy Gad},
      year={2021},
      eprint={2106.06158},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

# Contact Us

* E-mail: ahmed.f.gad@gmail.com
* [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
* [Amazon Author Page](https://amazon.com/author/ahmedgad)
* [Heartbeat](https://heartbeat.fritz.ai/@ahmedfgad)
* [Paperspace](https://blog.paperspace.com/author/ahmed)
* [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
* [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
* [GitHub](https://github.com/ahmedfgad)
