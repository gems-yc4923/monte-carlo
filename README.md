# Project Name: Metropolis Monte Carlo algorithm

This is my project that can run a Monte Carlo Simulation on a system consisting of a 2 Dimensional Lattice.

_note_: I have created a testing notebook file. It serves as proof that I have written all my code by myself. It also has an animation plot that I did not want to include
for fear of time and ressource consumption.
For the optimization task, I did not put any documentations to the steps I took. However, a comparison between my final file and the testing ground should be able to show the steps I took.
OPTIMIZATION:
I have managed to make the driver.drive() simulation run from 3 minutes 6 seconds to 1 minute 6 seconds, saving 2 minutes of time.

## Installation

To install the required packages necessary to run this file, please run the following command
on an empty environment:

```bash
pip install .
```

Alternatively, you can install the environment that is provided by running the following command:

```bash
conda env create -f environment.yml
```

Then proceed to activate the environemnt

```bash
conda activate c_yassine2
```

Then, navigate to the notebook file named _magnetic-skyrmion.ipynb_ and run the code.
The environment should automatically install the `mcsim` package, and you
should be able to run it.

## Visualization

Here's an example plotting of a simulation algorithm.
The initital plot of the state (Randomized) looks like this:

<img
  src="my-research\initialstate.png"
  alt="Initial"
  title="Initial Random State of the Lattice"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

  And this is the final plot of the state after running the Monte Carlo Simulation:

<img
  src="my-research\finalstate.png"
  alt="#Finalized"
  title="Initial Random State of the Lattice"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
