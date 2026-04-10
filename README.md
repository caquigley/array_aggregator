# array_aggregator
Code for assessing array accuracy on a set of earthquakes. Final project for GEOS694 Computational Geosciences (Bryant Chow).

## What
This code is for assessing the accuracy and precision of small-aperture seismic arrays at recording earthquakes or other seismicity.

## How
This code does the following in the specified order:
1. Grab earthquakes/events in the vicinity of a seismic array for a set time period. 
2. Compute the "real" backazimuth and slowness based on the catalog location and a given velocity model.
3. Use an STA/LTA scheme to idenfity whether the event "triggers" the array, following the parameters in the EPIC algorithm of ShakeAlert. This includes different handlings for events with multiple triggers and events with no triggers.
4. Compute array processing for a specified time window and waveform filtering around STA/LTA trigger. Current array processing algorithms incorporated are: Least Trimmed Squares (Bishop et al., 2020), Least Squares with cross-correlation (Bishop et al., 2020), and Frequency-Wavenumber (Obspy).
5. Plotting tools for comparing the catalog backazimuth and slowness to the array calculated backazimuth and slowness. These include:
- map of earthquakes in vicinity of array
- backazimuth error (catalog backazimuth - array backazimuth) as a function of the catalog backazimuth.
- slowness error (catalog slowness - array slowness) as a function of the catalog backazimuth.
- map of backazimuth error
- map of slowness error

## Why
Small aperture arrays have grown in interest in the earthquake monitoring and Earthquake Early Warning communities in recent years. This is largely due to an arrays ability to provide accurate backazimuth and slowness estimates of incoming seismic energy. This code provides the ability to assess how accurate these estimates are, check a number of input parameters, and identify systematic patterns in error and provide possible correction criteria.

## Installation

Create a fork of this repository to your github. Then, create a local repository through your terminal. For example:

```python
git clone <github_link_to_repo>
```

A conda environment with the needed dependencies are provided in environment.yaml. This can be created with the following command in your local repository:

```python
conda env create -f environment.yml
```

This will create a environment called arrayseis with the following dependencies:
- python3
- obspy
- pygmt
- numpy (2.14)
- lts_array
    - numba
## Usage

Once your python environment is installed, this can be activated as:

```python
conda activate arrayseis
```

The input parameters for the code can be found in the input_parameters.yaml file. The possible inputs are found in the documentation

In your local repository, the main code can be run as:

```python
python array_aggregator.py input_parameters.yaml
```


## Example
An example from an array in the Aleutian Islands is provided in input_parameters.yaml.

## Project Task 1
For project task 1, I will be implementing parallel/concurrency into my code.

## Project Task 2
For project task 2, I will be doing a parameter input system. In addition, I will also either incorporate Chinook Compatability or Advanced Documentation.

## For class reviewers
The relevant python file for class reviews is array_aggregator.py. The other main file is array_functions.py, but I'll leave this up to your discretion.
