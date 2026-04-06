# array_aggregator
Code for assessing array accuracy on a set of earthquakes. Final project for GEOS694 Computational Geosciences (Bryant Chow).

## What
This code is for assessing the accuracy and precision of small-aperture seismic arrays at recording earthquakes or other seismicity.

## How
This code does the following:
- Grab earthquakes/events in the vicinity of a seismic array for a set time period.
- Compute the "real" backazimuth and slowness based on the catalog location and a given velocity model.
- Use an STA/LTA scheme to idenfity whether the event "triggers" the array, following the parameters in the EPIC algorithm of ShakeAlert. This includes different handlings for events with multiple triggers and events with no triggers.
- Compute array processing for specified time window and waveform filtering around STA/LTA trigger. Current array processing algorithms incorporated are: Least Trimmed Squares (Bishop et al., 2020), Least Squares with cross-correlation (Bishop et al., 2020), and Frequency-Wavenumber (Obspy).
- Plotting tools for comparing the catalog backazimuth and slowness to the array calculated backazimuth and slowness.

## Why
Small aperture arrays have grown in interest in the earthquake monitoring and Earthquake Early Warning communities in recent years. This is largely due to an arrays ability to provide accurate backazimuth and slowness estimates of incoming seismic energy. This code provides the ability to assess how accurate these estimates are, check a number of input parameters, and identify systematic patterns in error and provide possible correction criteria.

## Project Task 1
For project task 1, I will be implementing parallel/concurrency into my code.

## Project Task 2
For project task 2, I will be doing a parameter input system. In addition, I will also either incorporate Chinook Compatability or Advanced Documentation

## For class reviewers
The relevant python file for class reviews is multiple_events_array.py. The other main file is array_functions.py, but I'll leave this up to your discretion.
