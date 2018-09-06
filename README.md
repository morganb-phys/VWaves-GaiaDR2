# VWAVES-GaiaDR2

## Overview

This repository includes all the code required to reproduce the results in Bennett & Bovy (2018, in prep.) which studies the vertical waves in the solar neighbourhood using Gaia DR2. We use the number counts and mean vertical velocity to examine perturbations to the vertical structure of the disk in the solar neighbourhood. We also consider an example where considering the asymmetry affects our measurement of the disk properties. In particular, we obtain a value of zsun = 19.90 +/- 0.4 pc. 

## AUTHOR
Morgan Bennett - bennett at astro dot utoronto dot ca

## Code

### 1. [py/NumberCount.ipynb](py/NumberCount.ipynb)

Calculates the number count asymmetry in the solar neighbourhood. The code starts by fitting a two component model to the number counts for different absolute magnitude-colour bins to recover the sun's vertical position, zsun. We then use this measurement to adjust the number counts and calculate the vertical asymmetry in the number counts about the Galactic mid-plane. Finally, we use an estimate of the number count asymmetry to refit the number counts to a new model which accounts for the perturbation.

### 2. [py/VerticalVelocities.ipynb](py/VerticalVelocities.ipynb)

Examines the mean vertical velocities above and below the galactic mid-plane in APOGEE, GALAH and the Gaia RV sample. We calculate the rolling median for all three samples due to the small number of stars in the first two. For the Gaia RV sample, we also examine the mean vertical velocities binned by vertical height to take advantage of the larger number of points. Also examines the position in absolute magnitude-colour space to compare to the number counts samples.

### 3. [py/CompletenessChecks.ipynb](py/CompletenessChecks.ipynb)

Throughout our analysis, we make certain assumptions about the completeness of the sample and the quality. This notebook examines some of these assumptions in greater detail.

### 4. [py/Summary.ipynb](py/Summary.ipynb)

Uses the asymmetry data saved from [py/NumberCount.ipynb](py/NumberCount.ipynb) and the Gaia RV binned median vertical velocities from  [py/VerticalVelocities.ipynb](py/VerticalVelocities.ipynb) to create a summary plot of the analysis.

## Publications
