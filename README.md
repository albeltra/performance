# MEWS Evaluation
This project in split into two major components with code accordingly separated.

**core**  
Functions to process EHR data, compute MEWS values, and calculate performance according to scoring functions of interest.
**analyze**  
Functions that apply the code in **core** to the data, and generate/store results.  

## core  
**constructors**  
This module contains all the essential classes for the evaluation of predictions 
from a predictive model.  
The **process** class is the heart of this module as it contains methods that dictate
how the data is processed, augmented(if at all), and scored. 
Assuming predictions come from a model come in the form an array with 
columns: (ID, Time (h), Model Output) we process the data in the following way:  

```python    
for ID in Unique_IDS:
    cur_data = data[ID]
    augmented_data = augmenter(cur_data)
    for cur_augmented_data in augmented_data: 
        for scorer in scorers:
            for threshold in thresholds:
                output.append(scorer(data, threshold))

Where,
augmenter -> function
scorers -> list of functions

```
The most important take aways from this structure are: 
- The unit of analysis for the scoring functions is all data for a particular ID (in our case a patient encounter)  
- Scoring functions must take the current data and a threshold to apply 
to the model outputs  
- An augmenting function is required but is not required to actually augment
the data; simply pass a function that returns the unaltered data.

This structure was chosen to remove redundant processing of data when calculating multiple scores 
for multiple thresholds. 

The remaining classes in the **constructors** module simply contain scoring and augmenting functions. 
Since the **process** class only requires some scoring functions and an augmenter function, they need not
come from the other classes in this module.

The **run** module takes the raw output of the processor and determines how the scores from all the patient's
data are combined. In this particular case, **run** assumes that we calculate some sort of count for each patient
and that these should be summed up to give a single score e.g. # true positives, # false positives, etc. This behaviour
isn't too restrictive since our scorers can return an output of any shape. For example, if we wanted to calculate average
number of warnings per hour, we could write a scoring function that outputs: # of warnings and time span of 
for a particular patient encounter. By summing these across all patients we get total # of warnings for all patients, 
and total time of all patient's stays. Divide the two and you get a single rate for the whole data set.
  
**mews**  
Contains all the functions for reading EHR data from CSV and calculating MEWS values.
These functions are not written for general cases and are won't work on data that is not
extracted in the same format as ours.  
The core of this module is the **mews_persist** function which implements the imputation and exclusion 
criteria used in our analysis, namely: 
- Scores are carried forward in time for imputation, indefinitely
- If no data is available for a particular vital sign exclude the patient from the analysis (except GCS)  
- Exclude time points where no data is available for imputation for any vital sign (at the beginning of a patient stay)

**metrics**  
Contains high level functions that implement the **mews** and **constructors** modules to calculate 
a few performance metrics of interest in our study. This is a good starting point to understand how to use 
the **process** class.

## analyze
Contains modules to calculate performance metrics for different kinds of data. Generating an ROC curve,
 for example, requires case and control data. Sensitivity at lead time only makes use of case data. The idea is to
 calculate as many scores as possible to reduce redundant run time.
 
 These methods have a __name__ == __main__ statement which calculates all metrics for each combination of 
 data source: **both**(case + control), **case**(only case data), **control**(only control data). There are additional
 modules with suffixes corresponding to the type of augmentation that was performed: **sub**(sample windows from the 
 control data), **shift** (shift the event time).
 
 These are most useful when a user wants to calculate several scoring functions quickly in a non user-friendly way.