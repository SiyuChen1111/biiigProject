# Idea
Recommend by agent, we can try a simple ANN for EEG recgnition. The ANN, which can recgnize central pariental positivity(CPP), is also a meaningful tool. Using this tool, we can input any EEG data and judge whether it in the data has a CPP value.  

If we can get CPP in any task without evidence accumulation, it means that the features of CPP which we defined are not enough.

## sub-idea
- Can we get CPP with raw eeg data? (❌it may not work)
    - I can try to plot raw data and see its pattern first.
    - Dose it can genelize to other tasks?
    - How to comfirm our model is useful? 
        - _train and test_
        - But how to define the answer of test dataset?
        - Because EEG has high noise, we should bin the data first.
            - We should epoch raw data first
- We have no idea about exact judgement of CPP, so it is still meaningful to recgnize it using preprocssed eeg data.(✅)
    - We can have a binary answer of a dataset, which is 1 if the data has a CPP value, and 0 if it does not.

## Process
1. Preprocess the data.
    Manning et al. (2021)
2. Build the standard pattern of CPP.
3. annotate the data.
4. Train the model.
5. Test the model.
6. Use another dataset to test the model.


```# Dataset
- EEG dataset, whose tasks are with the evidence accumulation process(EMP), was confirmed to have CPP.
    -
- EEG dataset, whose tasks are without the evidence accumulation process(EMP), was confirmed to have no CPP.
    -
- EEG dataset, whose tasks are with the evidence accumulation process(EMP), but did not find CPP before.
    -

```# Pattern of CPP
- Step up before response.
- Step down after response.
- The windows of CPP is about -150ms to -50ms.
