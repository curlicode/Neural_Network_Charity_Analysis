# Neural_Network_Charity_Analysis

## Overview of the analysis

*Explain the purpose of this analysis.*

Because the Alphabet Soup foundation receives numerous funding recipients (34,000), the objective is to employ machine learning and neural network techniques (via TensorFlow pipelines) to define a model that accurately predicts organization success based on historic funding patterns and outcomes.  Additional objectives include: demonstrating preprocessing skills, such as managing categorical data and cleansing/normalizing data sets; tuning neural network model optimization; and evaluating machine learning model performance.

## Results

* *Data Preprocessing*

  * *What variable(s) are considered the target(s) for your model?*
    * __Deliverable(s) (1), (2) and (3).__ In preprocessing the data for it to be ingested by the machine learning model, the `IS_SUCCESSFUL` field is the single target boolean value, indicating whether a funding application was successfully funded (`1 = True`, `0 = False`). The single classification data field suggests the use of a single output neuron and the choice of the `sigmoid` activation function, ideal for binary classification.

  * *What variable(s) are considered to be the features for your model?*
    * __Deliverable(s) (1) and (2).__ The following fields were included as the default features for the model:
      * `APPLICATION_TYPE`
        * `APPLICATION_TYPE` was binned
      * `AFFILIATION`
      * `CLASSIFICATION`
      * `USE_CASE`
      * `ORGANIZATION`
      * `STATUS`
      * `INCOME_AMT`
      * `SPECIAL_CONSIDERATIONS`
      * `ASK_AMT`
    * __Deliverable(s) (3).__ The following fields were included as the default features for the model:
      * `APPLICATION_TYPE`
        * `APPLICATION_TYPE` was binned
      * `AFFILIATION`
      * `CLASSIFICATION`
        * `CLASSIFICATION` was binned
      * `USE_CASE`
      * `ORGANIZATION`
      * `STATUS`
      * `INCOME_AMT`
        * `INCOME_AMT` values were grouped with a custom function
      * `SPECIAL_CONSIDERATIONS`
      * `ASK_AMT`
        * `ASK_AMT` values were grouped with a custom function

  * *What variable(s) are neither targets nor features, and should be removed from the input data?*
    * __Deliverable(s) (1) and (2).__ `EIN` and `NAME` are omitted as they are textual labels without any bearing on the calculation and determination of weights for the model.  Categorical features were also removed once the same fields had been encoded, scaled, and standardized.
    * __Deliverable(s) (3).__ `EIN` and `NAME` are omitted as they are textual labels without any bearing on the calculation and determination of weights for the model.  Categorical features were also removed once the same fields had been encoded, scaled, and standardized.  Upon reviewing the plot density of unique values for the data fields, the frequency parameter to bin the data was adjusted as needed to reduce the effect of value variability or skewed distributions on the model, which may have undue influence on parameter weight estimation.  Additional columns were dropped during evaluation of the model, but it seems no additional features needed to be removed given the model performance and grouping and binning performed.

* *Compiling, Training, and Evaluating the Model*
  * *How many neurons, layers, and activation functions did you select for your neural network model, and why?*
    * __Deliverable(s) (1) and (2).__ There are a total of four (4) layers including the input and output layer in addition to two (2) hidden layers.  An input layer of 50 inputs was chosen to match the features within the data set.  The first hidden layer has 80 neurons, and a second hidden layer has 30 neurons.  The output layer only consists of a single neuron with a sigmoid activation function for the purpose of binary classification.  This model was selected to match the demonstrated example output.  
    * __Deliverable(s) (3).__ There are a total of four (4) layers including the input and output layer in addition to two (2) hidden layers.  An input layer of 34 inputs was based off of the number of features within the data set after one-hot encoding the features.  The first hidden layer has 80 neurons, and a second hidden layer has 30 neurons--based on initial default selection.  The output layer only consists of a single neuron with a sigmoid activation function for the purpose of binary classification.  This model was selected to match the demonstrated example output.  

  * *Were you able to achieve the target model performance?*
    * __Deliverable(s) (1) and (2).__ The base model achieves {loss: 0.5432, accuracy: 0.7335} for the training set.  The base model achieves a {loss: 0.5482, accuracy: 0.7348} after 10 epochs.

    * __Deliverable(s) (3).__ After running the model for 10 epochs, the model achieves {loss: 0.001170 accuracy: 0.9999} after 10 epochs without modifying or requiring additional epochs, exceeding the target model performance of {accuracy: 0.75}.

  * *What steps did you take to try and increase model performance?*
    * __Deliverable(s) (1) and (2).__ None (base case).

    * __Deliverable(s) (3).__ Upon reviewing the plot density of unique values for the data fields, the frequency parameter to bin the data was adjusted as needed to reduce the effect of value variability or skewed distributions on the model, which may have undue influence on parameter weight estimation.  Additional hidden layers and increasing the number of neurons was attempted, but the biggest performance gain was achieved by looking at monetary and funding variables and regrouping "secure" organizational funding for easier classification.  Moreover, additional hidden layers potentially yields a better fit but risks overfitting data.  Additional methods attempted included dropping features systematically and changing the activation function, but given the initial accuracy achieved, no further steps or modifications were made.

## Summary

*Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.*

Neural networks and namely deep-learning neural networks offer tremendous power for solving classification problems.  Because of the single classification problem of the `IS_SUCCESSFUL` feature, the model uses a single output neuron with `sigmoid` activation function, suitable for binary classification.  The inputs match the number of features with a sufficient number of neurons that fit the rule-of-thumb of having three times as many neurons in the hidden layer as the number of inputs.  Although modifications to the activation function, number of epochs, amount of neurons and hidden layers was tested, no additional model changes were made given binning and re-grouping efforts made to the raw data.  Seeing a likely contextual determinant for funding being based on ask and income amounts, these fields had their values simplified to relevant values—in essence trimming outlier and extreme values, which might otherwise have unwanted influence in the calculation of weight parameter within the model.  This appears to have made enough difference to achieve a near-1.0 accuracy and minimal loss value.

When considering other machine learning models, logistic regression is not considered because although similar to basic neural network with a sigmoid activation function, it risks issues with vanishing gradients during the model training process, whereas our the neural network implemented here utilized the ReLu activation function, which helps to avoid some of these same issues.  The simplicity of logistic regression for this case risks under-fitting the data.

SVMs are a suitable candidate as a binary classifier.  While SVMs are a low-code and robust solution, its performance is predicated on linearly separable data while being less less suitable for nonlinear data.  Without the use of multiple SVM models, it classifies against simpler relationship patterns.  

Random forest classifiers as a type of ensemble learning, decision trees, seem a likely candidate for such a classification problem because it parallels the decision making process while being robust with outliers and nonlinear data.  Ensemble learning models are well-suited for tabular data while performing quickly with less coding and comparable accuracy given an adequate tree depth.

Given these comparisons, random forest classifiers would be a preferred alternative to the neural network deep learning approach for its ability to make similar binary classification while managing variability in data.  Random forest classifiers are performant and match the contextual problem.


# Criteria

* There is a title, and there are multiple sections (2 pt)
* Each section has a heading and subheading (2 pt)
* Links to images are working, and code is formatted and displayed correctly (2 pt).

* Overview of the loan prediction risk analysis:
  * The purpose of this analysis is well defined (4 pt)
* Results:
  * There is a bulleted list that answers all six questions (15 pt)
* Summary:
  * There is a summary of the results (2 pt)
  * There is a recommendation on using a different model to solve the classification problem, and justification (3 pt)
