# These are advanced CNN Models which Classifies the Age of a face , The Gender of a Face and The Ethnicity of a face,

# Dataset Contains 22k images

 Age Classifier is saved as : 'age_class_model.sav'
 Gender Classifier is saved as : 'gender_class_model.sav'
 Ethnicity Classifier is saved as : 'ethnicity_class_model.sav'
 
 You Can Use Them to predict or train a new set of data
 
 While Prediciting:
  Note :
      Input Image Size should be reshaped to [48,48,1] 
      
      output may vary according to the model selected:
        1. for age_classifier : output is an age value(in float datatype)
        2. for gender_classifier : output greater than 0.5 should be considered as 1 and if less than 0.5 as 0
        3. for ethnicity : np.argmax(predicted_value) gives the class of ethnicity the input image belongs to
                            These classes of Ethnic group are [0,1,2,3,4]
                        
                        
                        
## Dataset Extracted From Kaggle:
train_data split into test and train data for evaluating purposes
Sequential API NN method Followed
## Modelled Using CNN
