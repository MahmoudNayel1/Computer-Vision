import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import svm

# Reading Folders
Training_Dir = "Training"
Testing_Dir = "Testing"
Validation_Dir = "Validation"
Training_path = os.listdir(Training_Dir)
Testing_path = os.listdir(Testing_Dir)
Validation_path = os.listdir(Validation_Dir)

# Creating Sift object
sift = cv2.xfeatures2d.SIFT_create()

# Reading images
Train_data = [] # Training [ Images , labels ]
Test_data = []  # Testing [ Images , labels ]
Label = 0  # Labels : ( Buildings = 0 , Food = 1 , Landscape = 2 , People = 3 )

for Folder_Name in Training_path:
    Path = os.path.join(Training_Dir, Folder_Name) # Join path (Training \ Folder Name )
    Dir = os.listdir(Path)
    for images in Dir:
        path = os.path.join(Path, images) # Join path ( Training \ Folder Name \ Image Name )
        image = cv2.imread(path,0) # Reading image colored ( 1 ) / Gray scale ( 0 )
        image = np.array(image)  #List -> Numpy array
        kp, dest = sift.detectAndCompute(image, None)   # Extracting Keypoints and Descriptors from image

        # Assigning labels
        if Folder_Name == "Building_Training":
            Label = 0
        elif Folder_Name == "Food_Training":
            Label = 1
        elif Folder_Name == "Landscape_Training":
            Label = 2
        elif Folder_Name == "People_Training":
            Label = 3
        Train_data.append([image, Label , dest])  # Training Data [0 --> images , 1 --> labels , 2 --> Descriptors]


np.random.shuffle(Train_data) # shuffling Data

#Feature Exrtaction
Vstack = np.array(Train_data[0][2])
for Descriptor in Train_data[1:] :
    Vstack = np.vstack((Vstack , Descriptor[2])) # Filling stack of descriptors

# K mean clustring
N_clusters = 12
Kmean_object = KMeans(n_clusters = N_clusters)
k_mean = Kmean_object.fit_predict(Vstack)

# Dictionary(Histogram) of all images initialized by 0
mega_histogram = np.array([np.zeros(N_clusters) for i in range(len(Train_data))])

# filling the Dictionary(Histogram) of all images
counter = 0
for i in range(len(Train_data)) : #images loop(600)
    j = len(Train_data[i][2])
    for k in range(j): #Descriptor length (ex:128)
        index = k_mean[counter + k]
        mega_histogram[i][index]+=1
    counter += j

#Standardization
scale = StandardScaler().fit(mega_histogram)
mega_histogram = scale.transform(mega_histogram)

image_data = []
label_data = []

for i in Train_data:
    label_data.append(i[1])


# SVM Model Training
Classifier1 = svm.SVC(kernel ='rbf', gamma = 0.9 , C = 0.6).fit(mega_histogram , label_data)
Classifier2 = svm.SVC(kernel ="poly" ,degree = 3 , C = 1).fit(mega_histogram , label_data)

# Training Accuracy
True_Values1 = 0
True_Values2 = 0

# loop itirate on images
for images in Train_data :
    image = images[0]
    Train_Keypoint , Train_descriptors = sift.detectAndCompute(image, None)
    Vocab = np.array([[0 for i in range(N_clusters)]])
    Vocab = np.array(Vocab, "float32")
    Train_Kmean = Kmean_object.predict(Train_descriptors)

    # Filling the vocab of the image
    for i in Train_Kmean:
        Vocab[0][i]+=1
    Vocab = scale.transform(Vocab)
    Training_predictions1 = Classifier1.predict(Vocab) # Getting the prediction of classifire 1
    Training_predictions2 = Classifier2.predict(Vocab) # Getting the prediction of classifire 2

    if images[1] == Training_predictions1 :
        True_Values1 += 1

    if images[1] == Training_predictions2 :
        True_Values2 += 1


Training_Accuracy1 = (True_Values1 / len(Train_data)) * 100
Training_Accuracy2 = (True_Values2 / len(Train_data)) * 100
print("Training Accuracy Classifier 1 = ",str(Training_Accuracy1))
print("Training Accuracy Classifier 2 = ",str(Training_Accuracy2))





# Validation
Validation_Label = 0
Validation_Counter = 0   # Validation images count
Validation_Truevalues1 = 0
Validation_Truevalues2 = 0

# first loop iterate on folder names
for folder in Validation_path :
    path = os.path.join(Validation_Dir, folder)
    images = os.listdir(path)

    # second loop iterate on image names
    for image in images:
        image_path = os.path.join(path,image)
        Image = cv2.imread(image_path, 0)  # Reading image colored ( 1 ) / Gray scale ( 0 )
        Image = np.array(Image)     # List -> Numpy array
        Validation_kp , Validation_dest = sift.detectAndCompute(Image, None)  # Extracting keypoints and Descreptors from image

        # Assigning labels
        if folder == "Building":
            Validation_Label = 0
        elif folder == "Food":
            Validation_Label = 1
        elif folder == "Landscape":
            Validation_Label = 2
        elif folder == "People":
            Validation_Label = 3

        # Creating the Vocab of the image
        vocab = np.array([[0 for i in range(N_clusters)]])
        vocab = np.array(vocab, "float32")
        Validation_Kmean = Kmean_object.predict(Validation_dest) # Getting Kmean predictions on the descriptors

        # Filling the vocab of the image
        for i in Validation_Kmean:
            vocab[0][i] += 1
        vocab = scale.transform(vocab)  # Standard scaler transform
        Validation_predictions1 = Classifier1.predict(vocab)
        Validation_predictions2 = Classifier2.predict(vocab)
        Validation_Counter += 1

        if Validation_predictions1 == Validation_Label:
            Validation_Truevalues1 += 1

        if Validation_predictions2 == Validation_Label:
            Validation_Truevalues2 += 1


Validation_Accuracy1 = (Validation_Truevalues1 / Validation_Counter)*100
print("Validation_Accuracy_Classifier1 = " + str(Validation_Accuracy1))

Validation_Accuracy2 = (Validation_Truevalues2 / Validation_Counter)*100
print("Validation_Accuracy_Classifier2 = " + str(Validation_Accuracy2))



#Test
Testing_Label = 0
Test_Counter = 0       # Test image count
Test_Truevalues1 = 0
Test_Truevalues2 = 0

# first loop iterate on folder names
for Folder in Testing_path :
    path = os.path.join(Testing_Dir, Folder)
    images = os.listdir(path)

    # second loop iterate on image names
    for image in images:
        image_path = os.path.join(path,image)
        Image = cv2.imread(image_path, 0)  # Reading image colored ( 1 ) / Gray scale ( 0 )
        Image = np.array(Image)  # List -> Numpy array
        Testing_kp , Testing_dest = sift.detectAndCompute(Image, None)

        #assigning labels
        if Folder == "Building_Testing":
            Testing_Label = 0
        elif Folder == "Food_Testing":
            Testing_Label = 1
        elif Folder == "Landscape_Testing":
            Testing_Label = 2
        elif Folder == "People_Testing":
            Testing_Label = 3

        # Creating image dictionary
        vocab = np.array([[0 for i in range(N_clusters)]])
        vocab = np.array(vocab, "float32")
        Validation_Kmean = Kmean_object.predict(Testing_dest)
        for i in Validation_Kmean:
            vocab[0][i] += 1
        vocab = scale.transform(vocab)
        Testing_predictions1 = Classifier1.predict(vocab) # Get the prediction of image label from classifier 1
        Testing_predictions2 = Classifier2.predict(vocab) # Get the prediction of image label from classifier 2
        Test_Counter += 1

        # comparing real values with predicted values
        if Testing_predictions1 == Testing_Label:
            Test_Truevalues1 += 1
        if Testing_predictions2 == Testing_Label:
            Test_Truevalues2 += 1


Testing_Accuracy1 = (Test_Truevalues1 / Test_Counter)*100
print("Testing_Accuracy_Classifier1 = " + str(Testing_Accuracy1))

Testing_Accuracy2 = (Test_Truevalues2 / Test_Counter)*100
print("Testing_Accuracy_Classifier2 = " + str(Testing_Accuracy2))



















