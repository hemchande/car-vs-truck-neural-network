# FlickrAPI Object to collect photos of Cars & Trucks

# Importing libraries
import urllib
import flickrapi

from time import sleep

# To authenticate, plug in your API Key as the first argument and the Secret Code as the second.
flickr = flickrapi.FlickrAPI('a7dddcff8c160df3d09123d77fafbf11', 'f1fc3f548395aa44', cache = True)

# Getting the photos through the walk method
car_photos = flickr.walk(text = 'car',
                     tag_mode = 'all',
                     tags = 'car',
                     extras = 'url_c',
                     per_page = 1000,
                     sort = 'relevance')

truck_photos = flickr.walk(text = 'truck',
                     tag_mode = 'all',
                     tags = 'truck',
                     extras = 'url_c',
                     per_page = 1000,
                     sort = 'relevance')




# Extracting URLs of photos from the response
car_urls = []
for i, photo in enumerate(car_photos):
    car_urls.append(photo.get('url_c'))

    if i > 1050:
        break

truck_urls = []
for i, photo in enumerate(truck_photos):
    truck_urls.append(photo.get('url_c'))

    if i > 1050:
        break


# Removing None type objects & taking only the first 960 URLs. 950 images for training & 10 images for testing
car_urls = [x for x in car_urls if x is not None][:960]
truck_urls = [x for x in truck_urls if x is not None][:960]

# =====================================  Downloading the images to folders ==========================================
# Mention full paths to either training/testing folders
# 1) Cat Images
for count, url in enumerate(car_urls[:480]):
    urllib.request.urlretrieve(url, 'training_folder_path-car'+ str(count) + '.jpg')

sleep(100)

for count, url in enumerate(car_urls[480:]):
    urllib.request.urlretrieve(url, 'training_folder_path-car'+ str(480 + count) + '.jpg')

# Owl Images
for count, url in enumerate(truck_urls[:480]):
      urllib.request.urlretrieve(url, 'training_folder_path-truck'+ str(count) + '.jpg')

sleep(100)

for count, url in enumerate(truck_urls[480: ]):
    urllib.request.urlretrieve(url, 'training_folder_path-truck'+ str(450 + count) + '.jpg')
    
    
# Import the Sequential model and layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

batch_size = 5    
    
    # Training Augmentation configuration
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Testing Augmentation - Only Rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)

# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('train/', target_size = (300, 300), 
                                                    batch_size = batch_size,
                                                    class_mode = 'binary') 

# Generator for validation data
validation_generator = test_datagen.flow_from_directory('test/', 
                                                        target_size = (300, 300),
                                                        batch_size = batch_size,
                                                        class_mode = 'binary')

# Fit the model on Training data
model.fit_generator(train_generator,
                    epochs = 5,
                    validation_data = validation_generator,
                    verbose = 1)

# Evaluating model performance on Testing data
loss, accuracy = model.evaluate(validation_generator)

print("\nModel's Evaluation Metrics: ")
print("---------------------------")
print("Accuracy: {} \nLoss: {}".format(accuracy, loss))

