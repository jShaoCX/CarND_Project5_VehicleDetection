import numpy as np
import cv2 as cv2
import glob
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.svm import LinearSVC
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
import time
from scipy.ndimage.measurements import label
from pipeline_functions import extract_features
from pipeline_functions import find_cars
from pipeline_functions import add_heat
from pipeline_functions import apply_threshold
from pipeline_functions import draw_labeled_bboxes
from pipeline_functions import draw_labeled_bboxes_single
from collections import deque

cars = []
notcars = []
os.chdir("./imagescombined")
images = glob.glob('*.png')
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

#had to combine the GTI images into a single folder and append to the car list
gti_cars = []
os.chdir("../GTI_combine")
images = glob.glob('*.png')
for image in images:
    gti_cars.append(image)

# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    im = cv2.imread(car_list[0])
    data_dict["image_shape"] = im.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = type(im[0, 0, 0])
    # Return data_dict
    return data_dict

#test method for generating writeup images
def process_image_single(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    _, med_on_windows = find_cars(image, y_start_stop[0], y_start_stop[1], 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    _, large_on_windows = find_cars(image, y_start_stop[0], y_start_stop[1]-128, 1.25, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    # Add heat to each box in box list
    on_windows = large_on_windows+med_on_windows
    heat = add_heat(heat, on_windows)

    plt.imshow(heat, cmap='gray')
    plt.show()

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    plt.imshow(labels[0], cmap='gray')
    plt.show()

    out_img = draw_labeled_bboxes_single(np.copy(image), labels)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return out_img

os.chdir("../imagescombined")
data_info = data_look(cars, notcars)

print('Your function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

'''
# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.show()
'''

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()

t=time.time()

os.chdir("../imagescombined")

car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

os.chdir("../GTI_combine")

gti_car_features = extract_features(gti_cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

t2 = time.time()

car_features = car_features+gti_car_features

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC(C=10)

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

os.chdir("../")
image = cv2.imread('./test_images/test1.jpg')

'''
#Sliding windows with overlap (not used)
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
draw_image = np.copy(image)
heat = np.zeros_like(image[:,:,0]).astype(np.float)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Add heat to each box in box list
heat = add_heat(heat, hot_windows)

# Apply threshold to help remove false positives
heat = apply_threshold(heat, 1)

# Visualize the heatmap when displaying
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)

window_img = draw_labeled_bboxes(np.copy(image), labels)# draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
window_img = cv2.cvtColor(window_img, cv2.COLOR_HSV2BGR)
#plt.imshow(window_img)
'''

'''
#running test images
out_img = process_image_single(image)
out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
plt.imshow(out_img)
plt.show()

image = cv2.imread('./test_images/test2.jpg')
out_img = process_image_single(image)
out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
plt.imshow(out_img)
plt.show()
image = cv2.imread('./test_images/test3.jpg')
out_img = process_image_single(image)
out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
plt.imshow(out_img)
plt.show()
image = cv2.imread('./test_images/test4.jpg')
out_img = process_image_single(image)
out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
plt.imshow(out_img)
plt.show()
image = cv2.imread('./test_images/test5.jpg')
out_img = process_image_single(image)
out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
plt.imshow(out_img)
plt.show()
image = cv2.imread('./test_images/test6.jpg')
out_img = process_image_single(image)
out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
plt.imshow(out_img)
plt.show()
'''
'''
#displaying hog image
image = cv2.imread('./vehicles/GTI_Right/image0364.png')
features, hog_image = hog(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          transform_sqrt=True,
                          visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
'''


#declare a deque for frame averaging
heat_map_list = deque()

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    _, med_on_windows = find_cars(image, y_start_stop[0], y_start_stop[1], 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    _, large_on_windows = find_cars(image, y_start_stop[0], y_start_stop[1]-128, 1.25, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    # Add heat to each box in box list
    on_windows = large_on_windows+med_on_windows
    heat = add_heat(heat, on_windows)

    heat_map_list.append(heat)

    if len(heat_map_list) > 24:
        heat_map_list.popleft()

    cumulative_heat = np.sum(heat_map_list, axis=0)

    # Apply threshold to help remove false positives
    heat = apply_threshold(cumulative_heat, 71)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    out_img = draw_labeled_bboxes(np.copy(image), labels)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return out_img

adv_output = 'object_detection_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
#clip1 = VideoFileClip("project_video.mp4").subclip(18,24) #19,23
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(adv_output, audio=False)
