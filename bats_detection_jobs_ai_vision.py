
import glob
import cv2
import os
import oci
import base64
import json
import ocifs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from oci.ai_vision import AIServiceVisionClient
from oci.ai_vision.models.analyze_image_details import AnalyzeImageDetails
from oci.ai_vision.models.image_classification_feature import ImageClassificationFeature
from oci.ai_vision.models.image_object_detection_feature import ImageObjectDetectionFeature
from oci.ai_vision.models.inline_image_details import InlineImageDetails
from oci.ai_vision.models.object_storage_image_details import ObjectStorageImageDetails


#####################################################################################################################
##################################################################################################################### Store config or auth for AI Vision
#####################################################################################################################

## to do
#config = oci.config.from_file('~/.oci/config', 'DEFAULT')  #in notebook
config = oci.config.from_file('./config', 'DEFAULT')  #in job

#####################################################################################################################
##################################################################################################################### Env variables
#####################################################################################################################

#video = "./0003_TRUE_BAT.mp4"

VIDEO_NAME = os.environ.get("VIDEO_NAME", "0120_TRUE_BAT.mp4")  #in Job. Shoud be the name of the video in the bucket to be analyzed.
#VIDEO_NAME = "./0120_TRUE_BAT.mp4"

BUCKET = os.environ.get("BUCKET", "West_BP")
NAMESPACE = os.environ.get("NAMESPACE", "frqap2zhtzbe")

print("Video used is " +str(VIDEO_NAME))
print("Bucket used is " +str(BUCKET))
print("Namespace used is " +str(NAMESPACE))

#define local folders
processed_images = "./processed_images"
splitted_images = "./splitted_images"
output_image_ai_vision = "./output_image_ai_vision"


#####################################################################################################################
##################################################################################################################### Get input video from bucket
#####################################################################################################################
print("Start getting the video from the bucket")

path_input_video= "./input_video" 

try:       
    if not os.path.exists(path_input_video):         
        os.makedirs(path_input_video)    

except OSError: 
    print ('Error: Creating directory of input video')

print("Copy video from bucket to Job block storage")
#copy the mentioned video from the bucket to local job storage
fs = ocifs.OCIFileSystem()
fs.get(f"oci://{BUCKET}@{NAMESPACE}/bats_detection/input_video/{VIDEO_NAME}", "./input_video/" , recursive=True, refresh=True)                                       #### SAMUEL

#####################################################################################################################
##################################################################################################################### Clip video in frames
#####################################################################################################################
print("Start clipping video in frames")

# Read the video from specified path 
cam = cv2.VideoCapture(f"./input_video/{VIDEO_NAME}")

fps = cam.get(cv2.CAP_PROP_FPS)
print("Number of frame per second = " +str(fps))

path_split_images = "./splitted_images" 

try:       
    # creating a folder named data 
    if not os.path.exists(path_split_images):         
        os.makedirs(path_split_images)    

except OSError: 
    print ('Error: Creating directory of data for split images')

currentframe = 0
  
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret:
        
        if currentframe < 10:   
            name = path_split_images + '/frame000000' + str(currentframe) + '.jpg'     
            
        elif currentframe >= 10 and currentframe < 100:   
            name = path_split_images + '/frame00000' + str(currentframe) + '.jpg'    
            
        elif currentframe >= 100 and currentframe < 1000:   
            name = path_split_images + '/frame0000' + str(currentframe) + '.jpg'

        elif currentframe >= 1000 and currentframe < 10000:   
            name = path_split_images + '/frame000' + str(currentframe) + '.jpg'

        elif currentframe >= 10000 and currentframe < 100000:   
            name = path_split_images + '/frame00' + str(currentframe) + '.jpg'

        elif currentframe >= 100000 and currentframe < 1000000:   
            name = path_split_images + '/frame0' + str(currentframe) + '.jpg'
            
        else:
            # if video is still left continue creating images 
            name = path_split_images + '/frame' + str(currentframe) + '.jpg'      
            
        print ('Creating...' + name) 
  
        # writing the extracted images 
        cv2.imwrite(name, frame) 
  
        currentframe += 1
    else: 
        break
    
cam.release() 

#####################################################################################################################
##################################################################################################################### Pre-process images
#####################################################################################################################
print("Start pre-processing fames")

processed_images = "./processed_images" 

try:       
    # creating a folder named data 
    if not os.path.exists(processed_images):         
        os.makedirs(processed_images)    

except OSError: 
    print ('Error: Creating directory of data for processed_images')

#start at second image because of difference. Count number of images, then loop through
dir_path = './splitted_images'
number_of_images = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))
print(number_of_images)

#list of images in numbers. Starts at image 000
number_of_images = list(range(1, number_of_images-1)) 

for image_number in number_of_images:
    
    print(f'image number = {image_number}')
    
    if image_number < 10:   ################################################ workaround for now. To make sure that split_images and processed_images have the same name
        zeros = "000000"

    elif image_number >= 10 and image_number < 100:   
        zeros = '00000'   

    elif image_number >= 100 and image_number < 1000:   
        zeros = '0000'

    elif image_number >= 1000 and image_number < 10000:   
        zeros = '000'

    elif image_number >= 10000 and image_number < 100000:   
        zeros = '00'

    elif image_number >= 100000 and image_number < 1000000:   
        zeros = '0'

    else:
        zeros = '' 
    
    ################# image 2
    if image_number < 9:   ################################################ workaround for now. To make sure that split_images and processed_images have the same name
        zerosx = "000000"

    elif image_number >= 9 and image_number < 99:   
        zerosx = '00000'   

    elif image_number >= 99 and image_number < 999:   
        zerosx = '0000'

    elif image_number >= 999 and image_number < 9999:   
        zerosx = '000'

    elif image_number >= 9999 and image_number < 99999:   
        zerosx = '00'

    elif image_number >= 99999 and image_number < 999999:   
        zerosx = '0'

    else:
        zerosx = ''
    

 
    test_image_1 = f"./splitted_images/frame{zeros+str(image_number)}.jpg"
    test_image_2 = f"./splitted_images/frame{zerosx+str(image_number+1)}.jpg"

    print(f'Image 1 = {test_image_1}')
    print(f'Image 2 = {test_image_2}')
    
    img1 = cv2.imread(test_image_1)
    img2 = cv2.imread(test_image_2)

    # 255 - = white/blac
    img3 = cv2.absdiff(img1,img2)

     #load image and convert to grayscale + blur slightly
    gray_1 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)  #convert to gray scale

    #increase pixel sizes
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(gray_1,kernel,iterations = 1)

    video_name = ""

    if image_number < 10:   
        name = processed_images + '/frame000000' + str(image_number) + '.jpg'     

    elif image_number >= 10 and image_number < 100:   
        name = processed_images + '/frame00000' + str(image_number) + '.jpg'    

    elif image_number >= 100 and image_number < 1000:   
        name = processed_images + '/frame0000' + str(image_number) + '.jpg'

    elif image_number >= 1000 and image_number < 10000:   
        name = processed_images + '/frame000' + str(image_number) + '.jpg'

    elif image_number >= 10000 and image_number < 100000:   
        name = processed_images + '/frame00' + str(image_number) + '.jpg'

    elif image_number >= 100000 and image_number < 1000000:   
        name = processed_images + '/frame0' + str(image_number) + '.jpg'

    else:
        # if video is still left continue creating images 
        name = processed_images + '/frame' + str(image_number) + '.jpg' 

    cv2.imwrite(name,dilation)

    
    print("-"*50)
        
    
    
    
#####################################################################################################################
##################################################################################################################### Apply AI Vision model
#####################################################################################################################

# Max Result to return
MAX_RESULTS = 5

# Vision Service endpoint
endpoint = "https://vision.aiservice.eu-frankfurt-1.oci.oraclecloud.com"

# Initialize client service_endpoint is optional if it's specified in config
ai_service_vision_client = AIServiceVisionClient(config=config, service_endpoint=endpoint)

#####################################################################################################################
##################################################################################################################### Function 1
#####################################################################################################################

def detect_logos(ai_service_vision_client, image):

    # Encode a sample image
    encoded_string = None
    
    # Open Image as Base64 encoded String
    with open(image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    
    
    # Set Max Results to Return
    MAX_RESULTS = 10

    # Set up request body with one or multiple Features (Type of Service)
    image_object_detection_feature = ImageObjectDetectionFeature()
    image_object_detection_feature.max_results = MAX_RESULTS
    image_object_detection_feature.model_id = "ocid1.aivisionmodel.oc1.eu-frankfurt-1.amaaaaaangencdya2is47mfop3vdhoeyqdnzhk3pmjvw4bno4jebkg34cbnq"  
 
    # List of Features
    features = [image_object_detection_feature]

    # Create Analyze Image Object and set Image and Features
    analyze_image_details = AnalyzeImageDetails()
    inline_image_details = InlineImageDetails()
    inline_image_details.data = encoded_string.decode('utf-8')
    analyze_image_details.image = inline_image_details
    analyze_image_details.features = features

    # Send analyze image request
    res = ai_service_vision_client.analyze_image(analyze_image_details=analyze_image_details)
    
    # Return Result
    return res


#####################################################################################################################
##################################################################################################################### Function 2
#####################################################################################################################

def parse_results(results, filename):
    
    
     # Define Directory
    result_directory = output_image_ai_vision
    split_images_directory = splitted_images  #processed images are input images, but we have to apply bounding boxes to original img


    try:       
        if not os.path.exists(result_directory):         
            os.makedirs(result_directory)    

    except OSError: 
        print ('Error: Creating directory of data for result images')
    
    # Print result
    print("**Analyze Image Result**")

    # Parse Response as JSON
    od_results = json.loads(str(results.data))

    # Extract Bounding Boxes
    od_bounding_boxes = od_results['image_objects']

    # Create Empty DataFrame
    results_list = []

    # Read in Image
    im = cv2.imread(os.path.join(split_images_directory, filename))

    # Get Dimensions of Image
    height, width, channels = im.shape

    # Extract Objects Boxes from Results
    obj = json.loads(str(results.data))['image_objects']
    
    # If there is nothing detected - just save the image in results directory
    if obj == None:
        # Write Image
        cv2.imwrite(os.path.join(result_directory, filename),im)
        
        print('Nothing Detected!\n')
        
    else:
        
        try:
            
            # Iterate over each Bounding Box
            for box in od_bounding_boxes:
                
                # Only Draw and Save bounding box if confidence is greater than 60%
                if box['confidence'] >= 0.6:

                    # Extract opposite coordinates for bounding box
                    # Un-Normalise the Data by scaling to the max image height and width
                    # Convert to Integer
                    coordinates_pt1_x = int(box['bounding_polygon']['normalized_vertices'][0]['x'] * width)
                    coordinates_pt1_y = int(box['bounding_polygon']['normalized_vertices'][0]['y'] * height)
                    coordinates_pt2_x = int(box['bounding_polygon']['normalized_vertices'][2]['x'] * width)
                    coordinates_pt2_y = int(box['bounding_polygon']['normalized_vertices'][2]['y'] * height)

                    # Build Points as Tuples
                    coordinates_pt1 = (coordinates_pt1_x, coordinates_pt1_y)
                    coordinates_pt2 = (coordinates_pt2_x, coordinates_pt2_y)

                    # Draw Bounding Boxes - Pass in Image, Top Left and Bottom Right Points, Colour, Line Thickness
                    cv2.rectangle(im, coordinates_pt1, coordinates_pt2, (0, 255, 0), 2)
                    # Plot Label just above the Top Left Point, Set Font, Size, Colour, Thickness
                    cv2.putText(im, box['name'], (coordinates_pt1_x, coordinates_pt1_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                    # Write Image with Bounding Boxes to file
                    cv2.imwrite(os.path.join(result_directory, filename),im)

                    # Extract Frame Name, Label and Confidence and append to results list
                    results_list.append([filename, box['name'], box['confidence']])
                    
                else:
                    # Write Image with No Bounding Boxes to file
                    cv2.imwrite(os.path.join(result_directory, filename),im)


        except Exception as e:
            print('Error Encountered')
            print('Error Message:', e.message)
        
 
        print('Object Detected!\n')
    
    
    return results_list

#####################################################################################################################
##################################################################################################################### Invoke function 1 and 2
#####################################################################################################################

!rm -r ./output_images_ai_vision

# Define Directory
directory = processed_images

# Define Empty List to store all objects detected
final_results = []

# Iterate over Files in Directory of Split Images
for filename in os.listdir(directory):
    
    # Check to make sure it is an Image
    if filename.endswith(".jpg"):
        
        # Define Image
        image =  os.path.join(directory, filename)

        # Detect 
        results = detect_logos(ai_service_vision_client, image)

        # Parse Results
        objects_detected = parse_results(results, filename)
        
        # Append List of Objects Detected in each imageto final results list
        final_results.append(objects_detected)
                    
    else:
        continue

# Remove Nulls from Results List
final_results = [x for x in final_results if x != []]

# Create a New Empty List to Store 2D List (Reformatted)
clean_list = []

# Convert 3D List to 2D List
for e1 in final_results:
    for e2 in e1:
        clean_list.append(e2)
        
print("Convert final dataframe")
# Convert Final Clean List to Data Frame
final_df = pd.DataFrame(clean_list, columns = ['FRAME', 'OBJECT', 'CONFIDENCE'])

# Sort DataFrame
final_df = final_df.sort_values(by=['FRAME'])

print("Save csv in bucket")
# Save DataFrame to CSV
final_df.to_csv(f'oci://{BUCKET}@{NAMESPACE}/bats_detection/input_video/objects-detected-results.csv', index=False)                                         ########## change bucket name
final_df
