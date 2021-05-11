import face_recognition
from PIL import Image, ImageDraw
import sys
from glob import glob 

def face_detection_and_facial_keypoints_on_a_folder(input_folder, output_folder):
    all_face_landmarks_list = []
    for ind, img_name in enumerate(glob(input_folder+"*")):
        # if ind == 2:
        #     break
        filename = img_name.split("\\")[-1]
        print("\n\nimg_name : ", img_name)
        image = face_recognition.load_image_file(img_name)

        face_locations = face_recognition.face_locations(image)
        print("face_locations : ", face_locations)

        amount = len(face_locations)
        print(f'There are {amount} face locations')
        try:
            first_face_location = face_locations[0]
            print(first_face_location)

            img = Image.fromarray(image, 'RGB')

            img_with_red_box = img.copy()
            img_with_red_box_draw = ImageDraw.Draw(img_with_red_box)

            img_with_red_box_draw.rectangle(
                [
                    (first_face_location[3], first_face_location[0]),
                    (first_face_location[1], first_face_location[2])
                ],
                outline="red",
                width=10
            )
            img_with_red_box.save(output_folder+filename)

            img_cropped = img.crop((
                first_face_location[3]-20,  # Left x
                first_face_location[0]-100,  # Top y
                first_face_location[1]+20,  # Right x
                first_face_location[2]+30   # Bottom y
            ))
            img_cropped.save(output_folder+"subface_"+filename)

            face_landmarks_list = face_recognition.face_landmarks(image)
            print("face_landmarks_list : ", len(face_landmarks_list[0]))
            all_face_landmarks_list.append((filename, face_landmarks_list[0]))
        
        except Exception as e:
                print("except : ", e, sys.exc_info()[-1].tb_lineno)
                imgfile = Image.open(img_name)
                imgfile.save(output_folder+filename)
                all_face_landmarks_list.append((filename, []))
    return all_face_landmarks_list

# filename = img_name.split("\\")[-1]
# print("\n\nimg_name : ", img_name)

def face_detection_and_facial_keypoints(img_name):
    faceFoundStatus = False
    image = face_recognition.load_image_file(img_name)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        faceFoundStatus = True
        try:
            first_face_location = face_locations[0]
            print(first_face_location)
            img = Image.fromarray(image, 'RGB')
            img_cropped = img.crop((
                first_face_location[3]-20,  # Left x
                first_face_location[0]-100,  # Top y
                first_face_location[1]+20,  # Right x
                first_face_location[2]+30   # Bottom y
            ))
            # img_cropped.save("output_facerecog\\subface_"+filename)
            face_landmarks_list = face_recognition.face_landmarks(image)
            print("face_landmarks_list : ", len(face_landmarks_list[0]))    
            return faceFoundStatus, img_cropped, face_landmarks_list

        except Exception as e:
            print("except : ", e, sys.exc_info()[-1].tb_lineno)
            imgfile = Image.open(img_name)
            # imgfile.save("output_facerecog\\"+filename)
            return faceFoundStatus, imgfile, []


input_folder = "input\\"
output_folder = "output_facerecognition\\"

all_face_landmarks_list = face_detection_and_facial_keypoints_on_a_folder(input_folder, output_folder)
print("\n")
for ind, keypoint in enumerate(all_face_landmarks_list):
    print(f"{ind} : {keypoint}\n")

img_name = "test.jpg"
faceFoundStatus, imgfile, face_landmarks_list = face_detection_and_facial_keypoints(img_name)
print("\nimg_name : ", faceFoundStatus, imgfile, face_landmarks_list)
