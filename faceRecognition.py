import face_recognition
import xlwt
import cv2

#output position data about face
def Output_facedata(face_landmarks_list,index,worksheet):
    col=0
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for feature_num in range(0,len(face_landmarks[facial_feature])):
                if index==0:
                    worksheet.write(0,col+feature_num, label =str(facial_feature)+'_'+str(feature_num))
                else:
                    worksheet.write(index,col+feature_num, label =str(face_landmarks[facial_feature][feature_num]))
            col+=feature_num+1
    
#face_recognition,record positions    
def Capture_position(vedio_name):
    video_full_path="./vedio/patient/"+vedio_name
    workbook = xlwt.Workbook(encoding = 'ascii')
    worksheet = workbook.add_sheet('face_data')
    cap  = cv2.VideoCapture(video_full_path)
    frame_count = 1
    success, frame = cap.read()
    while(success):
        face_landmarks_list = face_recognition.face_landmarks(frame)
        Output_facedata(face_landmarks_list,frame_count-1,worksheet)
        success, frame = cap.read()
        print ('Read a new frame: ', success,frame_count)
        frame_count = frame_count + 1
    cap.release()
    workbook.save('./patientData/'+vedio_name+'.xls')


import os
path="./vedio/patient/"  
path_list=os.listdir(path)
path_list.sort() 
for filename in path_list:
    print(os.path.join(filename))
    vedio_name = os.path.join(filename)
    Capture_position(vedio_name)
