import cv2
import csv
import os
import unicodedata
import time
import datetime
import pandas as pd
import numpy as np
from PIL import Image
from threading import Thread
import yagmail
import socket
import getpass

class attendance():

 

    def project_title(self):
        os.system('cls')  # will work for windows only

        print("\t----------------------------------------------")
        print("\t   ######## Smart Attendance System ########")
        print("\t----------------------------------------------")

    ###############################################################
    #To send confirmation email to student after marking attendance
    ###############################################################

    def confirm_email(self, student_id, current_date, time_stamp):
        student_email = ''
        student_id = str(student_id)
        with open("Details"+os.sep+"Details.csv", 'r') as csv_file:
            sdetails = csv.reader(csv_file)
            for student in sdetails:
                if student_id == student[0]:
                    student_email = student[3]
                elif student[0] == "admin":
                    admin_email = student[3]
                    admin_pass = student[4]

            if student_email:
                yag = yagmail.SMTP(user=admin_email, password=admin_pass, host='smtp.gmail.com')
                yag.send(
                    to=student_email,
                    subject="Attendance marked successfully for: " + str(student_id),
                    contents=["Attendance was marked on: "+current_date+"at "+time_stamp],  
                )
                print("Email sent successfully")

    ###############################################################
    #          To mark Attendance for a student                   #
    ###############################################################

    def mark_attendence(self):
        face_detector = cv2.face.LBPHFaceRecognizer_create()
        face_detector.read("TrainingImagesLabels"+os.sep+"Trainner.yml")
        cascade = "haarcascade.xml"
        fCascade = cv2.CascadeClassifier(cascade)
        dframe = pd.read_csv("Details"+os.sep+"Details.csv")
        ft = cv2.FONT_HERSHEY_SIMPLEX
        att_frame = pd.DataFrame(columns=['SId', 'StdName', 'AttDate', 'AttTime'])

        # Video capturing
        footage = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        footage.set(3, 640) 
        footage.set(4, 480)  
        # Defining window size 
        min_width = 0.1 * footage.get(3)
        min_height = 0.1 * footage.get(4)

        while True:
            tmp, freader = footage.read()
            bgrgrey = cv2.cvtColor(freader, cv2.COLOR_BGR2GRAY)
            images = fCascade.detectMultiScale(bgrgrey, 1.2, 5,minSize = (int(min_width), int(min_height)),flags = cv2.CASCADE_SCALE_IMAGE)
            for(A, B, wd, ht) in images:
                cv2.rectangle(freader, (A, B), (A+wd, B+ht), (10, 159, 255), 2)
                ID, config = face_detector.predict(bgrgrey[B:B+ht, A:A+wd])

                if config < 100:
                    ab = dframe.loc[dframe['Id'] == str(ID), 'Name'].values
                    configstr = "  {0}%".format(round(100 - config))
                    temp = str(ID)+"-"+ab
                else:
                    ID = '  Unknown  '
                    temp = str(ID)
                    configstr = "  {0}%".format(round(100 - config))

                if (100-config) > 60:
                    get_time = time.time()
                    current_date = datetime.datetime.fromtimestamp(get_time).strftime('%Y-%m-%d')
                    time_stamp = datetime.datetime.fromtimestamp(get_time).strftime('%H:%M:%S')
                    ab = str(ab)[2:-2]
                    att_frame.loc[len(att_frame)] = [ID, ab, current_date, time_stamp]

                temp= str(temp)[2:-2]
                if(100-config) > 60:
                    temp = temp + " [Pass]"
                    cv2.putText(freader, str(temp), (A+5,B-5), ft, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(freader, str(temp), (A + 5, B - 5), ft, 1, (255, 255, 255), 2)

                if (100-config) > 60:
                    cv2.putText(freader, str(configstr), (A + 5, B + ht - 5), ft,1, (0, 255, 0),1 )
                elif (100-config) > 50:
                    cv2.putText(freader, str(configstr), (A + 5, B + ht - 5), ft, 1, (0, 255, 255), 1)
                else:
                    cv2.putText(freader, str(configstr), (A + 5, B + ht - 5), ft, 1, (0, 0, 255), 1)

            att_frame = att_frame.drop_duplicates(subset=['SId'], keep='first')
            cv2.imshow('Attendance', freader)
            if cv2.waitKey(1) & 0xFF == 27:    
                break
            
        get_time = time.time()
        current_date = datetime.datetime.fromtimestamp(get_time).strftime('%Y-%m-%d')
        time_stamp = datetime.datetime.fromtimestamp(get_time).strftime('%H:%M:%S')
        Hour, Minute, Second = time_stamp.split(":")

        if os.path.exists('AttendanceReports'):
            pass
        else:
            os.mkdir('AttendanceReports')

        if os.path.exists('AttendanceReports'+os.sep+'Attendance_Report.csv'):
            pass
        else:
            header = ['SId', 'StdName', 'AttDate', 'AttTime']
            with open('AttendanceReports'+os.sep+'Attendance_Report.csv', 'w+', newline = '') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
                csv_file.close()

        if (100-config) > 60:
            with open('AttendanceReports'+os.sep+'Attendance_Report.csv', 'a+', newline = '') as csv_file:
                att_frame.to_csv(csv_file, index=False, header=False)
                csv_file.close()
                print("Attendance marked successfully")
                self.confirm_email(ID, current_date, time_stamp)
        else:
            print("Face ID doesn't match! Please try again or contact your administrator") 
       
        footage.release()
        cv2.destroyAllWindows()
        return ID, current_date, time_stamp

    ###############################################################
    #       To verify the credentials of a student or admin       #
    ###############################################################

    def IdCheck(self):
        status = False
        os.system('cls')
        self.project_title()
        Id = input("Enter Your ID: ")
        Pass = getpass.getpass("Enter Your Password: ")

        with open("Details"+os.sep+"Details.csv", 'r') as csv_file:
            sdetails = csv.reader(csv_file)
            for std in sdetails:
                if Id == std[0] and Pass == std[2]:
                    status = True
                    if Id == "admin":
                        self.Menu()
                    else:
                        ID, current_date, time_stamp = self.mark_attendence()
                        input("Press ENTER key to display main menu ")
                        self.IdCheck()
        if not status:
            input("You have entered incorrect ID/Password. Please try again or contact your administrator! \n")
        
    ###############################################################
    #       To check whether the system has a camera              #
    ###############################################################

    def check_camera(self):
        # Loading haar cascade
        face_cascade = cv2.CascadeClassifier('haarcascade.xml')

        # Capturing video from web camera
        try:
            captured_frame = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            print("please click on the webcam window and press ESC button to exit")
        except:
            print("Camera not found!")
            exit()           

        while True:
            # Reading the captured frame
            _, image = captured_frame.read()

            # Convert to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detecting face
            captured_face = face_cascade.detectMultiScale(grayscale_image, 1.3, 5, minSize=(31, 31),flags = cv2.CASCADE_SCALE_IMAGE)

            # Drawing bounding boxes around the face
            for (A, B, wd, ht) in captured_face:
                cv2.rectangle(image, (A, B), (A + wd, B + ht), (10,159,255), 2)

            # Displaying captured image
            cv2.imshow('Webcam Check', image)
            #print("please click on the webcam window and press ESC button to exit")

            # To close the camera window press esc key
            if cv2.waitKey(1) & 0xFF == 27:    
                break

        # Release the VideoCapture object
        captured_frame.release()
        cv2.destroyAllWindows()

    ###############################################################
    #               To capture images from a web cam              #
    ###############################################################

    def capture_images(self):
        Id = input("Enter Student ID: ")
        Name = input("Enter Your Full Name: ")
        Pass = getpass.getpass("Set Your Password: ")
        Email = input("Enter Student Email ID: ")
        if Id and Name and Pass and Email:
            with open("Details"+os.sep+"Details.csv", 'r') as dfile:
                sdetails = csv.reader(dfile)
                for std in sdetails:
                    if Id == std[0]:
                        continue 
        else:
            print("All fields are mandatory. Please try again!")
            input("Press ENTER key to try again! ")
            self.Menu()

        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cascade = "haarcascade.xml"
        classifier = cv2.CascadeClassifier(cascade)
        num_count = 0

        while(True):
            temp, image = capture.read()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            person_face = classifier.detectMultiScale(gray_image, 1.3, 5, minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in person_face:
                cv2.rectangle(image, (x, y), (x+w, y+h), (10, 159, 255), 2)
                #incrementing the image number
                num_count = num_count+1
                #saving the captured images into the folder TrainingImage
                if os.path.exists('TrainingImages'):
                    pass
                else:
                    os.mkdir('TrainingImages')
                cv2.imwrite("TrainingImages" + os.sep + Name + "."+Id + '.' + str(num_count) + ".jpg", gray_image[y:y+h, x:x+w])
                #displaying the frame
                cv2.imshow('frame', image)
                
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # breaking the loop if the image number is more than 10
            elif num_count > 20:
                break
        capture.release()
        cv2.destroyAllWindows()
        
        row = [Id, Name, Pass, Email]
        with open("Details"+os.sep+"Details.csv", 'a+', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(row)
        csv_file.close()
        print("Images are saved for the ID : " + Id + " Name : " + Name)

    ###############################################################
    #       To get the labels of the captured images              #
    ###############################################################

    def getImageswithLabels(self, path):
        paths = [os.path.join(path, f) for f in os.listdir(path)]
        images_list = []
        Ids_list = []
        for ipath in paths:
            p_image = Image.open(ipath).convert('L')
            n_image = np.array(p_image, 'uint8')
            Id = int(os.path.split(ipath)[-1].split(".")[1])
            images_list.append(n_image)
            Ids_list.append(Id)
        return images_list, Ids_list

    ###############################################################
    #    Count generator to display how many images are trained   #
    ###############################################################

    def counter_image(self, path):
        counter = 1
        dirs = [os.path.join(path, f) for f in os.listdir(path)]
        for dir in dirs:
            print(str(counter) + " images are trained", end="\r")
            time.sleep(0.01)
            counter += 1
        print(str(counter) + " images are trained", end="\r")
        time.sleep(1)

    ###############################################################
    #       To train images using HaarCascade algorithm           #
    ###############################################################

    def train_images(self):
        recognize = cv2.face.LBPHFaceRecognizer_create()
        cascade = "haarcascade.xml"
        detect_image = cv2.CascadeClassifier(cascade)
        
        if os.path.exists('TrainingImagesLabels'):
            pass
        else:
            os.mkdir('TrainingImagesLabels')
            
        images_list, Ids_list = self.getImageswithLabels("TrainingImages")
        Thread(target = recognize.train(images_list, np.array(Ids_list))).start()
        Thread(target = self.counter_image("TrainingImages")).start()
        recognize.save("TrainingImagesLabels"+os.sep+"Trainner.yml")

    ###############################################################
    #      To send email about the statistics of attendance       #
    ###############################################################

    def send_email(self):
        student_id = input("Enter Student ID: ")

        if student_id:
            with open("Details"+os.sep+"Details.csv", 'r') as sfile:
                sdetails = csv.reader(sfile)
                for std in sdetails:
                    if student_id == std[0]:
                        student_email = std[3]
                    elif std[0] == "admin":
                        admin_email = std[3]
                        admin_pass = std[4]
        else:
            input("Student ID not registered. Please contact admin!")
            self.Menu()

        if student_email:
            file_name = 'AttendanceReports'+os.sep+'Attendance_Report.csv'
            date = datetime.date.today().strftime("%B %d, %Y")

            dframe = pd.read_csv(file_name)
            students_per = dframe.groupby('SId')['AttDate'].nunique()
            classes_per = len(dframe['AttDate'].unique())
            student_per = (students_per[int(student_id)]/classes_per)*100

            student_file = 'AttendanceReports'+os.sep+'Attendance_Report_'+student_id+'.csv'
            student_dframe = dframe.loc[dframe['SId'] == int(student_id)]
            student_dframe.to_csv(student_file, columns = ['AttDate', 'AttTime'], mode='w+', index=False)

            yag = yagmail.SMTP(user=admin_email, password=admin_pass, host='smtp.gmail.com')

            yag.send(
                to=student_email,
                subject="Auto generated attendance for student ID: " + str(student_id),
                contents=["Your attendance percentage is" +str(student_per)+ "%. Please find the attachment"], 
                attachments= student_file 
            )
            print("Email sent successfully")

    ###############################################################
    #       creating main menu for the admin                      #
    ###############################################################

    def Menu(self):
        self.project_title()
        
        print('\n')
        print(10 * "*", "MAIN MENU", 10 * "*")
        print("[1] Check my Camera")
        print("[2] Capture my Face")
        print("[3] Train my Images")
        print("[4] Recognize me & mark my attendance")
        print("[5] Send Email")
        print("[6] Quit")

        while True:
            try:
                choice = int(input("Enter Choice: "))
                if choice == 1:
                    self.check_camera()
                    print("Camera Found!")
                    input("Press ENTER key to display main menu ")
                    self.Menu()
                elif choice == 2:
                    self.capture_images()
                    input("Press ENTER key to display main menu ")
                    self.Menu()
                elif choice == 3:
                    self.train_images()
                    input("Press ENTER key to display main menu ")
                    self.Menu()
                elif choice == 4:
                    ID, current_date, time_stamp = self.mark_attendence()
                    input("Press ENTER key to display main menu ")
                    self.Menu()
                elif choice == 5:
                    self.send_email()
                    input("Press ENTER key to display main menu ")
                    self.Menu()
                elif choice == 6:
                    print("Closing the application!")
                    time.sleep(1)
                    exit()
                else:
                    print("Not a valid Choice. Try Again. Please enter numbers between 1 to 6")
                    input("Press ENTER key to display main menu ")
                    self.Menu()
            except ValueError:
                print("Not a valid Choice. Try Again. Please enter numbers between 1 to 6")
                input("Press ENTER key to display main menu ")
                self.Menu()
        exit()

    ###############################################################

def main():
    obj = attendance()
    obj.IdCheck()

if __name__ == "__main__":
    main()
