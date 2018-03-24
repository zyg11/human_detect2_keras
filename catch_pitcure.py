import cv2
from PIL import Image
def CatchPICFromVideo(window_name,camera_idx,catch_pic_num,path_name):
    cv2.namedWindow(window_name)
    #视频来源，可以来自一段视频，也可以直接来自USB摄像头
    cap=cv2.VideoCapture(camera_idx)

    #使用人脸识别分类器
    classifier=cv2.CascadeClassifier('F:/haarcascades/haarcascade_frontalface_default.xml')
    #识别出人脸后要画的边框颜色，RGB格式
    color=(0,255,0)

    num=0
    while cap.isOpened():
        ok,frame=cap.read()#读取一帧数据
        if not ok:
            break
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#将当前帧图像转换成灰度图像
        #人脸检测， 1.2分别为图片缩放比例和需要检测的有效点数
        faceRects=classifier.detectMultiScale(grey,scaleFactor=1.2,minNeighbors=3, minSize=(32,32))
        if len(faceRects)>0:        #大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x,y,w,h=faceRect
                #将当前帧保存为图片
                img_name='%s/%d.jpg'%(path_name,num)
                image=frame[y-10:y+h+10,x-10:x+w+10]
                cv2.imwrite(img_name,image)

                num+=1
                if num>(catch_pic_num):  #如果超过指定最大保存数量退出循环
                    break
                #画出矩形框
                cv2.rectangle(frame,(x-10,y-10), (x+w+10,y+h+10), color, 2)
                #显示当前捕捉了多少人脸图片，
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d'%(num),(x+30,y+30),font,1,(255,0,255),4)
                #超过制定最大数量结束程序
        if num>(catch_pic_num):
            break
        #显示图像
        cv2.imshow(window_name,frame)
        c=cv2.waitKey(5)
        if c&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__=='__main__':
    CatchPICFromVideo("get face",0,982,'E:/keras_data/face4/xueliqiang')
