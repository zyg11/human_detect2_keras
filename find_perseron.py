import cv2
from keras.models import load_model
from human_detect_2.face_train2 import Model
import numpy as np
from human_detect_2.face_dataset2 import resize_image
from keras import backend as K
IMAGE_SIZE=64


def face_predict(image):
    # 依然是根据后端系统确定维度顺序
    # data = np.empty((1, 3, 64, 64), dtype="float32")
    # image = resize_image(image)
    # image=image.reshape(1,-1)
    # arr= np.asarray(image,dtype="float32")
    # data[0, :, :, :] = [arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]
    # image = arr.reshape((1,64, 64, 3))
    # image = image.reshape((1, 64, 64, 3))
    if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
        image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
    elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
        image = resize_image(image)
        image = image.reshape((1, 64, 64, 3))

    # 浮点并归一化
    image = image.astype('float32')
    image /= 255.0
    # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
    result = model.predict_proba(image)
    print('result: ', result)
    # 给出类别预测
    result = model.predict_classes(image)
    # 返回类别预测结果
    return result[0]
if __name__=='__main__':
    #加载模型
    model=load_model('E:/keras_data/face6_model/faces_model_6_1.h5')
    # model = Model()
    # model.load_model(file_path='E:/keras_data/face5/face_model5.h5')
    #框柱人脸的矩形边框颜色
    color=(0,255,0)
    #捕获实时视频流
    cap=cv2.VideoCapture(0)
    #人脸识别分类器路径
    cascade_path="F:/haarcascades/haarcascade_frontalface_default.xml"
    #循环检测人脸
    while True:
        _,frame=cap.read()#读取一帧视频
        #图像灰化，降低计算复杂度
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #使用人脸分类器
        cascade=cv2.CascadeClassifier(cascade_path)
        #利用分类器识别哪个区域为人脸
        faceRects=cascade.detectMultiScale(frame_gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        if len(faceRects)>0:
            for faceRect in faceRects:
                x,y,w,h,=faceRect
                #截取脸部图像识别
                image=frame[y-10:y+h+10,x-10:x+w+10]

                faceID=face_predict(image)
                #如果是我
                if faceID==0:
                    cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,
                                  thickness=2)
                    #文字提示
                    cv2.putText(frame,
                                'zhuyangang',
                                (x+30,y+30), #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,#字体
                                1,#字号
                                (255,0,255),#颜色
                                2)#线宽
                elif faceID==1:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color,
                                  thickness=2)
                    # 文字提示
                    cv2.putText(frame,
                                'Girl',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 线宽
                elif faceID == 2:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color,
                                  thickness=2)
                    # 文字提示
                    cv2.putText(frame,
                                'lilong',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 线宽
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    # 文字提示是谁
                    cv2.putText(frame,
                                "others",
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 线宽
        cv2.imshow("find",frame)
        # 等待10毫秒看是否有按键输入
        k=cv2.waitKey(10)
        # 如果输入q则退出循环
        if k&0xff==ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

