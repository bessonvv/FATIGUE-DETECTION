
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import pyglet
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# воспроизведение звукового сигнала
	music = pyglet.resource.media('alarm.wav')
        music.play()
        pyglet.app.run()

def eye_aspect_ratio(eye):
	# вычисление евклидового расстояния (вертикаль)
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# вычисление евклидового расстояния (горизонталь)
	C = dist.euclidean(eye[0], eye[3])

	# вычислите соотношение сторон глаз
	ear = (A + B) / (2.0 * C)

	return ear
 
ap = argparse.ArgumentParser()

ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# определение двух констант: одной для соотношения сторон глаз, указывающей на моргание, 
# а второй  для количества последовательных кадров, в течение 
# которых глаз должен находиться ниже порогового значения, 
# чтобы сработал сигнал тревоги
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# инициализиуем счетчик кадров, 
# а также логическое значение, используемое для указания, 
# срабатывает ли сигнал тревоги
COUNTER = 0
ALARM_ON = False

# HOG и dlib
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("68 face landmarks.dat")

# установка указателей лицевых ориентиров для левого и правого глаза соответственно
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Запуск видео
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# перебор кадров из видео
while True:
	# извелечение кадра из потокового видеофайла и 
	# преобразование его в оттенки серого
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# распознавание лица в рамке оттенков серого
	rects = detector(gray, 0)

	# повторение цикла распознавания лица
	for rect in rects:
		# определение ориентиров лица для области лица,
		#  затем преобразование координат ориентира лица (x, y) в массив NumPy
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# извлечение координат левого и правого глаза,
		# затем использование этих координат для 
		# вычисления соотношения сторон глаз для обоих глаз
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Среднее соотношение сторон глаз для обоих глаз
		ear = (leftEAR + rightEAR) / 2.0

		# вычисление выпуклой оболочки для левого и правого глаза,
		# затем визуализация каждой из них
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# проверка, не находится ли соотношение сторон глаз 
		# ниже порогового значения для моргания, и если это так, 
		# увеличить счетчик кадров моргания
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# если глаза были закрыты в течение достаточного времени, 
			# раздается звуковой сигнал
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# если будильник не включен, включить его
				if not ALARM_ON:
					ALARM_ON = True

					# проверка, был ли предоставлен файл аварийного сигнала,
					# и если да, запустить поток, чтобы звуковой сигнал 
					# воспроизводился в фоновом режиме
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# вывод надписи сигнала тревоги на экране
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# в противном случае соотношение сторон глаз 
		# не будет ниже порогового значения мигания, 
		# поэтому сбросится счетчик и прозвучит сигнал тревоги.
		else:
			COUNTER = 0
			ALARM_ON = False

		# отображение вычисленного соотношения сторон глаз на кадре,
		# чтобы облегчить отладку и установку правильных
		# пороговых значений соотношения сторон глаз и счетчиков кадров
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# Отображение рамки
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# если была нажата клавиша "q", выход из цикла
	if key == ord("q"):
		break

# Закрытие программы
cv2.destroyAllWindows()
vs.stop()
