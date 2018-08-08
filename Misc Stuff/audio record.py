# import librosa as rs
# import soundcard as nd

import pyaudio
import wave

import os.path

save_path = "C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX"

name_of_file = input("What is the name of the file: ")

WAVE_OUTPUT_FILENAME = os.path.join(save_path, name_of_file + ".wav")

# WAVE_OUTPUT_FILENAME = "output.wav"

# import math
# test = math.inf

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
# RECORD_SECONDS = test


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
				channels=CHANNELS,
				rate=RATE,
				input=True,
				frames_per_buffer=CHUNK)

# key = ord(getch())
# if key == 27: #ESC
# 	break

import keyboard #Using module keyboard

frames = []

from msvcrt import getch


# print("Press ESC to Record")


# while True:#function

# 	# print("Press ENTER to Record")
# 	# try: #used try so that if user pressed other than the given key error will not be shown
	
# 	key = ord(getch())
# 	if key == 27: #ESC

# 		while key == 27:

# 			print("* recording")
# 			# frames = []

# 			# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
# 			data = stream.read(CHUNK)
# 			frames.append(data)

# 			print("Press ESC to Stop Recording")

# 			# if keyboard.is_pressed("q"):

# 		if keyboard.is_pressed("q"):

# 			print("* done recording")

# 			stream.stop_stream()
# 			stream.close()
# 			p.terminate()

# 			key += 1

# 			break

import time


print("Press ESC to Record")

while True:

	key = ord(getch())
	if key == 27: #ESC
		print("* recording")
		print("Press 'q' to Stop Recording")
		# frames = []

		# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		stream.start_stream()
		while stream.is_active():
			# time.sleep(2)
			data = stream.read(CHUNK)
			frames.append(data)

		# pass

			if keyboard.is_pressed("q"):
				print("* done recording")

				stream.stop_stream()
				stream.close()
				p.terminate()
				break
	break


		# print("Press ESC to Stop Recording")

	# if keyboard.is_pressed("enter"):#if key 'enter' is pressed 
	# 	print("* recording")
	# 	# frames = []

	# 	# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	# 	data = stream.read(CHUNK)
	# 	frames.append(data)

	# 	print("Press ESC to Stop Recording")

	# if keyboard.is_pressed("q"):
	# 	print("* done recording")

	# 	stream.stop_stream()
	# 	stream.close()
	# 	p.terminate()
	# 	break

# print("* recording")

# frames = []

# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)

# print("* done recording")

# stream.stop_stream()
# stream.close()
# p.terminate()

# record_stuff()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# -------------------------------------------------------------------

# import os.path

# save_path = "C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS"

# name_of_file = input("What is the name of the file: ")

# completeName = os.path.join(save_path, name_of_file + ".txt")   

# file1 = open(completeName, "wb")

# # file1.write(str(final_list))

# # for line in final_list:
# #     file1.write(line)

# file1.close()

print("hello world")