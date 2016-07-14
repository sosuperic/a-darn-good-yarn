import cv2

PATH = '/Users/eric/Movies/Forrest Gump (1994)/ForrestGump.mp4'

vidcap = cv2.VideoCapture(PATH)

count = 0
success = True
while success:
  success, image = vidcap.read()
  if count % (24) == 0:
  	image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
  	cv2.imwrite("/Users/eric/Movies/Forrest Gump (1994)/frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1