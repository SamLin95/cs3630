from myro import *

initialize('/dev/tty.Fluke2-07E6-Fluke2')

while getObstacle(1) < 1000:
	forward(1, 0.5)

turnBy(90, 'deg')

move = True

while move:
	forward(1, 1)
	turnBy(-90, 'deg')
	if getObstacle(1) == 0:
		stop()
		move = False
	turnBy(90, 'deg') 