from mosaic import *
from matplotlib import pyplot as plt
#from myro import *


CAM_KX = 720.
CAM_KY = CAM_KX
CAM_CX = 320.
CAM_CY = 200.



def get_y_angles(folder):
    print "start processing for folder %s:"%(folder)
    x = [i*5 for i in xrange(1, 9)]
    y = []
    for i in xrange(1, 10):
        img_a = cv2.imread('%s/%d_degree_1.png'%(folder, i*5))
        img_b = cv2.imread('%s/%d_degree_2.png'%(folder, i*5))

        img_a = img_a[::2,::2,:]
        img_b = img_b[::2,::2,:]

        length_ab, width_ab = 1280, 720
        img_ab, best_H = img_combine_homog(img_a, img_b, length_ab, width_ab)

        K = cam_params_to_mat(CAM_KX, CAM_KY, CAM_CX, CAM_CY)
        R = rot_from_homog(best_H, K)
        assert(np.shape(R) == (3,3) and isinstance(R, np.matrix))
        y_ang = extract_y_angle(R)
        y.append(y_ang)
        cv2.imshow('Combined Image', img_ab)

    y = map(lambda x: x*57.2958, y)
    print 'calculated ys are: ', y
    return y

def get_pictures(folder_name):
    initialize('/dev/tty.Fluke2-07E6-Fluke2')
    x = [i*10 for i in xrange(1, 6)]
    for i in xrange(1, 10):
        img1 = takePicture('gray')
        turnBy(i*5, 'deg')
        img2 = takePicture('gray')
        savePicture(img1, '%s/%d_degree_1.png'%(folder_name, i*5))
        savePicture(img2, '%s/%d_degree_2.png'%(folder_name, i*5))

def get_plot():
    folders  = ['photo%d'%(i) for i in xrange(2, 3)]
    x = [i*5 for i in xrange(1, 10)]
    for folder in folders:
        y = get_y_angles(folder)
        plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    get_pictures('photo1')




