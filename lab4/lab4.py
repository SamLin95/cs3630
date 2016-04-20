from __future__ import division
import sys
sys.path.append('/usr/local/Cellar/opencv/2.4.11_1/lib/python2.7/site-packages/')
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.stats import norm


def detectBlobs(im):
    """ Takes and image and locates the potential location(s) of the red marker
        on top of the robot

    Hint: bgr is the standard color space for images in OpenCV, but other color
          spaces may yield better results

    Note: you are allowed to use OpenCV function calls here

    Returns:
      keypoints: the keypoints returned by the SimpleBlobDetector, each of these
                 keypoints has pt and size values that should be used as
                 measurements for the particle filter
    """

    #YOUR CODE HERE

    lowerBound = np.array([0, 50, 0])
    upperBound = np.array([0, 255, 0])
    #extract the red mark as green portion
    ret, thresh1 = cv2.threshold(im, 144, 255, cv2.THRESH_BINARY)
    #cv2.imshow('thresholded 1', thresh1)

    params = cv2.SimpleBlobDetector_Params()
    
    params.filterByColor = 1
    params.blobColor = 0

    params.filterByArea = 1
    params.minArea = 500
    params.maxArea = 10000000000

    params.filterByCircularity = 1
    params.minCircularity = 0.0001
    params.maxCircularity = 1

    params.filterByConvexity = 1
    params.minConvexity = 0.0001
    params.maxConvexity = 1

    params.filterByInertia = 1
    params.minInertiaRatio = 0.0001
    params.maxInertiaRatio = 1


    params.minThreshold = 0

    params.maxThreshold = 255

    mask = cv2.inRange(thresh1, lowerBound, upperBound)

    #cv2.imshow('mask', mask)
    ret, thresh2 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV) 


    detector = cv2.SimpleBlobDetector(params)
    cv2.imshow('t2', thresh2)
    keypoints = detector.detect(thresh2)
    return keypoints

def predict(particles, predictSigma):
    """ Predict particles one step forward. The motion model should be additive
        Gaussian noise with sigma predictSigma
      
    Returns:
      particles: list of predicted particles (same size as input particles)
    """
    
    #YOUR CODE HERE
    new_particles = particles.copy()

    noise_x = np.random.normal(loc=0, scale=predictSigma, size=len(new_particles))
    noise_y = np.random.normal(loc=0, scale=predictSigma, size=len(new_particles))
    #np.random.shuffle(noise)
    new_particles[:,0] += noise_x
    new_particles[:, 1] += noise_y
    return new_particles

def update(particles, weights, keypoints):
    """ Resample particles and update weights accordingly after particle filter
        update
      
    Returns:
      newParticles: list of resampled partcles of type np.array
      weights: weights updated after sampling
    """
    new_weights = weights.copy()
    for i in range(len(particles)):
      current_particle  = particles[i]
      new_weights[i] = get_weight(current_particle, keypoints)
    new_weights = normalize(new_weights)
    return particles, new_weights


def get_weight(particle, keypoints):
  ini_weight = 1.0
  for keypoint in keypoints:
    #print "keypoint" , keypoint.pt
    dist = distance.euclidean(particle, keypoint.pt)
    ini_weight *= norm.pdf(dist/60)
  return ini_weight

def normalize(weights):
  #print 'sum of weights is', sum(weights)
  #print 'weights:\n', weights
  s = sum(weights)
  return np.array(map(lambda x: x/s, weights))


def resample(particles, weights):
    """ Resample particles and update weights accordingly after particle filter
        update
      
    Returns:
      newParticles: list of resampled partcles of type np.array
      wegiths: weights updated after sampling
    """
    
    #YOUR CODE HERE
    random_new_num = 0

    idx = range(len(particles))
    weights = normalize(weights)
    print "particles num", len(particles)
    new_idxs = np.random.choice(idx, size=min(len(particles), 100),replace=True, p=weights)
    
    add_xs = np.random.uniform(0, high=1280, size=random_new_num)
    add_ys = np.random.uniform(0, high=800, size=random_new_num)

    resample_num = len(new_idxs)

    new_samples = np.zeros((resample_num + random_new_num, 2))
    new_weights = np.zeros(resample_num + random_new_num)

    for i, idex in enumerate(new_idxs):
      new_samples[i, :] = particles[idex, :]
      new_weights[i] = weights[idex]

    for i in xrange(random_new_num):
        new_samples[i + resample_num, 0] = add_xs[i]
        new_samples[i + resample_num, 1] = add_ys[i]
        new_weights[i + resample_num] = 0.00001
    new_weights = normalize(new_weights)
    return new_samples, new_weights

def visualizeParticles(im, particles, weights, color=(0,0,255)):
    """ Plot particles as circles with radius proportional to weight, which
        should be [0-1], (default color is red). Also plots weighted average
        of particles as blue circle. Particles should be a numpy.ndarray of
        [x, y] particle locations.

    Returns:
      im: image with particles overlaid as red circles
    """
    im_with_particles = im.copy()    
    s = (0, 0)
    for i in range(0, len(particles)):
      s += particles[i]*weights[i]
      cv2.circle(im_with_particles, tuple(particles[i].astype(int)), radius=int(weights[i]*250), color=(0,0,255), thickness=3)
    cv2.circle(im_with_particles, tuple(s.astype(int)), radius=3, color=(255,0,0), thickness=6)    
    return im_with_particles

def visualizeKeypoints(im, keypoints, color=(0,255,0)):
    """ Draw keypoints generated by blob detector on image in color specified
        (default is green)

    Returns:
      im_with_keypoints: the image with keypoints overlaid
    """    
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints

if __name__ == "__main__":
  """ Iterate through a dataset of sequential images and use a blob detector and
      particle filter to track the robot(s) visible in the images. A couple
      helper functions were included to visualize blob keypoints and particles.

  """

  #some initial variables you can use
  imageSet='ImageSet2'
  imageWidth = 1280
  imageHeight = 800
  numParticles = 1000
  initialScale = 50
  predictionSigma = 150
  x0 = np.array([600, 300])  #seed location for particles
  particles = [] #YOUR CODE HERE: make some normally distributed particles
  xs = np.random.normal(loc=x0[0], scale=initialScale, size=numParticles)
  ys = np.random.normal(loc=x0[1], scale=initialScale, size=numParticles)
  particles = np.array(zip(xs, ys))
  weights = [1/numParticles for i in xrange(numParticles)] #YOUR CODE HERE: make some weights to go along with the particles

  # if imageSet = 'ImageSet'

  for i in range(0, 43):
    #read in next image
    im = cv2.imread(imageSet+'/'+imageSet+'_' + '%02d.jpg'%i)
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
 
    #visualize particles
    im_to_show = visualizeParticles(im, particles, weights)
    cv2.imshow("Current Particles", im_to_show)
    cv2.imwrite('processed/'+imageSet+'_' + '%02d_'%i+'1_Current.jpg', im_to_show)

    #predict forward
    particles = predict(particles, predictionSigma)
    im_to_show = visualizeParticles(im, particles, weights)
    cv2.imshow("Prediction", im_to_show)
    cv2.imwrite('processed/'+imageSet+'_' + '%02d_'%i+'2_Predicted.jpg', im_to_show)
    
    #detected keypoint in measurement
    keypoints = detectBlobs(yuv)
  

    #update paticleFilter using measurement if there was one
    if keypoints:
      particles, weights = update(particles, weights, keypoints)

    im_to_show = visualizeKeypoints(im, keypoints)
    im_to_show = visualizeParticles(im_to_show, particles, weights)
    cv2.imshow("Reweighted", im_to_show)
    cv2.imwrite('processed/'+imageSet+'_' + '%02d_'%i+'3_Reweighted.jpg', im_to_show)

    #resample particles
    particles, weights = resample(particles, weights)
    im_to_show = visualizeKeypoints(im, keypoints)
    im_to_show = visualizeParticles(im_to_show, particles, weights)
    cv2.imshow("Resampled", im_to_show)
    cv2.imwrite('processed/'+imageSet+'_' + '%02d_'%i+'4_Resampled.jpg', im_to_show)
    cv2.waitKey(0)
    
