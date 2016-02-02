# Copyright (C) 1993-2002, by Peter I. Corke

# $Log: rtfkdemo.m,v $
# Revision 1.3  2002/04/02 12:26:48  pic
# Handle figures better, control echo at end of each script.
# Fix bug in calling ctraj.
#
# Revision 1.2  2002/04/01 11:47:17  pic
# General cleanup of code: help comments, see also, copyright, remnant dh/dyn
# references, clarification of functions.
#
# $Revision: 1407 $
figure(2)
    echo on
# Forward kinematics is the problem of solving the Cartesian position and 
# orientation of a mechanism given knowledge of the kinematic structure and 
# the joint coordinates.
#
# Consider the Puma 560 example again, and the joint coordinates of zero,
# which are defined by qz
    qz
#
# The forward kinematics may be computed using fkine() with an appropropriate 
# kinematic description, in this case, the matrix p560 which defines 
# kinematics for the 6-axis Puma 560.
    fkine(p560, qz)
#
# returns the homogeneous transform corresponding to the last link of the 
# manipulator
pause % any key to continue
#
# fkine() can also be used with a time sequence of joint coordinates, or 
# trajectory, which is generated by jtraj()
#
    t = [0:.056:2]; 	% generate a time vector
    q = jtraj(qz, qr, t); % compute the joint coordinate trajectory
#
# then the homogeneous transform for each set of joint coordinates is given by
    T = fkine(p560, q);

#
# where T is a 3-dimensional matrix, the first two dimensions are a 4x4 
# homogeneous transformation and the third dimension is time.
#
# For example, the first point is
    T(:,:,1)
#
# and the tenth point is
    T(:,:,10)
pause % any key to continue
#
# Elements (1:3,4) correspond to the X, Y and Z coordinates respectively, and 
# may be plotted against time
    subplot(3,1,1)
    plot(t, squeeze(T(1,4,:)))
    xlabel('Time (s)');
    ylabel('X (m)')
    subplot(3,1,2)
    plot(t, squeeze(T(2,4,:)))
    xlabel('Time (s)');
    ylabel('Y (m)')
    subplot(3,1,3)
    plot(t, squeeze(T(3,4,:)))
    xlabel('Time (s)');
    ylabel('Z (m)')
pause % any key to continue
#
# or we could plot X against Z to get some idea of the Cartesian path followed
# by the manipulator.
#
    subplot(1,1,1)
    plot(squeeze(T(1,4,:)), squeeze(T(3,4,:)));
    xlabel('X (m)')
    ylabel('Z (m)')
    grid
pause % any key to continue
echo off
