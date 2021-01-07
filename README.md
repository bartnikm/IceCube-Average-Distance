# IceCube-Average-Distance
This repository holds all files used to calculate the average distance neutrinos take within the IceCube Neutrino Observatory

File Descriptions:

IceCube R 1120m.ipynb - Displays Dimensions of IceCube Observatory and path of neutrino within observatory from a user-inputted azimuthal and zenith angle. This notebook was not used in the calculations but is here so users can interact with different angles

RanAng.py - Calculates distances from 6500000 random azimuthal and zenith angle combinations and saves them, along with the points of entry and exit, to a csv entitled "1120_m_rand_angs.csv". This csv has been omitted this repository due to its size (~ 1 GB)

RanNormPlane.py - Creates a normal plane based on path of neutrino from an azimuthal and zenith angle and shifts the path's intersection point on plane by a random amount within the observatory's dimensions and calculates distance to shifted entry and exit point. This process is repeated 10000 times for every integer azimuthal and zenith angle between 0 and 90 degrees, and outputs a csv entitled "Datapoints.csv" that contains the azimuthal angles, zenith angles, and the average distance per angle. 

The_Graph.ipynb - Reads in the Datapoints.csv file and creates a heat map of distances per angle, saving the resulting image as "10000P_Phi_V_Theta.png" It also highlights the angles that resulted in distances within some arbitrary error from the overall average distance

CSV_Reading.ipynb - Reads in the 1120_m_rand_angs.csv file and explores its properties. It is here that the final average distance between all paths in every azimuthal and zenith angle is given in the fourth cell in the "mean" row and "Length" column.

Other Notes:
1. All distances are reported in km unless otherwise specified
2. Zenith Angles of 0 degrees were omitted from calculations due to it creating divide-by-zero errors and due to the fact that a zenith angle of zero degrees would make every measured distance equal to the radius of the observatory.
3. The Diameter of the IceCube Neutrino Observatory is calculated here to be ~1120 meters. 
