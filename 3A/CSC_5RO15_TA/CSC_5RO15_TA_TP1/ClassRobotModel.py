# -*- coding: utf-8 -*-
"""

Tutorials of Robotics

Class computing the robot models

Mathieu GROSSARD

"""





import math as m


class RobotModel:
    
    """ 
    Computation of the numerical geometric, kinematic and dynamic models of a fully actuated serial structure made of rigid links
                         
    """
                
    def __init__( self, **parameters):
        
        # Number of actuated joints
        try:
            nJoints= parameters.get('nJoints')
            assert type(nJoints) == int
            self.numberJoints = nJoints
            print("Attribute (int): self.numberJoints = ", self.numberJoints)
        except AssertionError:
            print("Error, the argument 'nberJoints' should be an integer value")
        # Type of actuated joints
        try:
            jointsType= parameters.get('jointsType')
            assert len(jointsType) == self.numberJoints
            try:
                assert [ jointsType[i]=='R' or jointsType[i]=='P' for i in range(0,len(jointsType)) ] == [1 for i in range(0,len(jointsType))]
                self.jointsType = jointsType
                print("Attribute (list): self.jointsType = ", self.jointsType)
                # creation of a list made of '0' if the joint type is 'Revolute' or '1' if the joint type is 'Prismatic' 
                self.sigma = [0 if self.jointsType[i] == 'R' else 1 for i in range(0,len(jointsType))] 
                print("Attribute (list - 0 if self.jointsType[i] == 'R' / 1 if self.jointsType[i] == 'P'): self.sigma = ", self.sigma)                      
            except AssertionError:
                print("Error, the argument in 'jointsType' should be either 'R' for revolute joints or 'P' for prismatic joint")
        except AssertionError:
            print("Error, the argument 'nberJoints' should be equal to the length of 'jointsType'")
        # Opening and closing the file named "DHM_parameters.txt"
        # Computation of a list of the MDH parameters of the robot
        if "fileDHM" in parameters:
            try: 
                fileDHM= parameters.get('fileDHM')
                with open(fileDHM, 'r') as filin:
                    lines = filin.readlines()
                    # Suppressing the lines put in comments or empty lines 
                    DHM_table_str= [ line.strip().split() for line in lines if ( not ('#' in line) and line != '\n') ]          
                    # Initialisation of DHM table
                    self.tableDHM = [[0 for x in range(len(DHM_table_str[y]))] for y in range(len(DHM_table_str))]
                    for i in range(len(DHM_table_str)):
                        for j in range(len(DHM_table_str[i])):                          
                            # Replacing string "pi" by its value
                            if ("pi" in DHM_table_str[i][j]):
                                DHM_table_str[i][j]=DHM_table_str[i][j].replace("pi","m.pi")
                            # Replacing string "pi" by its value
                            elif ("Pi" in DHM_table_str[i][j]):
                                DHM_table_str[i][j]=DHM_table_str[i][j].replace("Pi","m.pi")   
                            # Converting string into float for each element of the DHM table
                            self.tableDHM[i][j] = float(eval(DHM_table_str[i][j]))
                print("Attribute (list - float): self.tableDHM = ", self.tableDHM)
            except:
                print("Error, the file containing the MDH parameters is not found!")  
        # Storing DHM parameters related to the tool frame 
        if "toolFrameDHM" in parameters:
            try:
                toolFrameDHM = parameters.get('toolFrameDHM')
                assert len(toolFrameDHM) == 4
                # Extraction of the four DHM parameters defining the tool frame R_E
                self.toolDHM=toolFrameDHM
                print("Attribute (list - float): self.toolDHM = ", self.toolDHM)
            except AssertionError:
                print("Error, four vectors of DHM parameters for the tool frame description should be specified")     
        # CoordCentersMass
        if "coordCentersMass" in parameters:
            try: # verification of the existence  of the key 'coordCentersMass' in the dictionary
                coordCentersMass = parameters.get('coordCentersMass')
                assert (coordCentersMass is not None)
                try: # verification of the size of 'coordCentersMass' 
                    assert len(coordCentersMass) == self.numberJoints 
                    self.coordCentersMass = coordCentersMass
                    print("Attribute (list - array of float): self.coordCentersMass = ", self.coordCentersMass)
                except AssertionError:
                    print("Error, the number of coordCenterMass should be equal to the length of 'jointsType'")
            except:
                pass
        # InertiaLinks
        if "InertiaLinks" in parameters:
            try: # verification of the existence  of the key 'inertia' in the dictionary
                inertiaLinks = parameters.get('InertiaLinks')
                assert (inertiaLinks is not None)
                try: # verification of the size of 'coordCentersMass' 
                    assert len(inertiaLinks) == self.numberJoints 
                    self.inertiaLinks = inertiaLinks
                    print("Attribute (list - array of float): self.inertiaLinks = ", self.inertiaLinks)
                except AssertionError:
                    print("Error, the number of inertia matrices should be equal to the length of 'jointsType'")
            except:
                pass
        # Mass
        if "Mass" in parameters:
            try: # verification of the existence  of the key 'Mass' in the dictionary
                mass = parameters.get('Mass')
                assert (mass is not None)
                try: # verification of the size of 'coordCentersMass' 
                    assert len(mass) == self.numberJoints 
                    self.mass = mass
                    print("Attribute (list - float): self.mass = ", self.mass)
                except AssertionError:
                    print("Error, the number of mass properties should be equal to the length of 'jointsType'")
            except:
                pass
        # MotorInertia
        if "MotorInertia" in parameters:
            try: # verification of the existence  of the key 'MotorInertia' in the dictionary
                motorInertia = parameters.get('MotorInertia')
                assert (motorInertia is not None)
                try: # verification of the size of 'coordCentersMass' 
                    assert len(motorInertia) == self.numberJoints 
                    self.motorInertia = motorInertia
                    print("Attribute (list - float): self.motorInertia = ", self.motorInertia)
                except AssertionError:
                    print("Error, the number of motor Inertia properties should be equal to the length of 'jointsType'")
            except:
                pass
        # ReductionRatio
        if "ReductionRatio" in parameters:
            try: # verification of the existence  of the key 'ReductionRatio' in the dictionary
                reductionRatio = parameters.get('ReductionRatio')
                assert (reductionRatio is not None)
                try: # verification of the size of 'coordCentersMass' 
                    assert len(reductionRatio) == self.numberJoints 
                    self.reductionRatio = reductionRatio
                    print("Attribute (list - float): self.reductionRatio = ", self.reductionRatio)
                except AssertionError:
                    print("Error, the number of reduction ratio parameters should be equal to the length of 'jointsType'")
            except:
                pass
