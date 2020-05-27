"""
The script can be used to bind predicted skeleton and original model together, with geodesic voxel skinning method of maya
"""

import pymel.core as pm

def getGeometryGroups():
    geo_list = []
    geometries = cmds.ls(type='surfaceShape')
    for geo in geometries:
        if 'ShapeOrig' in geo:
            '''
            we can also use cmds.ls(geo, l=True)[0].split("|")[0]
            to get the upper level node name, but stick on this way for now
            '''
            geo_name = geo.replace('ShapeOrig', '')
            geo_list.append(geo_name)
    if not geo_list:
        geo_list = cmds.ls(type='surfaceShape')
    return geo_list


def load_skel(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
    for li in lines:
        words = li.split()
        if words[5] == 'None':
            root = words[1]
            print 'root: '+root
            pos = (float(words[2]), float(words[3]), float(words[4]))
            cmds.joint(p=(pos[0], pos[1], pos[2]), name=root)
            break
    this_level = [root]
    while this_level:
        next_level = []
        for pname in this_level:
            for li in lines:
                words = li.split()
                name_li = words[1]
                name_pa = words[5]
                if name_pa == pname:
                    #print name_li
                    cmds.select(pname, r=True)
                    pos = (float(words[2]), float(words[3]), float(words[4]))
                    cmds.joint(p=(pos[0], pos[1], pos[2]), name=name_li)
                    next_level.append(name_li)
        this_level = next_level
    return root


if __name__ == '__main__':
    obj_name = 'DATA_PATH\\obj\\1195.obj'
    skel_name = 'DATA_PATH\\skel\\1195.txt'
    
    # import obj
    cmds.file(new=True,force=True)
    cmds.file(obj_name, o=True)

    #import skel
    root = load_skel(skel_name)
    
    # geodesic volumetric skinning
    geo_list = getGeometryGroups() 
    cmds.skinCluster(root, geo_list[0]) # The line only works for mesh with a single component. If the mesh has multiple groups, this line is incorrect!
    cmds.select('skinCluster1', r=True)
    cmds.select(geo_list[0], add=True)
    cmds.geomBind(bm=3, fo=0.5, mi=3) # adjust the parameters
    cmds.skinPercent('skinCluster1', geo_list[0], pruneWeights=0.2) # adjust the parameters
  
    # export fbx
    # pm.mel.FBXExport(f=out_name)