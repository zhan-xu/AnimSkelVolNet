import numpy as np
import open3d as o3d


def find_lines_from_tree(root, line_list, pos_list):
    if not root.children:
        return
    else:
        for ch in root.children:
            pos_list.append(list(root.pos))
            pos_list.append(list(ch.pos))
            line_list.append([len(pos_list)-2, len(pos_list)-1])
            find_lines_from_tree(ch, line_list, pos_list)


def drawCube(center, radius, color=[0.0,0.0,0.0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def show_obj_skel(mesh, root):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # show obj mesh
    line_set_mesh = o3d.geometry.LineSet()
    line_set_mesh.points = o3d.utility.Vector3dVector(mesh.v)
    lines_mesh = np.concatenate((mesh.f[:, [0, 1]] - 1, mesh.f[:, [0, 2]] - 1, mesh.f[:, [1, 2]] - 1), axis=0)
    line_set_mesh.lines = o3d.utility.Vector2iVector(lines_mesh)
    colors = [[0.8, 0.8, 0.8] for i in range(len(lines_mesh))]
    line_set_mesh.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set_mesh)

    # show skel
    line_list_skel = []
    joint_pos_list = []
    find_lines_from_tree(root, line_list_skel, joint_pos_list)
    line_set_skel = o3d.geometry.LineSet()
    for joint_pos in joint_pos_list:
        vis.add_geometry(drawCube(joint_pos, 0.007, color=[1.0,0.0,0.0]))

    line_set_skel.points = o3d.utility.Vector3dVector(joint_pos_list)
    line_set_skel.lines = o3d.utility.Vector2iVector(line_list_skel)
    colors = [[0.0, 0.0, 1.0] for i in range(len(line_list_skel))]
    line_set_skel.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set_skel)

    vis.run()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image




