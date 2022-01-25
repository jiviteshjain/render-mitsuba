import numpy as np
from plyfile import PlyData
from scipy.spatial import distance_matrix
import open3d as o3d

import os
import subprocess
from argparse import ArgumentParser
import uuid
import re

XML_HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="5"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="1024"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1280"/>
            <integer name="height" value="720"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""

XML_TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-1.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

XML_SPHERE = """
    <shape type="sphere">
        <float name="radius" value="{radius}"/>
        <transform name="toWorld">
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
"""

XML_CYL = """
    <shape type="cylinder">
        <float name="radius" value="0.004"/>
        <point name="p0" x="0" y="0" z="0"/>
        <point name="p1" x="0" y="0" z="0.16"/>
        <transform name="toWorld">
            <lookat origin="{x0},{y0},{z0}" target="{x1},{y1},{z1}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
"""

rotation_string = """
<!DOCTYPE ViewState>
<project>
 <VCGCamera FocalMm="32.833435" LensDistortion="0 0" PixelSizeMm="0.0369161 0.0369161" BinaryData="0" TranslationVector="-0.73592 1.35511 -1.30807 1" CenterPx="788 513" RotationMatrix="-0.902698 -0.0731013 0.424019 0 -0.235703 -0.740437 -0.629442 0 0.359972 -0.668139 0.651161 0 0 0 0 1 " CameraType="0" ViewportPx="1577 1027"/>
 <ViewSettings TrackScale="1.3827369" NearPlane="0.30310887" FarPlane="8.79844"/>
</project>

"""
# rotation_string = ""

if rotation_string.strip() == '':
    R = np.eye(3)

else:
    match = re.search(r'RotationMatrix=".*0 0 0 1 "', rotation_string)
    s = str(match.group(0))
    elements = [float(x) for x in s.split('=')[1].strip('"').split()]
    R = np.array(elements).reshape((4, 4))
    R = R[:3, :3].T

class Renderer:
    def __init__(self, mitsuba, small_radius, big_radius, downsample_bg, out_file, xml_head, xml_tail, xml_sphere, scale_radius=False):
        self.mitsuba = mitsuba
        self.small_radius = small_radius
        self.big_radius = big_radius
        self.downsample_bg = downsample_bg
        self.out_file = out_file
        self.xml_head = xml_head
        self.xml_tail = xml_tail
        self.xml_sphere = xml_sphere
        self.scale_radius = scale_radius
        self.data = []

    def standardize_bbox(self, xyz, box=None):
        if box is None:
            box = {}
            box['mins'] = np.amin(xyz, axis=0)
            box['maxs'] = np.amax(xyz, axis=0)
            box['center'] = (box['mins'] + box['maxs']) / 2.
            box['scale'] = np.amax(box['maxs'] - box['mins'])
        # scale = 12.365791555384012
        
        print(f"Center: {box['center']}, Scale: {box['scale']}")

        if self.scale_radius:
            self.small_radius /= box['scale']
            self.big_radius /= box['scale']

        return ((xyz - box['center']) / box['scale']).astype(np.float32), box  # [-0.5, 0.5]

    def default_colormap(self, xyz):
        # return [1.0, 0.4980392156862745, 0.054901960784313725] #yellow
        # return [0.17254901960784313, 0.6274509803921569, 0.17254901960784313] #green
        # return [0.12156862745098039, 0.4666666666666667, 0.7058823529411765] #blue
        return [0.3, 0.3, 0.3] #grey
        xyz = xyz.copy()
        xyz += 0.5
        xyz[2] -= 0.0125

        xyz = np.clip(xyz, 0.001, 1.0)
        norm = np.sqrt(np.sum(xyz ** 2))
        xyz /= norm
        return xyz.tolist()

    def downsample_fps(self, x, idx=None):
        num_points = self.downsample_bg
        nv = x.shape[0]
        if num_points >= nv:
            return x
        # d = distance_matrix(x, x)
        if idx is None:
            idx = np.random.randint(low=0, high=nv - 1)
        elif idx == 'center':
            c = np.mean(x, axis=0, keepdims=True)
            d = distance_matrix(c, x)
            idx = np.argmax(d)

        y = np.zeros(shape=(num_points, 3))
        indices = np.zeros(shape=(num_points,), dtype=np.int32)
        p = x[np.newaxis, idx, ...]
        dist = distance_matrix(p, x)
        for i in range(num_points):
            y[i, ...] = p
            indices[i] = idx
            d = distance_matrix(p, x)
            dist = np.minimum(d, dist)
            idx = np.argmax(dist)
            p = x[np.newaxis, idx, ...]
        return y  # , indices

    def downsample_random(self, xyz):
        if self.downsample_bg >= xyz.shape[0]:
            return xyz
        
        pt_indices = np.random.choice(xyz.shape[0], self.downsample_bg, replace=False)
        np.random.shuffle(pt_indices)
        return xyz[pt_indices, :]

    def add_pcd(self, path, variant, rot=None):
        if path == 'boo':
            self.data.append(None)
            return
        ply = PlyData.read(path)
        
        xyz = np.column_stack((np.asarray(ply['vertex']['x']),
                               np.asarray(ply['vertex']['y']),
                               np.asarray(ply['vertex']['z'])))
        
        axes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        xyz = np.concatenate((xyz, axes,), axis=0)
        
        if rot is not None:
            xyz = xyz @ rot
        if variant == 'bg':
            xyz = self.downsample_fps(xyz)

        if variant == 'self':
            rgb = np.column_stack((np.asarray(ply['vertex']['red']),
                                   np.asarray(ply['vertex']['green']),
                                   np.asarray(ply['vertex']['blue']))).astype(np.float32) / 255
        else:
            rgb = None
        
        self.data.append({
            'xyz': xyz,
            'rgb': rgb,
            'variant': variant,
        })

    def preprocess_grid(self, row_dir, col_dir, row_spacing, col_spacing=None, offset=0):
        if col_spacing is None:
            col_spacing = row_spacing

        data = self.data[offset:]
        
        num_clouds = len(data)
        num_rows = np.floor(np.sqrt(num_clouds))
        num_rows = 1

        # self.data[1]['xyz'] += (ROW_DIR * spacing)

        for i, data_ in enumerate(data):
            row = i // num_rows
            col = i % num_rows

            col_trans = col_dir * col_spacing * col
            row_trans = row_dir * row_spacing * row
            if data_ is None:
                continue

            data_['xyz'] += (row_trans + col_trans)

    def process(self, rot_dir, angle, box=None):
        points = np.concatenate([t['xyz'] for t in self.data if t is not None], axis=0)
        rot_dir /= np.linalg.norm(rot_dir)
        rot_dir *= angle
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_dir)
        points, box = self.standardize_bbox(points, box)
        points = points @ R

        

        points = points[:, (2, 0, 1)]
        points[:, 0] *= -1
        points[:, 2] += 0.0125

        print(f'{points.shape[0]} points from {len(self.data)} files.')

        xml_segments = [self.xml_head, ]
        offset = 0

        for data in self.data:
            if data is None:
                continue
            radius = self.small_radius if data['variant'] == 'bg' else self.big_radius
            lim = data['xyz'].shape[0] - 4
            for j in range(data['xyz'].shape[0]):
                point = points[offset + j, :]

                if j == lim:
                    origin = point
                    continue

                if j > lim:
                    tip = point
                    print('Origin', origin, 'Tip', tip)
                    print(np.linalg.norm(tip - origin))
                    if j == lim+1:
                        color = [1,0,0]
                    elif j == lim+2:
                        color = [0,1,0]
                    else:
                        color = [0,0,1]
                    xml_segments.append(
                        XML_CYL.format(x0=origin[0], y0=origin[1], z0=origin[2],
                                       x1=tip[0], y1=tip[1], z1=tip[2],
                                       r=color[0], g=color[1], b=color[2])
                    )
                    sphere_tip = origin + (tip - origin) * 0.75
                    xml_segments.append(
                        self.xml_sphere.format(radius=0.008,
                                        x=sphere_tip[0], y=sphere_tip[1], z=sphere_tip[2],
                                        r=color[0], g=color[1], b=color[2])
                    )
                    print(j)
                    continue

                if data['variant'] == 'self':
                    color = data['rgb'][j, :].tolist()
                elif data['variant'] == 'bg':
                    color = [0.3, 0.3, 0.3]
                else:
                    color = self.default_colormap(point)



                xml_segments.append(
                    self.xml_sphere.format(radius=radius,
                                      x=point[0], y=point[1], z=point[2],
                                      r=color[0], g=color[1], b=color[2])
                )

            offset += data['xyz'].shape[0]
        
        xml_segments.append(self.xml_tail)
        xml_string = ''.join(xml_segments)

        temp_file_name = f'{uuid.uuid4()}.xml'
        with open(temp_file_name, 'w') as f:
            f.write(xml_string)

        subprocess.run([self.mitsuba, temp_file_name, '-o', self.out_file, '-m', 'packet_spectral'])
        os.remove(temp_file_name)

        return box





def run():
    parser = ArgumentParser()
    
    parser.add_argument('--self_color_files', type=str, nargs='+', default=[])
    parser.add_argument('--default_color_files', type=str, nargs='+', default=[])
    parser.add_argument('--bg_color_files', type=str, nargs='+', default=[])
    
    parser.add_argument('--small_radius', type=float, default=0.008)
    parser.add_argument('--big_radius', type=float, default=0.015)
    
    parser.add_argument('--mitsuba', type=str, default='/Users/jiviteshjain/Installs/mitsuba2/build/dist/mitsuba', help='Path to mitsuba2 installation.')
    parser.add_argument('--out_file', type=str, required=True, help='Output file path.')
    parser.add_argument('--in_dir', type=str, default='', help='Optional path prefex for input files.')

    parser.add_argument('--downsample_bg', type=int, default=256)
    
    args = parser.parse_args()

    renderer = Renderer(args.mitsuba, args.small_radius, args.big_radius, args.downsample_bg, args.out_file, XML_HEAD, XML_TAIL, XML_SPHERE)
    
    for file in args.self_color_files:
        renderer.add_pcd(os.path.join(args.in_dir, file), 'self', R)
    
    for file in args.default_color_files:
        renderer.add_pcd(os.path.join(args.in_dir, file), 'default', R)

    for file in args.bg_color_files:
        renderer.add_pcd(os.path.join(args.in_dir, file), 'bg', R)

    renderer.process()




if __name__ == '__main__':
    run()