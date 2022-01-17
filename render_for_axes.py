import numpy as np
from plyfile import PlyData
from scipy.spatial import distance_matrix

import os
import subprocess
from argparse import ArgumentParser
import uuid

XML_HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
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
            <translate x="0" y="0" z="-0.5"/>
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

    def standardize_bbox(self, xyz):
        mins = np.amin(xyz, axis=0)
        maxs = np.amax(xyz, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs - mins)
        print(f'Center: {center}, Scale: {scale}')

        if self.scale_radius:
            self.small_radius /= scale
            self.big_radius /= scale

        return ((xyz - center) / scale).astype(np.float32)  # [-0.5, 0.5]

    def default_colormap(self, xyz):
        return [0.3, 0.3, 0.3]
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
        # print(xyz.shape)

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
        # num_rows = np.floor(np.sqrt(num_clouds))
        num_rows = 4

        # self.data[1]['xyz'] += (ROW_DIR * spacing)

        for i, data_ in enumerate(data):
            col = i // num_rows
            row = i % num_rows

            col_trans = col_dir * col_spacing * col
            row_trans = row_dir * row_spacing * row

            data_['xyz'] += (row_trans + col_trans)

    def process(self):
        points = np.concatenate([t['xyz'] for t in self.data], axis=0)
        points = self.standardize_bbox(points)
        points = points[:, (2, 0, 1)]
        points[:, 0] *= -1
        points[:, 2] += 0.0125

        print(f'{points.shape[0]} points from {len(self.data)} files.')

        xml_segments = [self.xml_head, ]
        offset = 0

        for data in self.data:
            radius = self.small_radius if data['variant'] == 'bg' else self.big_radius
            origin = None
            lim = data['xyz'].shape[0] - 4
            for j in range(data['xyz'].shape[0]):
                # print(j)
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

                print(j)
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
        renderer.add_pcd(os.path.join(args.in_dir, file), 'self')
    
    for file in args.default_color_files:
        renderer.add_pcd(os.path.join(args.in_dir, file), 'default')

    for file in args.bg_color_files:
        renderer.add_pcd(os.path.join(args.in_dir, file), 'bg')

    renderer.process()




if __name__ == '__main__':
    run()