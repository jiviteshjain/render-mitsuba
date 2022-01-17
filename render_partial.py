from render import Renderer

from argparse import ArgumentParser
import os
import re

import numpy as np
#1,1,0.9  0.5,1.5,4.5
#1,1,1.5  0.7,1,4.5
#-0.1,-0.1,2.4  1,1,2
XML_HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="2,2,2.3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="1024"/>
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

    <bsdf type="plastic" id="surfaceMaterial2">
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>

"""

XML_TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="0,0,-0.19" target="-2.5,-2.5,10" up="0,0,1"/>
        </transform>
    </shape>
    
    <shape type="sphere">
        <float name="radius" value="1.6"/>
        <transform name="toWorld">
            <translate x="0" y="0" z="5"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="10,10,10"/>
        </emitter>
    </shape>

    <emitter type="constant">
        <rgb name="radiance" value="0.35,0.35,0.35"/>
    </emitter>
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

ROW_DIR = np.array([-3, 3, 0], dtype=np.float32)
ROW_DIR /= np.linalg.norm(ROW_DIR)
ROW_DIR = ROW_DIR[(1, 2, 0), ]
ROW_DIR[0] *= -1

COL_DIR = np.array([3, 3, -1.5], dtype=np.float32)
COL_DIR /= np.linalg.norm(COL_DIR)
COL_DIR = COL_DIR[(1, 2, 0), ]
COL_DIR[0] *= -1

print(ROW_DIR, COL_DIR)

rotation_string = """
<!DOCTYPE ViewState>
<project>
 <VCGCamera RotationMatrix="0.463326 0.292905 0.836382 0 0.576064 0.617638 -0.535419 0 -0.673409 0.729884 0.117436 0 0 0 0 1 " FocalMm="32.833416" CenterPx="788 513" CameraType="0" LensDistortion="0 0" ViewportPx="1577 1027" PixelSizeMm="0.0369161 0.0369161" BinaryData="0" TranslationVector="1.72784 -1.6932 -0.313683 1"/>
 <ViewSettings FarPlane="7.4027324" NearPlane="0.30310887" TrackScale="1.245262"/>
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



def run():
    parser = ArgumentParser()

    parser.add_argument('--self_color_files', type=str, nargs='+', default=[])
    parser.add_argument('--default_color_files', type=str, nargs='+', default=[])
    parser.add_argument('--bg_color_files', type=str, nargs='+', default=[])
    
    parser.add_argument('--radius', type=float, default=0.007 * 1.6)

    parser.add_argument('--spacing', type=float, default=2)
    
    parser.add_argument('--mitsuba', type=str, default='/Users/jiviteshjain/Installs/mitsuba2/build/dist/mitsuba', help='Path to mitsuba2 installation.')
    parser.add_argument('--out_file', type=str, required=True, help='Output file path.')
    parser.add_argument('--in_dir', type=str, default='', help='Optional path prefex for input files.')
    
    args = parser.parse_args()

    renderer = Renderer(args.mitsuba, args.radius * 0.5, args.radius, 700, args.out_file, XML_HEAD, XML_TAIL, XML_SPHERE, scale_radius=True)
    
    for i, file in enumerate(args.self_color_files):
        if i != 100:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'self', R)
        else:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'self', np.eye(3))

    renderer.preprocess_grid(ROW_DIR, COL_DIR, args.spacing * 0.6, args.spacing * 1.2)

    for i, file in enumerate(args.default_color_files):
        if i != 100:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'default', R @ R)
        else:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'default', np.eye(3))

    renderer.preprocess_grid(ROW_DIR, COL_DIR, args.spacing * 0.6, args.spacing * 1.2, offset=len(args.self_color_files))

    for i, file in enumerate(args.bg_color_files):
        if i != 100:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'bg', R)
        else:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'bg', np.eye(3))

    renderer.preprocess_grid(ROW_DIR, COL_DIR, args.spacing * 0.6, args.spacing * 1.2,  offset=len(args.self_color_files)+len(args.default_color_files))
    
    renderer.process()

if __name__ == '__main__':
    run()