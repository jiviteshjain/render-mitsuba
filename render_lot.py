from render import Renderer

from argparse import ArgumentParser
import os
import re

import numpy as np
#1,1,0.9  0.5,1.5,4.5
#1,1,1.5  0.7,1,4.5
#-0.1,-0.1,2.4  1,1,2
#0.9,0.9,2
XML_HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="1.5,1.5,2.7" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="4"/>
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
            <lookat origin="0,0,-0.1" target="-2.5,-2.5,10" up="0,0,1"/>
        </transform>
    </shape>
    
    <shape type="sphere">
        <float name="radius" value="1.6"/>
        <transform name="toWorld">
            <translate x="0" y="0" z="5"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="11,11,11"/>
        </emitter>
    </shape>

    <emitter type="constant">
        <rgb name="radiance" value="0.18,0.18,0.18"/>
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

Rs = []

Ss = [
    """
<!DOCTYPE ViewState>
<project>
 <VCGCamera FocalMm="32.833435" TranslationVector="0.664746 -2.25634 -0.372164 1" ViewportPx="1577 1027" CenterPx="788 513" CameraType="0" BinaryData="0" LensDistortion="0 0" PixelSizeMm="0.0369161 0.0369161" RotationMatrix="0.727616 0.0788547 0.681437 0 0.643869 0.264198 -0.718076 0 -0.236658 0.96124 0.141462 0 0 0 0 1 "/>
 <ViewSettings NearPlane="0.30310887" FarPlane="7.4027324" TrackScale="1.245262"/>
</project>
""",
"""
<!DOCTYPE ViewState>
<project>
 <VCGCamera BinaryData="0" CameraType="0" ViewportPx="1577 1027" PixelSizeMm="0.0369161 0.0369161" LensDistortion="0 0" RotationMatrix="-0.275013 0.772173 0.572815 0 -0.95127 -0.304969 -0.0456044 0 0.139476 -0.557444 0.818415 0 0 0 0 1 " FocalMm="32.833454" TranslationVector="-0.49805 1.43134 -2.11628 1" CenterPx="788 513"/>
 <ViewSettings TrackScale="1.1246486" NearPlane="0.30310887" FarPlane="8.6538372"/>
</project>
""",
"""
<!DOCTYPE ViewState>
<project>
 <VCGCamera BinaryData="0" FocalMm="32.833454" LensDistortion="0 0" TranslationVector="1.41071 0.41359 -1.67635 1" CenterPx="788 513" RotationMatrix="-0.673956 -0.396539 -0.62333 0 0.434634 -0.895095 0.099491 0 -0.597392 -0.203868 0.775604 0 0 0 0 1 " PixelSizeMm="0.0369161 0.0369161" ViewportPx="1577 1027" CameraType="0"/>
 <ViewSettings NearPlane="0.30310887" FarPlane="8.6567221" TrackScale="1.3598638"/>
</project>

""",
"""
<!DOCTYPE ViewState>
<project>
 <VCGCamera CameraType="0" LensDistortion="0 0" PixelSizeMm="0.0369161 0.0369161" TranslationVector="-0.791357 -1.14453 1.85737 1" CenterPx="788 513" ViewportPx="1577 1027" BinaryData="0" RotationMatrix="-0.449984 -0.653999 -0.608112 0 -0.80566 0.59106 -0.0394969 0 0.385262 0.472159 -0.792868 0 0 0 0 1 " FocalMm="32.833454"/>
 <ViewSettings FarPlane="8.79844" NearPlane="0.30310887" TrackScale="1.3827369"/>
</project>
""",
"""
<!DOCTYPE ViewState>
<project>
 <VCGCamera FocalMm="32.833435" TranslationVector="0.664746 -2.25634 -0.372164 1" ViewportPx="1577 1027" CenterPx="788 513" CameraType="0" BinaryData="0" LensDistortion="0 0" PixelSizeMm="0.0369161 0.0369161" RotationMatrix="0.727616 0.0788547 0.681437 0 0.643869 0.264198 -0.718076 0 -0.236658 0.96124 0.141462 0 0 0 0 1 "/>
 <ViewSettings NearPlane="0.30310887" FarPlane="7.4027324" TrackScale="1.245262"/>
</project>
""",
"""
<!DOCTYPE ViewState>
<project>
 <VCGCamera BinaryData="0" FocalMm="32.833454" LensDistortion="0 0" TranslationVector="1.41071 0.41359 -1.67635 1" CenterPx="788 513" RotationMatrix="-0.673956 -0.396539 -0.62333 0 0.434634 -0.895095 0.099491 0 -0.597392 -0.203868 0.775604 0 0 0 0 1 " PixelSizeMm="0.0369161 0.0369161" ViewportPx="1577 1027" CameraType="0"/>
 <ViewSettings NearPlane="0.30310887" FarPlane="8.6567221" TrackScale="1.3598638"/>
</project>
""",
]
# rotation_string = ""

for rotation_string in Ss:
    
    if rotation_string.strip() == '':
        R = np.eye(3)

    else:
        match = re.search(r'RotationMatrix=".*0 0 0 1 "', rotation_string)
        s = str(match.group(0))
        elements = [float(x) for x in s.split('=')[1].strip('"').split()]
        R = np.array(elements).reshape((4, 4))
        R = R[:3, :3].T

    Rs.append(R)



def run():
    parser = ArgumentParser()

    parser.add_argument('--self_color_files', type=str, nargs='+', default=[])
    parser.add_argument('--default_color_files', type=str, nargs='+', default=[])
    parser.add_argument('--bg_color_files', type=str, nargs='+', default=[])
    
    parser.add_argument('--radius', type=float, default=0.020 * 2)

    parser.add_argument('--spacing', type=float, default=2.5)
    
    parser.add_argument('--mitsuba', type=str, default='/Users/jiviteshjain/Installs/mitsuba2/build/dist/mitsuba', help='Path to mitsuba2 installation.')
    parser.add_argument('--out_file', type=str, required=True, help='Output file path.')
    parser.add_argument('--in_dir', type=str, default='', help='Optional path prefex for input files.')
    
    args = parser.parse_args()

    renderer = Renderer(args.mitsuba, args.radius * 0.5, args.radius, 600, args.out_file, XML_HEAD, XML_TAIL, XML_SPHERE, scale_radius=True)
    
    for i, file in enumerate(args.self_color_files):
        if i != 100:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'self', Rs[i // 4])
        else:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'self', Rs[0])

    renderer.preprocess_grid(ROW_DIR, COL_DIR, args.spacing * 1, args.spacing * 1.1)

    for i, file in enumerate(args.default_color_files):
        if i != 100:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'default', Rs[i // 4])
        else:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'default', np.eye(3))

    renderer.preprocess_grid(ROW_DIR, COL_DIR, args.spacing * 1, args.spacing * 1.1, offset=len(args.self_color_files))

    for i, file in enumerate(args.bg_color_files):
        if i != 100:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'bg', Rs[i // 4])
        else:
            renderer.add_pcd(os.path.join(args.in_dir, file), 'bg', Rs[0])

    renderer.preprocess_grid(ROW_DIR, COL_DIR, args.spacing * 1, args.spacing * 1.1,  offset=len(args.self_color_files)+len(args.default_color_files))
    
    renderer.process()

if __name__ == '__main__':
    run()