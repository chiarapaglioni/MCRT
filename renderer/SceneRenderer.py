import numpy as np
import tifffile
import mitsuba as mi

WIDTH = 512
HEIGHT = 512
DEBUG = False

class SceneRenderer:
    def __init__(self, scene_path, width=WIDTH, height=HEIGHT, debug=DEBUG):
        self.scene_path = scene_path
        self.width = width
        self.height = height
        self.debug = debug

    def generate_samples_from_scene(self, seed, spp=1):
        """
        Generate radiance samples from a scene patch using full-image rendering.

        Parameters:
        - seed (int): random seed
        - spp (int): samples per pixel

        Returns:
        - image (ndarray): (height, width, 3)
        """
        print("Using.. scalar_rgb")
        mi.set_variant("scalar_rgb")
        scene = mi.load_file(str(self.scene_path))

        x, y = self.width // 2, self.height // 2

        old_sensor = scene.sensors()[0]
        params = mi.traverse(old_sensor)

        # Retrieve integrator
        old_integrator = scene.integrator()
        integrator_params = mi.traverse(old_integrator)

        new_sensor = mi.load_dict({
            'type': "perspective",
            'fov': params['x_fov'],
            'near_clip': old_sensor.near_clip(),
            'far_clip': old_sensor.far_clip(),
            'to_world': old_sensor.world_transform(),
            'sampler': {
                'type': 'independent',
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.width,
                'height': self.height,
                'rfilter': {'type': 'box'},
                'pixel_format': 'rgb',
                'component_format': 'float32'
            }
        })

        max_depth = integrator_params.get('max_depth', -1)
        hide_emitters = integrator_params.get('hide_emitters', False)

        new_integrator = mi.load_dict({
            'type': 'path',
            'max_depth': max_depth,
            'hide_emitters': hide_emitters
        })

        image = mi.render(scene, spp=spp, sensor=new_sensor, integrator=new_integrator, seed=seed)

        if self.debug:
            print("Rendered Image:", image.shape)
            print("Center Pixel RGB:", image[x, y, :])

        return np.array(image)

    def render_n_images(self, n, spp=1, seed_start=0):
        """
        Renders N images and stores them in a list.

        Returns:
        - List of (H, W, 3) images
        """
        images = []
        for i in range(n):
            seed = seed_start + i
            img = self.generate_samples_from_scene(seed, spp)
            if self.debug:
                print(f"Rendered image {i+1}/{n} with seed {seed}")
            images.append(img)
        return images

    def save_images_to_tiff(self, images, output_path):
        """
        Saves a list of images to a multi-page TIFF file.
        """
        stack = np.stack(images, axis=0)
        tifffile.imwrite(output_path, stack, photometric='rgb', dtype='float32')
        if self.debug:
            print(f"Saved {len(images)} images to {output_path}")


    def render_albedo_image(self):
        """
        Render the albedo (diffuse reflectance) of the scene using Mitsuba's 'field' integrator.

        Returns:
        - albedo_image (ndarray): (height, width, 3)
        """
        print("Using.. scalar_rgb (for albedo)")
        mi.set_variant("scalar_rgb")
        scene = mi.load_file(str(self.scene_path))

        old_sensor = scene.sensors()[0]
        params = mi.traverse(old_sensor)

        sensor = mi.load_dict({
            'type': "perspective",
            'fov': params['x_fov'],
            'near_clip': old_sensor.near_clip(),
            'far_clip': old_sensor.far_clip(),
            'to_world': old_sensor.world_transform(),
            'sampler': {
                'type': 'independent',
                'sample_count': 1
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.width,
                'height': self.height,
                'rfilter': {'type': 'box'},
                'pixel_format': 'rgb',
                'component_format': 'float32'
            }
        })

        integrator = mi.load_dict({
            'type': 'aov',
            'aovs': 'aa:albedo'
        })

        albedo_image = mi.render(scene, sensor=sensor, integrator=integrator)

        if self.debug:
            print("Albedo Image shape:", albedo_image.shape)

        return np.array(albedo_image)

    def render_normal_image(self):
        """
        Render the shading normals of the scene using Mitsuba's 'field' integrator.

        Returns:
        - normal_image (ndarray): (height, width, 3), normals in [0,1] range
        """
        print("Using.. scalar_rgb (for normals)")
        mi.set_variant("scalar_rgb")
        scene = mi.load_file(str(self.scene_path))

        old_sensor = scene.sensors()[0]
        params = mi.traverse(old_sensor)

        sensor = mi.load_dict({
            'type': "perspective",
            'fov': params['x_fov'],
            'near_clip': old_sensor.near_clip(),
            'far_clip': old_sensor.far_clip(),
            'to_world': old_sensor.world_transform(),
            'sampler': {
                'type': 'independent',
                'sample_count': 1
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.width,
                'height': self.height,
                'rfilter': {'type': 'box'},
                'pixel_format': 'rgb',
                'component_format': 'float32'
            }
        })

        integrator = mi.load_dict({
            'type': 'aov',
            'aovs': 'nn:sh_normal',
        })

        normal_image = mi.render(scene, sensor=sensor, integrator=integrator)

        if self.debug:
            print("Normal Image shape:", normal_image.shape)

        # Normals are in [-1,1], convert to [0,1] for visualization/storage
        normal_image = (np.array(normal_image) + 1.0) * 0.5

        return normal_image
