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
        PATH TRACING
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
        albedo_np = np.array(albedo_image)

        if self.debug:
            print("Albedo Image shape:", albedo_np.shape)
            print(f"Albedo Min {albedo_np.min()} - Max {albedo_np.max()}")
        return albedo_np

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
        normal_np = np.array(normal_image)

        if self.debug:
            print("Normal Image shape:", normal_np.shape)
            print(f"Normal Min {normal_np.min()} - Max {normal_np.max()}")
        return normal_np


    def render_patch(self, spp, crop_offset, crop_size, seed=0):
        """
        Render a rectangular patch (crop window) of the image at given spp.

        Parameters:
        - spp (int): samples per pixel
        - crop_offset (tuple): (x_start, y_start) pixel coordinates of patch origin
        - crop_size (tuple): (width, height) size of the patch
        - seed (int): random seed

        Returns:
        - patch_image (ndarray): (patch_height, patch_width, 3)
        """
        import mitsuba as mi
        mi.set_variant("scalar_rgb")
        scene = mi.load_file(str(self.scene_path))

        old_sensor = scene.sensors()[0]
        params = mi.traverse(old_sensor)

        x_start, y_start = crop_offset
        crop_width, crop_height = crop_size

        # Build sensor with crop window film
        sensor = mi.load_dict({
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
                'crop_window': [x_start / self.width,
                                y_start / self.height,
                                (x_start + crop_width) / self.width,
                                (y_start + crop_height) / self.height],
                'rfilter': {'type': 'box'},
                'pixel_format': 'rgb',
                'component_format': 'float32'
            }
        })

        integrator = scene.integrator()

        patch_image = mi.render(scene, sensor=sensor, integrator=integrator, seed=seed)
        patch_np = np.array(patch_image)

        # Extract the patch region pixels only
        patch_np_cropped = patch_np[y_start:y_start + crop_height, x_start:x_start + crop_width, :]
        return patch_np_cropped

    def adaptive_patch_render(self, base_spp, importance_map, patch_size=32, threshold=0.5, extra_spp=16):
        """
        Adaptive rendering by splitting image into patches and rendering
        only those patches whose importance exceeds the threshold with extra spp.

        Parameters:
        - base_spp (int): spp for the full base image render
        - importance_map (2D ndarray): importance per pixel (H, W)
        - patch_size (int): size of square patch
        - threshold (float): importance threshold (0-1) to decide if patch needs extra spp
        - extra_spp (int): spp to use for important patches

        Returns:
        - final_image (ndarray): (H, W, 3) image with patches combined
        """
        H, W = importance_map.shape
        assert H == self.height and W == self.width

        # Step 1: Render full image at base spp
        base_image = self.generate_samples_from_scene(seed=0, spp=base_spp)

        # Step 2: Normalize importance map [0,1]
        imp_norm = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)

        final_image = base_image.copy()

        # Step 3: Iterate patches, check if patch importance max > threshold, then re-render
        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                patch_imp = imp_norm[y:y + patch_size, x:x + patch_size]
                if patch_imp.size == 0:
                    continue
                if patch_imp.max() >= threshold:
                    # Render patch with extra spp
                    patch_w = min(patch_size, W - x)
                    patch_h = min(patch_size, H - y)
                    patch_img = self.render_patch(
                        spp=extra_spp,
                        crop_offset=(x, y),
                        crop_size=(patch_w, patch_h),
                        seed=42 + x + y  # arbitrary seed per patch
                    )
                    # Replace patch pixels in final image
                    final_image[y:y + patch_h, x:x + patch_w, :] = patch_img

                    if self.debug:
                        print(f"Re-rendered patch at ({x}, {y}) with spp={extra_spp}")

        return final_image