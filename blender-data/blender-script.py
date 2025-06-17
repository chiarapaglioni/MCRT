import bpy
import os

# Output directory
output_dir = os.path.expanduser("~/Desktop/blender_output")
os.makedirs(output_dir, exist_ok=True)

# Configure Cycles
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 4
bpy.context.scene.cycles.use_denoising = False

# Enable passes
view_layer = bpy.context.scene.view_layers["ViewLayer"]
view_layer.use_pass_normal = True  # Normals
view_layer.use_pass_z = True       # Depth
view_layer.use_pass_diffuse_color = True  # Albedo

# Set output format
bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
bpy.context.scene.render.image_settings.color_depth = '32'

# Render (generates all passes)
bpy.context.scene.render.filepath = os.path.join(output_dir, "noisy.exr")
bpy.ops.render.render(write_still=True)  # Saves noisy.exr

# Now extract and save each pass using the Compositor nodes
scene = bpy.context.scene
node_tree = scene.node_tree

if node_tree is None:
    # Enable Compositor and use nodes
    scene.use_nodes = True
    node_tree = scene.node_tree

# Clear existing nodes
for node in node_tree.nodes:
    node_tree.nodes.remove(node)

# Create a Render Layers node
render_layers_node = node_tree.nodes.new('CompositorNodeRLayers')
render_layers_node.layer = "ViewLayer"

# Define passes to save
passes_to_save = {
    "Image": "noisy.exr",       # Already saved, but we can re-save if needed
    "Normal": "normals.exr",
    "Depth": "depth.exr",
    "DiffCol": "albedo.exr"
}

for pass_name, file_name in passes_to_save.items():
    # Create a new output node
    output_node = node_tree.nodes.new('CompositorNodeOutputFile')
    output_node.base_path = output_dir
    output_node.format.file_format = 'OPEN_EXR'
    output_node.format.color_depth = '32'
    
    # Set the output file name
    output_node.file_slots[0].path = file_name.replace(".exr", "")  # Blender adds .exr automatically
    
    # Link the pass to the output node
    if pass_name == "Image":
        node_tree.links.new(render_layers_node.outputs["Image"], output_node.inputs[0])
    else:
        node_tree.links.new(render_layers_node.outputs[pass_name], output_node.inputs[0])
    
    # Execute the node setup to save the pass
    bpy.ops.render.render()
    
    # Remove the output node for the next pass
    node_tree.nodes.remove(output_node)

print(f"All passes saved to: {output_dir}")