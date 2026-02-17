// Uncomment the next line to not show the console window
// #![windows_subsystem = "windows"]

extern crate sdl3;

use gltf::Document;
use sdl3::event::Event;
use sdl3::gpu::*;
use sdl3::keyboard::Keycode;
use sdl3::libc::c_float;
use sdl3::pixels::Color;
use sdl3::sys::gpu::*;
use std::time::Duration;

// The vertex input layout
#[allow(dead_code)] // Compiler doesn't see that the code is used to pass info to the gpu
#[derive(Copy, Clone, Debug)]
struct Vertex {
    // vec3 Position
    x: c_float,
    y: c_float,
    z: c_float,
    // vec4 Color
    r: c_float,
    g: c_float,
    b: c_float,
    a: c_float,
    // vec2 Texture Coord
    u: c_float,
    v: c_float,
}

struct UniformBufferPosition {
    x_pos: c_float,
    y_pos: c_float,
}

struct PrimitiveData {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: u32,
}

struct MeshData {
    primitives: Vec<PrimitiveData>,
}

struct ImageData<'a> {
    texture: Texture<'a>,
    sampler: Sampler,
}

struct ModelData<'a> {
    meshes: Vec<MeshData>,
    images: Vec<ImageData<'a>>,
    document: Document,
}

fn load_model_and_copy_to_gpu<'a>(model_path: &str, gpu_device: &Device) -> ModelData<'a> {
    // Load the model with the gltf crate's import function
    let (model_gltf, buffers, images) = gltf::import(model_path).unwrap();

    // Start a copy pass
    let copy_command_buffer = gpu_device.acquire_command_buffer().unwrap();
    let copy_pass = gpu_device.begin_copy_pass(&copy_command_buffer).unwrap();

    let meshes_vec = model_gltf
        .meshes()
        .map(|mesh| -> MeshData {
            let primitives_vec = mesh
                .primitives()
                .map(|primitive| -> PrimitiveData {
                    // Create reader
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                    // Get texture coordinates
                    let mut tex_coord_temp_vector = Vec::new();
                    if let Some(iter) = reader.read_tex_coords(0) {
                        for tex_coord in iter.into_f32() {
                            let tex_coord_u = tex_coord[0];
                            let tex_coord_v = tex_coord[1];
                            tex_coord_temp_vector.push((tex_coord_u, tex_coord_v));
                        }
                    }
                    // reader.read_normals()
                    // reader.read_tangents()
                    // reader.read_joints(set)
                    // reader.read_weights(set)
                    // reader.read_morph_targets()

                    // Get vertex colors
                    let mut colors_temp_vector = Vec::new();
                    if let Some(iter) = reader.read_colors(0) {
                        for vertex_color in iter.into_rgba_f32() {
                            colors_temp_vector.push(vertex_color);
                        }
                    }

                    // Create vertices vector, then read vertex positions and add to vector
                    let mut vertices = Vec::new();
                    if let Some(iter) = reader.read_positions() {
                        for (index, vertex_position) in iter.enumerate() {
                            // Use vertex color from model data  or use default value (white)
                            let vertex_color = colors_temp_vector
                                .get(index)
                                .or(Some(&[1f32, 1f32, 1f32, 1f32]))
                                .unwrap();
                            // Use texture coordinate from model data or use default value (0, 0)
                            let texture_coordinate = tex_coord_temp_vector
                                .get(index)
                                .or(Some(&(0f32, 0f32)))
                                .unwrap();
                            vertices.push(Vertex {
                                x: vertex_position[0],
                                y: vertex_position[1],
                                z: vertex_position[2],
                                r: vertex_color[0],
                                g: vertex_color[1],
                                b: vertex_color[2],
                                a: vertex_color[3],
                                u: texture_coordinate.0,
                                v: texture_coordinate.1,
                            });
                        }
                    }

                    // Create indices vector, then read index values and put into vector
                    let mut indices = Vec::new();
                    if let Some(indices_raw) = reader.read_indices() {
                        indices.append(&mut indices_raw.into_u32().collect::<Vec<u32>>());
                    }

                    // How big is the vertex data to transfer
                    let primitive_vertices_size = (vertices.len() * size_of::<Vertex>()) as u32;

                    // How big is the index data to transfer
                    let primitive_indices_size = (indices.len() * size_of::<u32>()) as u32;

                    // Determine which is larger
                    let larger_size = primitive_vertices_size.max(primitive_indices_size);

                    // Create the transfer buffer, the vertices and the indices will both use this to upload to the gpu
                    let transfer_buffer = gpu_device
                        .create_transfer_buffer()
                        .with_size(larger_size)
                        .with_usage(TransferBufferUsage::UPLOAD)
                        .build()
                        .unwrap();

                    // Create the vertex buffer
                    let vertex_buffer = gpu_device
                        .create_buffer()
                        .with_size(primitive_vertices_size)
                        .with_usage(BufferUsageFlags::VERTEX)
                        .build()
                        .unwrap();

                    // Fill the transfer buffer
                    let mut buffer_mem_map = transfer_buffer.map(&gpu_device, true);
                    let buffer_mem_map_mem_mut: &mut [Vertex] = buffer_mem_map.mem_mut();

                    for (index, &value) in vertices.iter().enumerate() {
                        buffer_mem_map_mem_mut[index] = value;
                    }
                    buffer_mem_map.unmap();

                    // Where is the data
                    let data_location = TransferBufferLocation::default()
                        .with_transfer_buffer(&transfer_buffer)
                        .with_offset(0u32);

                    // Where to upload the data
                    let buffer_region = BufferRegion::default()
                        .with_buffer(&vertex_buffer)
                        .with_size(primitive_vertices_size)
                        .with_offset(0u32);

                    // Upload the data
                    copy_pass.upload_to_gpu_buffer(data_location, buffer_region, true);

                    // Index stuff

                    // Create the index buffer
                    let index_buffer = gpu_device
                        .create_buffer()
                        .with_size(primitive_indices_size)
                        .with_usage(BufferUsageFlags::INDEX)
                        .build()
                        .unwrap();

                    // Fill the transfer buffer
                    let mut buffer_mem_map = transfer_buffer.map(&gpu_device, true);

                    let buffer_mem_map_mem_mut: &mut [u32] = buffer_mem_map.mem_mut();

                    for (index, &value) in indices.iter().enumerate() {
                        buffer_mem_map_mem_mut[index] = value;
                    }

                    buffer_mem_map.unmap();

                    // Where is the data
                    let data_location = TransferBufferLocation::default()
                        .with_transfer_buffer(&transfer_buffer)
                        .with_offset(0u32);

                    // Where to upload the data
                    let buffer_region = BufferRegion::default()
                        .with_buffer(&index_buffer)
                        .with_size(primitive_indices_size)
                        .with_offset(0u32);

                    // Upload the data
                    // The next line causes the model to not show up...
                    copy_pass.upload_to_gpu_buffer(data_location, buffer_region, true);

                    // Release the index transfer buffer
                    drop(transfer_buffer);

                    PrimitiveData {
                        vertex_buffer: vertex_buffer,
                        index_buffer: index_buffer,
                        index_count: indices.len() as u32,
                    }
                })
                .collect();
            MeshData {
                primitives: primitives_vec,
            }
        })
        .collect();

    // Upload images
    let images_vec = images
        .iter()
        .map(|image| -> ImageData {
            let format_mode = match image.format {
                gltf::image::Format::R8 => TextureFormat::R8Unorm,
                gltf::image::Format::R8G8 => TextureFormat::R8g8Unorm,
                gltf::image::Format::R8G8B8 => TextureFormat::R8g8b8a8Unorm, // Not a perfect match
                gltf::image::Format::R8G8B8A8 => TextureFormat::R8g8b8a8Unorm,
                gltf::image::Format::R16 => TextureFormat::R16Unorm,
                gltf::image::Format::R16G16 => TextureFormat::R16g16Unorm,
                gltf::image::Format::R16G16B16 => TextureFormat::R16g16b16a16Unorm, // Not a perfect match
                gltf::image::Format::R16G16B16A16 => TextureFormat::R16g16b16a16Unorm, // Not a perfect match
                gltf::image::Format::R32G32B32FLOAT => TextureFormat::R32g32b32a32Float, // Not a perfect match
                gltf::image::Format::R32G32B32A32FLOAT => TextureFormat::R32g32b32a32Float,
            };
            let texture_create_info = TextureCreateInfo::new()
                .with_type(TextureType::_2D)
                .with_format(format_mode)
                .with_usage(TextureUsage::SAMPLER)
                .with_width(image.width)
                .with_height(image.height)
                .with_layer_count_or_depth(1)
                .with_num_levels(1);
            let texture = gpu_device.create_texture(texture_create_info).unwrap();

            let sampler_create_info = SamplerCreateInfo::new()
                // TODO - get sampler info from gltf model
                // Not sure how best to do that when starting with the images...
                .with_min_filter(Filter::Nearest)
                .with_mag_filter(Filter::Nearest)
                .with_mipmap_mode(SamplerMipmapMode::Nearest)
                .with_address_mode_u(SamplerAddressMode::Repeat)
                .with_address_mode_v(SamplerAddressMode::Repeat)
                .with_address_mode_w(SamplerAddressMode::Repeat);
            // .with_mip_lod_bias(value)
            // .with_max_anisotropy(value)
            // .with_compare_op(value)
            // .with_min_lod(value)
            // .with_max_lod(value)
            // .with_enable_anisotropy(enable)
            // .with_enable_compare(enable);
            let sampler = gpu_device.create_sampler(sampler_create_info).unwrap();

            let texture_transfer_buffer = gpu_device
                .create_transfer_buffer()
                .with_size(image.width * image.height * (size_of::<c_float>() as u32) * 4)
                .with_usage(TransferBufferUsage::UPLOAD)
                .build()
                .unwrap();
            let texture_transfer_info = TextureTransferInfo::new()
                .with_transfer_buffer(&texture_transfer_buffer)
                .with_offset(0)
                .with_pixels_per_row(image.width)
                .with_rows_per_layer(image.height);
            let texture_region = TextureRegion::new()
                .with_texture(&texture)
                .with_width(image.width)
                .with_height(image.height)
                .with_depth(1);

            let mut texture_buffer_mem_map = texture_transfer_buffer.map(&gpu_device, true);
            let texture_buffer_mem_map_mem_mut: &mut [_] = texture_buffer_mem_map.mem_mut();
            for pixel_coord in 0..(image.width * image.height) as usize {
                match image.format {
                    gltf::image::Format::R8G8B8 => {
                        texture_buffer_mem_map_mem_mut[pixel_coord] = [
                            image.pixels[pixel_coord * 3],
                            image.pixels[pixel_coord * 3 + 1],
                            image.pixels[pixel_coord * 3 + 2],
                            u8::MAX,
                        ];
                    }
                    gltf::image::Format::R8G8B8A8 => {
                        texture_buffer_mem_map_mem_mut[pixel_coord] = [
                            image.pixels[pixel_coord * 4],
                            image.pixels[pixel_coord * 4 + 1],
                            image.pixels[pixel_coord * 4 + 2],
                            image.pixels[pixel_coord * 4 + 3],
                        ];
                    }
                    _ => panic!("Image format not supported yet: {:?}", image.format),
                }
            }
            texture_buffer_mem_map.unmap();
            copy_pass.upload_to_gpu_texture(texture_transfer_info, texture_region, true);

            ImageData {
                texture: texture,
                sampler: sampler,
            }
        })
        .collect();

    // End the copy pass
    gpu_device.end_copy_pass(copy_pass);
    copy_command_buffer.submit().unwrap();

    // Return a struct with the primitives, materials, and original gltf document
    ModelData {
        meshes: meshes_vec,
        images: images_vec,
        document: model_gltf,
    }
}

pub fn main() {
    // Initialize SDL3
    let sdl_context = sdl3::init().unwrap();

    // Get video subsystem
    let video_subsystem = sdl_context.video().unwrap();

    // Create a window
    let window = video_subsystem
        .window("Rust SDL3 GPU - GLTF Model Viewer", 512, 512)
        .resizable()
        .position_centered()
        .build()
        .unwrap();

    // Create the gpu device
    let gpu_device = Device::new(ShaderFormat::SPIRV, true)
        .unwrap()
        .with_window(&window)
        .unwrap();

    // Load the vertex shader code
    let vertex_shader_code = include_bytes!("../shaders/vertex.spv");

    // Create the vertex shader
    let vertex_shader = gpu_device
        .create_shader()
        .with_code(
            ShaderFormat::SPIRV,
            vertex_shader_code,
            sdl3::gpu::ShaderStage::Vertex,
        )
        .with_entrypoint(c"main")
        .with_samplers(0)
        .with_storage_buffers(0)
        .with_storage_textures(0)
        .with_uniform_buffers(1)
        .build()
        .unwrap();

    // Don't need to free the vertex shader file because it is included as bytes (?)

    // Load the fragment shader code
    let fragment_shader_code = include_bytes!("../shaders/fragment.spv");

    // Create the fragment shader
    let fragment_shader = gpu_device
        .create_shader()
        .with_code(
            ShaderFormat::SPIRV,
            fragment_shader_code,
            sdl3::gpu::ShaderStage::Fragment,
        )
        .with_entrypoint(c"main")
        .with_samplers(1)
        .with_storage_buffers(0)
        .with_uniform_buffers(0)
        .build()
        .unwrap();

    // Don't need to free the fragment shader file because it is included as bytes (?)

    // Describe the vertex buffers (for the vertex input state, which is for the pipeline)
    let vertex_buffer_description = VertexBufferDescription::new()
        .with_slot(0)
        .with_input_rate(sdl3::gpu::VertexInputRate::Vertex)
        .with_instance_step_rate(0)
        .with_pitch(size_of::<Vertex>() as u32);

    // Vertex attribute for position. a_position  (for the vertex input state, which is for the pipeline)
    let vertex_attribute1 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(0)
        .with_format(sdl3::gpu::VertexElementFormat::Float3)
        .with_offset(0);

    // Vertex attribute for color. a_color (for the vertex input state, which is for the pipeline)
    let vertex_attribute2 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(1)
        .with_format(sdl3::gpu::VertexElementFormat::Float4)
        .with_offset(size_of::<f32>() as u32 * 3); //offset 3 f32's over to pick (xyz)

    // Vertex attribute for texture coordinate. a_tex_coord (for the vertex input state, which is for the pipeline)
    let vertex_attribute3 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(2)
        .with_format(sdl3::gpu::VertexElementFormat::Float2)
        // .with_offset(0)
        .with_offset(size_of::<f32>() as u32 * 7); // offset 7 f32's over (xyz, rgba)

    // Vertex input state (for the pipeline)
    let vertex_input_state = sdl3::gpu::VertexInputState::new()
        .with_vertex_buffer_descriptions(&[vertex_buffer_description])
        .with_vertex_attributes(&[vertex_attribute1, vertex_attribute2, vertex_attribute3]);

    // Describe the color target (for the target info, which is for the pipeline)
    let color_target_description =
        ColorTargetDescription::new().with_format(gpu_device.get_swapchain_texture_format(&window));
    // Skipping blend state for now, can add that later

    let depth_stencil_texture_create_info = TextureCreateInfo::new()
        .with_type(TextureType::_2D)
        .with_format(TextureFormat::D16Unorm)
        .with_usage(TextureUsage::SAMPLER | TextureUsage::DEPTH_STENCIL_TARGET)
        .with_width(512)
        .with_height(512)
        .with_layer_count_or_depth(1)
        .with_num_levels(1)
        .with_sample_count(SampleCount::NoMultiSampling);

    let mut depth_stencil_texture = gpu_device
        .create_texture(depth_stencil_texture_create_info)
        .unwrap();

    // Set up depth stencil
    let depth_stencil_target_info = DepthStencilTargetInfo::new()
        .with_texture(&mut depth_stencil_texture)
        // .with_clear_depth(1.0)
        .with_clear_depth(0.0)
        .with_load_op(LoadOp::CLEAR)
        .with_store_op(StoreOp::STORE)
        .with_stencil_load_op(LoadOp::CLEAR)
        .with_stencil_store_op(StoreOp::STORE)
        .with_cycle(true)
        // .with_clear_stencil(0);
        .with_clear_stencil(1);

    let depth_stencil_state = DepthStencilState::new()
        // .with_compare_op(CompareOp::LessOrEqual)
        .with_compare_op(CompareOp::Greater)
        // .with_back_stencil_state(value)
        // .with_front_stencil_state(value)
        // .with_compare_mask(value)
        // .with_write_mask(value)
        .with_enable_depth_test(true)
        .with_enable_depth_write(true);
    // .with_enable_stencil_test(value)

    // Make the target info
    let target_info = GraphicsPipelineTargetInfo::new()
        .with_color_target_descriptions(&[color_target_description])
        // .with_has_depth_stencil_target(false);
        .with_has_depth_stencil_target(true)
        .with_depth_stencil_format(TextureFormat::D16Unorm);

    // Create the graphics pipeline
    let pipeline = gpu_device
        .create_graphics_pipeline()
        .with_vertex_shader(&vertex_shader)
        .with_fragment_shader(&fragment_shader)
        .with_primitive_type(sdl3::gpu::PrimitiveType::TriangleList)
        .with_vertex_input_state(vertex_input_state)
        .with_target_info(target_info)
        .with_depth_stencil_state(depth_stencil_state)
        .build()
        .unwrap();

    // Shaders should be released at this point
    drop(vertex_shader);
    drop(fragment_shader);

    let mut loaded_models = Vec::new();
    loaded_models.push(load_model_and_copy_to_gpu(
        "models/Low-Poly-Base_copy.glb",
        &gpu_device,
    ));
    // loaded_models.push(load_model_and_copy_to_gpu( "models/Avocado.glb", &gpu_device, ));
    // loaded_models.push(load_model_and_copy_to_gpu("models/MinimalTriangle.gltf", &gpu_device));
    // loaded_models.push(load_model_and_copy_to_gpu("models/ABeautifulGame.glb", &gpu_device));
    // loaded_models.push(load_model_and_copy_to_gpu( "models/GlassHurricaneCandleHolder.glb", &gpu_device, ));

    // Create the position uniform
    let mut position_uniform = UniformBufferPosition {
        x_pos: 0.0,
        y_pos: 0.0,
    };

    // Done with gpu init

    // Initialize game variables
    let mut player_x = 0.0;
    let mut player_y = 0.0;
    // let mut game_ticks = 0u32;

    // Keyboard state variables
    let mut key_up = false;
    let mut key_down = false;
    let mut key_left = false;
    let mut key_right = false;

    // Event handling
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        // game_ticks += 1;
        for event in event_pump.poll_iter() {
            match event {
                // Esc - Quit button
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                // Up, down, left, right PRESSED events
                Event::KeyDown {
                    keycode: Some(Keycode::Up),
                    ..
                } => {
                    key_up = true;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Down),
                    ..
                } => {
                    key_down = true;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Left),
                    ..
                } => {
                    key_left = true;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Right),
                    ..
                } => {
                    key_right = true;
                }
                // Up, down, left, right RELEASED events
                Event::KeyUp {
                    keycode: Some(Keycode::Up),
                    ..
                } => {
                    key_up = false;
                }
                Event::KeyUp {
                    keycode: Some(Keycode::Down),
                    ..
                } => {
                    key_down = false;
                }
                Event::KeyUp {
                    keycode: Some(Keycode::Left),
                    ..
                } => {
                    key_left = false;
                }
                Event::KeyUp {
                    keycode: Some(Keycode::Right),
                    ..
                } => {
                    key_right = false;
                }
                // Anything else
                _ => {}
            }
        }
        // Move player position
        if key_up {
            player_y += 0.02;
        }
        if key_down {
            player_y -= 0.02;
        }
        if key_left {
            player_x -= 0.02;
        }
        if key_right {
            player_x += 0.02;
        }

        // End of game code

        // Acquire the command buffer
        let mut command_buffer = gpu_device.acquire_command_buffer().unwrap();

        // Update the position value in the position uniform
        position_uniform.x_pos = player_x;
        position_uniform.y_pos = player_y;

        // Push the position uniform data
        command_buffer.push_vertex_uniform_data(0, &position_uniform);

        // Get the swapchain texture
        let swapchain_texture = command_buffer
            .wait_and_acquire_swapchain_texture(&window)
            .unwrap();
        // Should check if swapchain buffer is actually available, and submit command buffer early if not available?

        // Create the color target
        let color_target_info = ColorTargetInfo::default()
            .with_clear_color(Color {
                r: 240u8,
                g: 250u8,
                b: 255u8,
                a: 255u8,
            })
            .with_load_op(SDL_GPU_LOADOP_CLEAR)
            .with_store_op(SDL_GPU_STOREOP_STORE)
            .with_texture(&swapchain_texture);

        // Begin a render pass
        let render_pass = gpu_device
            .begin_render_pass(
                &command_buffer,
                &[color_target_info],
                Some(&depth_stencil_target_info),
            )
            .unwrap();

        // Bind the graphics pipeline
        render_pass.bind_graphics_pipeline(&pipeline);

        // Loop through all loaded models
        // (Later - render the current state of the game or simulation, likely rendering multiples of each model)
        for model in &loaded_models {
            // Is there always at least 1 scene? I think so but not sure
            let scene = model
                .document
                .default_scene()
                .or(model.document.scenes().next())
                .unwrap();

            let mut remaining_node_transform_pairs = Vec::new();

            for node in scene.nodes() {
                remaining_node_transform_pairs.push((node, ()));
            }

            while !remaining_node_transform_pairs.is_empty() {
                // Take a node off the list
                let (node, transform) = remaining_node_transform_pairs.pop().unwrap();
                if let Some(mesh) = node.mesh() {
                    // Render mesh with transform

                    let mesh_data = model.meshes.get(mesh.index()).unwrap();
                    for primitive in mesh.primitives() {
                        // Look up the primitive data
                        let primitive_data = mesh_data.primitives.get(primitive.index()).unwrap();

                        // Setup the buffer bindings for vertices
                        let buffer_bindings_vertex = BufferBinding::new()
                            .with_buffer(&primitive_data.vertex_buffer)
                            .with_offset(0);

                        // Bind the vertex buffer
                        render_pass.bind_vertex_buffers(0, &[buffer_bindings_vertex]);

                        // Setup the buffer bindings for indices
                        let buffer_bindings_index = BufferBinding::new()
                            .with_buffer(&primitive_data.index_buffer)
                            .with_offset(0);

                        // Bind the index buffer
                        render_pass
                            .bind_index_buffer(&buffer_bindings_index, IndexElementSize::_32BIT);

                        // Determine the texture(s) to use
                        let image_index = primitive
                            .material()
                            .pbr_metallic_roughness()
                            .base_color_texture()
                            .unwrap()
                            .texture()
                            .source()
                            .index();

                        let image_data = model.images.get(image_index).unwrap();

                        // Bind sampler? (Not sure)
                        let texture_sampler_binding = TextureSamplerBinding::new()
                            .with_texture(&image_data.texture)
                            .with_sampler(&image_data.sampler);
                        render_pass.bind_fragment_samplers(0, &[texture_sampler_binding]);

                        // Issue the draw call using the indexes as well as the
                        render_pass.draw_indexed_primitives(primitive_data.index_count, 1, 0, 0, 0);
                    }

                    // End of rendering a mesh
                }
                for child_node in node.children() {
                    // Add child nodes to the list
                    remaining_node_transform_pairs.push((child_node, ()));
                }
            }
        }

        // End the render pass
        gpu_device.end_render_pass(render_pass);

        // Submit the command buffer
        command_buffer.submit().unwrap();

        // Wait for next frame
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
    println!("Exited loop");

    // Release buffers and pipeline, destroy gpu device and window here
    // drop(pipeline);
    // drop(window);
    // drop(gpu_device);
}
