// Uncomment the next line to not show the console window
// #![windows_subsystem = "windows"]

extern crate sdl3;

use gltf::image::Data;
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

// Uniform value to pass to shader
// struct UniformBufferTime {
//     time: c_float,
// }
struct UniformBufferPosition {
    x_pos: c_float,
    y_pos: c_float,
}

#[derive(Debug)]
struct Primitive {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

#[derive(Debug)]
struct Mesh {
    primitive_vec: Vec<Primitive>,
}

#[derive(Debug)]
struct Model {
    mesh_vec: Vec<Mesh>,
    images_vec: Vec<Data>,
}

struct PrimitiveBufferStruct {
    vertex_buffer: Buffer,
    #[allow(dead_code)] // Not used yet - may be unnecessary to keep track of this?
    vertex_count: u32,
    index_buffer: Buffer,
    index_count: u32,
}

fn load_model() -> Model {
    // Open model file
    let (model_gltf, buffers, images) = gltf::import("models/Low-Poly-Base_copy.glb").unwrap();
    for image in images.iter() {
        println!(
            "image here with format {:?} width {} height {}",
            image.format, image.width, image.height
        );
    }

    let textures = model_gltf.textures();
    for texture in textures.clone() {
        println!("Texture - index is: {:?}", texture.index());
        // texture.sampler();
        // let texture_sampler = texture.sampler();
        // dbg!(texture_sampler.mag_filter());
        // let texture_source = texture.source();
        // dbg!(texture_source);
    }
    // let texture = textures
    //     .clone()
    //     .find(|texture| -> bool { texture.index() == 0 });

    /* for image in images {
        println!("image - {:?}", image.name);
    } */

    let mut mesh_vec: Vec<Mesh> = Vec::new();

    // Loop through meshes in the model
    for mesh in model_gltf.meshes() {
        let mut primitive_vec: Vec<Primitive> = Vec::new();

        // Loop through primitives in each mesh
        for primitive in mesh.primitives() {
            // Create reader
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let mut tex_coord_temp_vector = Vec::new();
            // TODO - figure out what the "set" argument for read_tex_coords(set) is for
            if let Some(iter) = reader.read_tex_coords(0) {
                for tex_coord in iter.into_f32() {
                    // tex_coord
                    let tex_coord_u = tex_coord[0];
                    let tex_coord_v = tex_coord[1];
                    // println!("u: {}, v: {}", tex_coord_u, tex_coord_v);
                    tex_coord_temp_vector.push((tex_coord_u, tex_coord_v));
                }
            }

            // Create vertices vector, then read vertex positions and add to vector
            let mut vertices = Vec::new();
            if let Some(iter) = reader.read_positions() {
                for (index, vertex_position) in iter.enumerate() {
                    vertices.push(Vertex {
                        x: vertex_position[0],
                        y: vertex_position[1],
                        z: vertex_position[2],
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                        // u: 0.5,
                        // v: 0.5,
                        u: tex_coord_temp_vector[index].0,
                        v: tex_coord_temp_vector[index].1,
                    });
                }
            }

            // Create indices vector, then read index values and put into vector
            let mut indices = Vec::new();
            if let Some(indices_raw) = reader.read_indices() {
                indices.append(&mut indices_raw.into_u32().collect::<Vec<u32>>());
            }

            let material = primitive.material();
            // println!("material - index is: {:?}", material.index());
            let texture = textures
                .clone()
                .find(|texture| -> bool { texture.index() == material.index().unwrap() })
                .unwrap();

            let image = texture.source();
            let image_source = image.source();
            // image_view should be the "view" of the buffer of the image data, I think...
            let image_view = match image_source {
                gltf::image::Source::View { view, mime_type: _ } => {
                    // println!("is view, {}", mime_type);
                    // println!("view index {}", view.index());
                    Some(view)
                }
                gltf::image::Source::Uri {
                    uri: _,
                    mime_type: _,
                } => {
                    // println!("is uri, {}", mime_type.unwrap());
                    None
                }
            }
            .unwrap();
            println!(
                "image_view has index {}, length {}, offset {}, stride {}, name {}, extras {}",
                image_view.index(),
                image_view.length(),
                image_view.offset(),
                image_view.stride().unwrap_or(0),
                image_view.name().unwrap_or("(no name)"),
                image_view.extras().to_string()
            );

            // Get the buffer of that view (not sure? Might actually just want the view)
            let _image_buffer = image_view.buffer();
            // println!(
            //     "this image_buffer has index {}, length {}, name {}, extras {}",
            //     image_buffer.index(),
            //     image_buffer.length(),
            //     image_buffer.name().unwrap_or("(no name)"),
            //     image_buffer.extras().to_string()
            // );

            // println!("{:?}", image_source);

            /* reader.read_normals();
            reader.read_tangents();
            reader.read_colors(0);
            reader.read_joints(0);
            reader.read_tex_coords(0);
            reader.read_weights(0);
            reader.read_morph_targets(); */

            // Print info (can be removed at some point)
            // println!(
            //     "Primitive added with {} vertices and {} indices. Size of the indices array in bytes is {}",
            //     vertices.len(),
            //     indices.len(),
            //     indices.len() * size_of::<u32>()
            // );

            // Make a primitive struct instance
            primitive_vec.push(Primitive {
                vertices: vertices,
                indices: indices,
            });
        }
        mesh_vec.push(Mesh {
            primitive_vec: primitive_vec,
        })
    }
    Model {
        mesh_vec: mesh_vec,
        images_vec: images,
    }
}

pub fn main() {
    let mut model_vec: Vec<Model> = Vec::new();

    model_vec.push(load_model());

    let number_of_samplers: u32 = model_vec
        .iter()
        .map(|model| -> u32 { model.images_vec.len() as u32 })
        .sum();
    println!("there are going to be {} samplers", number_of_samplers);

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

    // TODO - working on texture. Still in the "what is going on" phase
    // Not sure which format, need to get from gltf data (I think?)
    // Also the other info of course
    // let my_texture_create_info = TextureCreateInfo::new()
    //     .with_type(TextureType::_2D)
    //     .with_format(TextureFormat::R8g8b8a8Unorm)
    //     // TODO - is this one supposed to be R8g8b8a8UnormSrgb ???
    //     .with_usage(TextureUsage::SAMPLER)
    //     .with_width(64)
    //     .with_height(64)
    //     .with_layer_count_or_depth(1)
    //     .with_num_levels(1);
    // let my_texture = gpu_device.create_texture(my_texture_create_info).unwrap();

    // TODO - sampler?
    // let my_sampler_create_info = SamplerCreateInfo::new()
    //     .with_min_filter(Filter::Nearest)
    //     .with_mag_filter(Filter::Nearest)
    //     .with_mipmap_mode(SamplerMipmapMode::Nearest)
    //     .with_address_mode_u(SamplerAddressMode::Repeat)
    //     .with_address_mode_v(SamplerAddressMode::Repeat)
    //     .with_address_mode_w(SamplerAddressMode::Repeat);
    // // .with_mip_lod_bias(value)
    // // .with_max_anisotropy(value)
    // // .with_compare_op(value)
    // // .with_min_lod(value)
    // // .with_max_lod(value)
    // // .with_enable_anisotropy(enable)
    // // .with_enable_compare(enable);
    // let my_sampler = gpu_device.create_sampler(my_sampler_create_info).unwrap();

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
        // .with_samplers(number_of_samplers)
        .with_storage_buffers(0)
        // .with_uniform_buffers(1)
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
        .with_offset(size_of::<f32>() as u32 * 3); //offset 3 f32's over to pick rgba out of xyzrgba vertex

    // Vertex attribute for texture coordinate. a_tex_coord (for the vertex input state, which is for the pipeline)
    let vertex_attribute3 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(2)
        .with_format(sdl3::gpu::VertexElementFormat::Float2)
        // .with_offset(0)
        .with_offset(size_of::<f32>() as u32 * 7); // offset 7 f32's over? (Not sure, but guessing it needs to be after xyzrgba?)

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
        .with_clear_depth(1.0)
        .with_load_op(LoadOp::CLEAR)
        .with_store_op(StoreOp::STORE)
        .with_stencil_load_op(LoadOp::CLEAR)
        .with_stencil_store_op(StoreOp::STORE)
        .with_cycle(true)
        .with_clear_stencil(0);

    let depth_stencil_state = DepthStencilState::new()
        .with_compare_op(CompareOp::LessOrEqual)
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

    // Keep a reference to the vertex buffers
    let mut primitive_buffer_struct_vec: Vec<PrimitiveBufferStruct> = Vec::new();

    //
    let mut samplers_vec: Vec<Sampler> = Vec::new();

    //
    let mut textures_vec: Vec<Texture> = Vec::new();

    // Start a copy pass
    let copy_command_buffer = gpu_device.acquire_command_buffer().unwrap();
    let copy_pass = gpu_device.begin_copy_pass(&copy_command_buffer).unwrap();

    // Transfer the texture data
    // let texture_transfer_buffer = gpu_device
    //     .create_transfer_buffer()
    //     .with_size((64 * 64 * size_of::<c_float>() * 4) as u32) // guessing at the size of the texture data right now
    //     .with_usage(TransferBufferUsage::UPLOAD)
    //     .build()
    //     .unwrap();
    // let texture_transfer_info = TextureTransferInfo::new()
    //     .with_transfer_buffer(&texture_transfer_buffer)
    //     .with_offset(0)
    //     .with_pixels_per_row(64)
    //     .with_rows_per_layer(64);
    // let texture_region = TextureRegion::new()
    //     .with_texture(&my_texture)
    //     .with_width(64)
    //     .with_height(64)
    //     .with_depth(1);
    // .with_mip_level(mip_level)
    // .with_layer(0)
    // .with_x(0)
    // .with_y(0)
    // .with_z(0)

    // let mut texture_color_rotate: u8 = 0;

    // TODO - Need to fill the transfer buffer first
    // (See below for how it was done with vertex and index data. Probably similar-ish)
    // let mut texture_buffer_mem_map = texture_transfer_buffer.map(&gpu_device, true);
    // let texture_buffer_mem_map_mem_mut: &mut [_] = texture_buffer_mem_map.mem_mut();
    // for pixel_coord in 0..(64 * 64) {
    // Using a fixed color for every pixel, should instead be from the gltf texture image
    // if texture_color_rotate == 0 {
    //     texture_buffer_mem_map_mem_mut[pixel_coord] = [255u8, 0u8, 0u8, 255u8];
    // }
    // if texture_color_rotate == 1 {
    //     texture_buffer_mem_map_mem_mut[pixel_coord] = [0u8, 255u8, 0u8, 255u8];
    // }
    // if texture_color_rotate == 2 {
    //     texture_buffer_mem_map_mem_mut[pixel_coord] = [0u8, 0u8, 255u8, 255u8];
    // }
    // if texture_color_rotate == 3 {
    //     texture_buffer_mem_map_mem_mut[pixel_coord] = [255u8, 255u8, 0u8, 255u8];
    // }
    // if texture_color_rotate == 4 {
    //     texture_buffer_mem_map_mem_mut[pixel_coord] = [255u8, 0u8, 255u8, 255u8];
    // }
    // if texture_color_rotate == 5 {
    //     texture_buffer_mem_map_mem_mut[pixel_coord] = [0u8, 255u8, 255u8, 255u8];
    // }
    // if texture_color_rotate == 6 {
    //     texture_buffer_mem_map_mem_mut[pixel_coord] = [180u8, 50u8, 90u8, 255u8];
    // }
    // texture_color_rotate += 1;
    // if texture_color_rotate == 7 {
    //     texture_color_rotate = 0;
    // }
    // }
    // texture_buffer_mem_map.unmap();

    // Upload the data
    // copy_pass.upload_to_gpu_texture(texture_transfer_info, texture_region, true);

    // Iterate through primitives in meshes in models
    for model in model_vec {
        for mesh in model.mesh_vec {
            for primitive in mesh.primitive_vec {
                // Vertex stuff

                // How big is the vertex data to transfer
                let primitive_vertices_size =
                    (primitive.vertices.len() * size_of::<Vertex>()) as u32;

                // How big is the index data to transfer
                let primitive_indices_size = (primitive.indices.len() * size_of::<u32>()) as u32;

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

                for (index, &value) in primitive.vertices.iter().enumerate() {
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

                // Create the vertex buffer
                let index_buffer = gpu_device
                    .create_buffer()
                    .with_size(primitive_indices_size)
                    .with_usage(BufferUsageFlags::INDEX)
                    .build()
                    .unwrap();

                // Fill the transfer buffer
                let mut buffer_mem_map = transfer_buffer.map(&gpu_device, true);

                let buffer_mem_map_mem_mut: &mut [u32] = buffer_mem_map.mem_mut();

                for (index, &value) in primitive.indices.iter().enumerate() {
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
                    // .with_size(primitive.indices.len() as u32)
                    .with_offset(0u32);

                // Upload the data
                // The next line causes the model to not show up...
                copy_pass.upload_to_gpu_buffer(data_location, buffer_region, true);

                // Release the index transfer buffer
                drop(transfer_buffer);

                // Keep track of buffers
                primitive_buffer_struct_vec.push(PrimitiveBufferStruct {
                    vertex_buffer: vertex_buffer,
                    vertex_count: primitive.vertices.len() as u32,
                    index_buffer: index_buffer,
                    index_count: primitive.indices.len() as u32,
                });
            }
        }
        for image in model.images_vec {
            let my_texture_create_info = TextureCreateInfo::new()
                .with_type(TextureType::_2D)
                .with_format(TextureFormat::R8g8b8a8Unorm)
                // TODO - is this one supposed to be R8g8b8a8UnormSrgb ???
                .with_usage(TextureUsage::SAMPLER)
                .with_width(image.width)
                .with_height(image.height)
                .with_layer_count_or_depth(1)
                .with_num_levels(1);
            let my_texture = gpu_device.create_texture(my_texture_create_info).unwrap();

            let my_sampler_create_info = SamplerCreateInfo::new()
                // TODO - get sampler info from gltf model
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
            let my_sampler = gpu_device.create_sampler(my_sampler_create_info).unwrap();
            samplers_vec.push(my_sampler);
            println!("added a sampler, now there are {}", samplers_vec.len());

            let texture_transfer_buffer = gpu_device
                .create_transfer_buffer()
                .with_size(image.width * image.height * (size_of::<c_float>() as u32) * 4) // guessing at the size of the texture data right now
                .with_usage(TransferBufferUsage::UPLOAD)
                .build()
                .unwrap();
            let texture_transfer_info = TextureTransferInfo::new()
                .with_transfer_buffer(&texture_transfer_buffer)
                .with_offset(0)
                .with_pixels_per_row(image.width)
                .with_rows_per_layer(image.height);
            let texture_region = TextureRegion::new()
                .with_texture(&my_texture)
                .with_width(image.width)
                .with_height(image.height)
                .with_depth(1);
            textures_vec.push(my_texture);

            let mut texture_buffer_mem_map = texture_transfer_buffer.map(&gpu_device, true);
            let texture_buffer_mem_map_mem_mut: &mut [_] = texture_buffer_mem_map.mem_mut();
            for pixel_coord in 0..(image.width * image.height) as usize {
                texture_buffer_mem_map_mem_mut[pixel_coord] = [
                    image.pixels[pixel_coord * 4],
                    image.pixels[pixel_coord * 4 + 1],
                    image.pixels[pixel_coord * 4 + 2],
                    image.pixels[pixel_coord * 4 + 3],
                ];
            }
            texture_buffer_mem_map.unmap();
            copy_pass.upload_to_gpu_texture(texture_transfer_info, texture_region, true);
        }
    }

    // End the copy pass
    gpu_device.end_copy_pass(copy_pass);
    copy_command_buffer.submit().unwrap();

    // Create the time uniform
    // let mut time_uniform = UniformBufferTime { time: 0.0 };

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

        // Update the time value in the time uniform
        // time_uniform.time = game_ticks as f32 * 0.1;

        // Push the time uniform data
        // command_buffer.push_fragment_uniform_data(0, &time_uniform);

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

        // Loop through all vertex buffers
        for (index, primitive_buffer_struct) in primitive_buffer_struct_vec.iter().enumerate() {
            // Setup the buffer bindings for vertices
            let buffer_bindings_vertex = BufferBinding::new()
                .with_buffer(&primitive_buffer_struct.vertex_buffer)
                .with_offset(0);

            // Bind the vertex buffer
            render_pass.bind_vertex_buffers(0, &[buffer_bindings_vertex]);

            // Setup the buffer bindings for indices
            let buffer_bindings_index = BufferBinding::new()
                .with_buffer(&primitive_buffer_struct.index_buffer)
                .with_offset(0);

            // Bind the index buffer
            render_pass.bind_index_buffer(&buffer_bindings_index, IndexElementSize::_32BIT);

            // Bind the texture buffer? (Not sure)
            // I feel like it shouldn't need to .clone() the texture
            // render_pass.bind_fragment_storage_textures(0, &[my_texture.clone()]);

            // TODO - right now I am manually picking which texture for which primitive, need to figure that out in code
            let tex_to_use = match index {
                0 => 0,
                1 => 1,
                2 => 1,
                3 => 2,
                4 => 3,
                5 => 4,
                6 => 5,
                _ => 0,
            };

            // Bind sampler? (Not sure)
            let texture_sampler_binding = TextureSamplerBinding::new()
                .with_texture(&textures_vec[tex_to_use])
                .with_sampler(&samplers_vec[tex_to_use]);
            render_pass.bind_fragment_samplers(0, &[texture_sampler_binding]);

            // Issue the draw call using the indexes as well as the
            render_pass.draw_indexed_primitives(primitive_buffer_struct.index_count, 1, 0, 0, 0);
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
