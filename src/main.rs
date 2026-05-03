// Uncomment the next line to not show the console window
// #![windows_subsystem = "windows"]

extern crate sdl3;

use gltf::Document;
use gltf::animation::util::ReadOutputs;
use imgui_sdl3::ImGuiSdl3;
use sdl3::event::Event;
use sdl3::gpu::*;
use sdl3::keyboard::Keycode;
use sdl3::libc::c_float;
use sdl3::libc::c_uint;
use sdl3::mouse::MouseButton;
use sdl3::pixels::Color;
use sdl3::sys::gpu::*;
use std::time::Duration;

// The vertex input layout
#[allow(dead_code)] // Compiler doesn't see that the code is used to pass info to the gpu
#[derive(Copy, Clone)]
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
    // joints
    j1: c_uint,
    j2: c_uint,
    j3: c_uint,
    j4: c_uint,
    // weights
    w1: c_float,
    w2: c_float,
    w3: c_float,
    w4: c_float,
    // morph targets
    m1: [c_float; 3],
    m2: [c_float; 3],
    m3: [c_float; 3],
    m4: [c_float; 3],
}

struct PrimitiveData {
    vertex_buffer: Buffer,
    // vertex_count: u32,
    index_buffer: Buffer,
    index_count: u32,
    // morph_target_buffer: Option<Buffer>,
    // morph_target_count: u32,
}

struct MeshData {
    primitives: Vec<PrimitiveData>,
}

struct ImageData<'a> {
    texture: Texture<'a>,
    sampler: Sampler,
}

struct AnimationData {
    channels: Vec<AnimationChannelData>,
    animation_index: usize,
}

#[derive(Debug)]
enum AnimationOutput {
    Translation([f32; 3]),
    Rotation([f32; 4]),
    Scale([f32; 3]),
    MorphTargetWeight(Vec<f32>),
}

struct AnimationChannelData {
    // target_property: Property,
    target_node_index: usize,
    // interpolation: Interpolation,
    inputs: Vec<f32>,
    outputs: Vec<AnimationOutput>,
    channel_index: usize,
}

struct ModelData<'a> {
    meshes: Vec<MeshData>,
    images: Vec<ImageData<'a>>,
    document: Document,
    animations: Vec<AnimationData>,
}

#[allow(dead_code)] // Compiler doesn't see that the code is used to pass info to the gpu
struct VertexUniformBuffer {
    transform_matrix: [[c_float; 4]; 4],
    morph_weights: [c_float; 4],
    // morph_target_count: c_uint,
    // vertex_count: c_uint,
    joint_matrices: Vec<[[c_float; 4]; 4]>,
}

fn load_model_and_copy_to_gpu<'a>(model_path: &str, gpu_device: &Device) -> ModelData<'a> {
    // Load the model with the gltf crate's import function
    let (document, buffers, images) = gltf::import(model_path).unwrap();

    // Start a copy pass
    let copy_command_buffer = gpu_device.acquire_command_buffer().unwrap();
    let copy_pass = gpu_device.begin_copy_pass(&copy_command_buffer).unwrap();

    for skin in document.skins() {
        // println!("skin name: {}", skin.name().unwrap_or("no name"));
        let skin_reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(inverse_bind_matrices) = skin_reader.read_inverse_bind_matrices() {
            println!(
                "Skin has {} inverse bind matrices",
                inverse_bind_matrices.len()
            );
        }
        if let Some(skeleton_node) = skin.skeleton() {
            println!("Skeleton node is index {}", skeleton_node.index());
        } else {
            println!("Skeleton is root node");
        }
    }

    let meshes_vec = document
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
                    // Currently unused:
                    // reader.read_normals()
                    // reader.read_tangents()
                    let mut joints_temp_vec = Vec::new();
                    if let Some(joints_reader) = reader.read_joints(0) {
                        for joint in joints_reader.into_u16() {
                            // println!("j: {:?}", joint);
                            let joint_u32 = [
                                u32::from(joint[0]),
                                u32::from(joint[1]),
                                u32::from(joint[2]),
                                u32::from(joint[3]),
                            ];
                            joints_temp_vec.push(joint_u32);
                        }
                    }
                    let mut weights_temp_vec = Vec::new();
                    if let Some(weights_reader) = reader.read_weights(0) {
                        for weight in weights_reader.into_f32() {
                            // println!("w: {:?}", weight);
                            weights_temp_vec.push(weight);
                        }
                    }

                    // let mut morph_positions_vector = Vec::new();
                    // let mut morph_positions_count: u32 = 0;

                    let mut morph_targets_temp_vec = Vec::new();
                    for morph_target in reader.read_morph_targets() {
                        // morph_target is (positions, normals, tangents) but each is optional
                        if let (Some(morph_positions), _normals, _tangents) = morph_target {
                            // Only looking at positions for now
                            // morph_positions_count += 1;

                            let mut morph_position_temp_vec = Vec::new();
                            for morph_position in morph_positions {
                                // morph_positions_vector.push(morph_position);
                                morph_position_temp_vec.push(morph_position);
                            }
                            morph_targets_temp_vec.push(morph_position_temp_vec);
                        }
                    }

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
                            // Use morph positions from model data or use default value (0, 0, 0)
                            let morph_1 = {
                                if let Some(morph_target_1) = morph_targets_temp_vec.get(0) {
                                    morph_target_1.get(index).unwrap_or(&[0f32, 0f32, 0f32])
                                } else {
                                    &[0f32, 0f32, 0f32]
                                }
                            };
                            let morph_2 = {
                                if let Some(morph_target_1) = morph_targets_temp_vec.get(1) {
                                    morph_target_1.get(index).unwrap_or(&[0f32, 0f32, 0f32])
                                } else {
                                    &[0f32, 0f32, 0f32]
                                }
                            };
                            let morph_3 = {
                                if let Some(morph_target_1) = morph_targets_temp_vec.get(2) {
                                    morph_target_1.get(index).unwrap_or(&[0f32, 0f32, 0f32])
                                } else {
                                    &[0f32, 0f32, 0f32]
                                }
                            };
                            let morph_4 = {
                                if let Some(morph_target_1) = morph_targets_temp_vec.get(3) {
                                    morph_target_1.get(index).unwrap_or(&[0f32, 0f32, 0f32])
                                } else {
                                    &[0f32, 0f32, 0f32]
                                }
                            };
                            let joint = joints_temp_vec
                                .get(index)
                                .unwrap_or(&[0u32, 0u32, 0u32, 0u32]);
                            let weight =
                                weights_temp_vec.get(index).unwrap_or(&[0.0, 0.0, 0.0, 0.0]);
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
                                j1: joint[0],
                                j2: joint[1],
                                j3: joint[2],
                                j4: joint[3],
                                w1: weight[0],
                                w2: weight[1],
                                w3: weight[2],
                                w4: weight[3],
                                m1: *morph_1,
                                m2: *morph_2,
                                m3: *morph_3,
                                m4: *morph_4,
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

                    // How big is the morph target data to transfer
                    // let primitive_morph_target_size =
                    //     (morph_positions_vector.len() * size_of::<[f32; 4]>()) as u32;

                    // Determine which is larger
                    let larger_size = primitive_vertices_size.max(primitive_indices_size);
                    // .max(primitive_morph_target_size);

                    // Create the transfer buffer, the vertices and the indices (and optionally the morph target) will both use this to upload to the gpu
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

                    // Fill the transfer buffer with the vertex data
                    let mut buffer_mem_map = transfer_buffer.map(&gpu_device, true);
                    let buffer_mem_map_mem_mut: &mut [Vertex] = buffer_mem_map.mem_mut();
                    for (index, &value) in vertices.iter().enumerate() {
                        buffer_mem_map_mem_mut[index] = value;
                    }
                    buffer_mem_map.unmap();

                    // Set the location of the data (it's at the start of the transfer buffer)
                    let data_location = TransferBufferLocation::default()
                        .with_transfer_buffer(&transfer_buffer)
                        .with_offset(0u32);

                    // Set what region of the buffer to transfer (the size of the vertices data)
                    let buffer_region = BufferRegion::default()
                        .with_buffer(&vertex_buffer)
                        .with_size(primitive_vertices_size)
                        .with_offset(0u32);

                    // Upload the data
                    copy_pass.upload_to_gpu_buffer(data_location, buffer_region, true);

                    // Create the index buffer
                    let index_buffer = gpu_device
                        .create_buffer()
                        .with_size(primitive_indices_size)
                        .with_usage(BufferUsageFlags::INDEX)
                        .build()
                        .unwrap();

                    // Fill the transfer buffer with the index data
                    let mut buffer_mem_map = transfer_buffer.map(&gpu_device, true);
                    let buffer_mem_map_mem_mut: &mut [u32] = buffer_mem_map.mem_mut();
                    for (index, &value) in indices.iter().enumerate() {
                        buffer_mem_map_mem_mut[index] = value;
                    }
                    buffer_mem_map.unmap();

                    // Set the location of the data (it's at the start of the transfer buffer)
                    let data_location = TransferBufferLocation::default()
                        .with_transfer_buffer(&transfer_buffer)
                        .with_offset(0u32);

                    // Set what region of the buffer to transfer (the size of the indices data)
                    let buffer_region = BufferRegion::default()
                        .with_buffer(&index_buffer)
                        .with_size(primitive_indices_size)
                        .with_offset(0u32);

                    // Upload the data
                    copy_pass.upload_to_gpu_buffer(data_location, buffer_region, true);

                    // Optionally Create the morph target buffer if there is 1 or more morph targets for the primitive
                    /* let morph_target_buffer: Option<Buffer> = if morph_positions_count > 0 {
                        let morph_target_buffer = gpu_device
                            .create_buffer()
                            .with_size(primitive_morph_target_size)
                            // Not sure on the buffer usage flag? I think "Graphics Storage Read" makes sense...
                            .with_usage(BufferUsageFlags::GRAPHICS_STORAGE_READ)
                            .build()
                            .unwrap();

                        // Fill the transfer buffer with the morph target data
                        let mut buffer_mem_map = transfer_buffer.map(&gpu_device, true);
                        let buffer_mem_map_mem_mut: &mut [[f32; 4]] = buffer_mem_map.mem_mut();
                        for (index, &value) in morph_positions_vector.iter().enumerate() {
                            // Pad with a 0.0 to make it a vec4 instead of vec3
                            // (which avoids issues in the shader)
                            let transfer_value: [f32; 4] = [value[0], value[1], value[2], 0.0];
                            buffer_mem_map_mem_mut[index] = transfer_value;
                        }
                        buffer_mem_map.unmap();

                        // Set the location of the data (it's at the start of the transfer buffer)
                        let data_location = TransferBufferLocation::default()
                            .with_transfer_buffer(&transfer_buffer)
                            .with_offset(0u32);

                        // Set what region of the buffer to transfer (the size of the indices data)
                        let buffer_region = BufferRegion::default()
                            .with_buffer(&morph_target_buffer)
                            .with_size(primitive_morph_target_size)
                            .with_offset(0u32);

                        // Upload the data
                        copy_pass.upload_to_gpu_buffer(data_location, buffer_region, true);

                        Some(morph_target_buffer)
                    } else {
                        None
                    }; */

                    // Release the index transfer buffer
                    drop(transfer_buffer);

                    PrimitiveData {
                        vertex_buffer: vertex_buffer,
                        // vertex_count: vertices.len() as u32,
                        index_buffer: index_buffer,
                        index_count: indices.len() as u32,
                        // morph_target_buffer: morph_target_buffer,
                        // morph_target_count: morph_positions_count,
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
            // Currently unused sampler options
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
                    // Haven't needed to implement the other formats yet. Should be similar to the above?
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

    // Get animation data
    let animations_vec = document
        .animations()
        .map(move |animation| -> AnimationData {
            // Get info about each channel in the animation
            let channels_vec = animation
                .channels()
                .map(|channel| -> AnimationChannelData {
                    // Create the reader
                    let animation_sampler_reader =
                        channel.reader(|buffer| Some(&buffers[buffer.index()]));

                    // Get the input data
                    let animation_inputs: Vec<f32> =
                        animation_sampler_reader.read_inputs().unwrap().collect();
                    // println!("animation inputs length: {}", animation_inputs.len());

                    // Get the output data
                    let animation_outputs: Vec<AnimationOutput> =
                        match animation_sampler_reader.read_outputs().unwrap() {
                            ReadOutputs::Translations(translations) => translations
                                .map(|translation| -> AnimationOutput {
                                    AnimationOutput::Translation(translation)
                                })
                                .collect(),
                            ReadOutputs::Rotations(rotations) => rotations
                                .into_f32()
                                .map(|rotation| -> AnimationOutput {
                                    AnimationOutput::Rotation(rotation)
                                })
                                .collect(),
                            ReadOutputs::Scales(scales) => scales
                                .map(|scale| -> AnimationOutput { AnimationOutput::Scale(scale) })
                                .collect(),
                            ReadOutputs::MorphTargetWeights(weights) => {
                                let weights_f32 = weights.into_f32();
                                let weight_chunk_size = weights_f32.len() / animation_inputs.len();
                                let weights_vec: Vec<f32> = weights_f32.collect();
                                let weights_chunks = weights_vec.chunks(weight_chunk_size);
                                weights_chunks
                                    .map(|chunk| -> AnimationOutput {
                                        let chunk_vec = chunk.to_vec();
                                        AnimationOutput::MorphTargetWeight(chunk_vec)
                                    })
                                    .collect()
                            }
                        };
                    AnimationChannelData {
                        // target_property: channel.target().property(),
                        target_node_index: channel.target().node().index(),
                        // interpolation: channel.sampler().interpolation(),
                        inputs: animation_inputs,
                        outputs: animation_outputs,
                        channel_index: channel.index(),
                    }
                })
                .collect();
            AnimationData {
                channels: channels_vec,
                animation_index: animation.index(),
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
        document,
        animations: animations_vec,
    }
}

// Make a solid white image and upload it to the gpu so that textures without an image can use this instead
// (Not sure if that's the right way to handle that?)
fn create_dummy_image_and_copy_to_gpu<'a>(gpu_device: &Device) -> ImageData<'a> {
    // Start a copy pass
    let copy_command_buffer = gpu_device.acquire_command_buffer().unwrap();
    let copy_pass = gpu_device.begin_copy_pass(&copy_command_buffer).unwrap();

    // how small can it be? 1x1 seems fine
    let img_size = 1u32;
    let texture_create_info = TextureCreateInfo::new()
        .with_type(TextureType::_2D)
        .with_format(TextureFormat::R8g8b8a8Unorm)
        .with_usage(TextureUsage::SAMPLER)
        .with_width(img_size)
        .with_height(img_size)
        .with_layer_count_or_depth(1)
        .with_num_levels(1);
    let texture = gpu_device.create_texture(texture_create_info).unwrap();

    let sampler_create_info = SamplerCreateInfo::new()
        .with_min_filter(Filter::Nearest)
        .with_mag_filter(Filter::Nearest)
        .with_mipmap_mode(SamplerMipmapMode::Nearest)
        .with_address_mode_u(SamplerAddressMode::Repeat)
        .with_address_mode_v(SamplerAddressMode::Repeat)
        .with_address_mode_w(SamplerAddressMode::Repeat);
    let sampler = gpu_device.create_sampler(sampler_create_info).unwrap();

    let texture_transfer_buffer = gpu_device
        .create_transfer_buffer()
        .with_size(img_size * img_size * (size_of::<c_float>() as u32) * 4)
        .with_usage(TransferBufferUsage::UPLOAD)
        .build()
        .unwrap();
    let texture_transfer_info = TextureTransferInfo::new()
        .with_transfer_buffer(&texture_transfer_buffer)
        .with_offset(0)
        .with_pixels_per_row(img_size)
        .with_rows_per_layer(img_size);
    let texture_region = TextureRegion::new()
        .with_texture(&texture)
        .with_width(img_size)
        .with_height(img_size)
        .with_depth(1);

    let mut texture_buffer_mem_map = texture_transfer_buffer.map(&gpu_device, true);
    let texture_buffer_mem_map_mem_mut: &mut [_] = texture_buffer_mem_map.mem_mut();
    for pixel_coord in 0..(img_size * img_size) as usize {
        texture_buffer_mem_map_mem_mut[pixel_coord] = [u8::MAX, u8::MAX, u8::MAX, u8::MAX];
    }
    texture_buffer_mem_map.unmap();
    copy_pass.upload_to_gpu_texture(texture_transfer_info, texture_region, true);

    // End the copy pass
    gpu_device.end_copy_pass(copy_pass);
    copy_command_buffer.submit().unwrap();

    ImageData {
        texture: texture,
        sampler: sampler,
    }
}

/* fn create_dummy_morph_buffer_and_upload_to_gpu(gpu_device: &Device) -> Buffer {
    // Start a copy pass
    let copy_command_buffer = gpu_device.acquire_command_buffer().unwrap();
    let copy_pass = gpu_device.begin_copy_pass(&copy_command_buffer).unwrap();

    // Has to be at least 4 bytes large or the gpu complains
    let morph_target_buffer = gpu_device
        .create_buffer()
        .with_size(4)
        // Not sure on the buffer usage flag? I think "Graphics Storage Read" makes sense...
        .with_usage(BufferUsageFlags::GRAPHICS_STORAGE_READ)
        .build()
        .unwrap();

    let transfer_buffer = gpu_device
        .create_transfer_buffer()
        .with_size(4)
        .with_usage(TransferBufferUsage::UPLOAD)
        .build()
        .unwrap();

    // Fill the transfer buffer with the morph target data
    // let mut buffer_mem_map = transfer_buffer.map(&gpu_device, true);
    // let buffer_mem_map_mem_mut: &mut [[f32; 4]] = buffer_mem_map.mem_mut();
    // for (index, &value) in morph_positions_vector.iter().enumerate() {
    //     // Pad with a 0.0 to make it a vec4 instead of vec3
    //     // (which avoids issues in the shader)
    //     let transfer_value: [f32; 4] = [value[0], value[1], value[2], 0.0];
    //     buffer_mem_map_mem_mut[index] = transfer_value;
    // }
    // buffer_mem_map.unmap();

    // Set the location of the data (it's at the start of the transfer buffer)
    let data_location = TransferBufferLocation::default()
        .with_transfer_buffer(&transfer_buffer)
        .with_offset(0u32);

    // Set what region of the buffer to transfer (the size of the indices data)
    let buffer_region = BufferRegion::default()
        .with_buffer(&morph_target_buffer)
        .with_size(4)
        .with_offset(0u32);

    // Upload the data
    copy_pass.upload_to_gpu_buffer(data_location, buffer_region, true);

    // End the copy pass
    gpu_device.end_copy_pass(copy_pass);
    copy_command_buffer.submit().unwrap();

    morph_target_buffer
} */

pub fn main() {
    // Initialize SDL3
    let mut sdl_context = sdl3::init().unwrap();

    // Get video subsystem
    let video_subsystem = sdl_context.video().unwrap();

    // Create a window
    let window = video_subsystem
        .window("Rust SDL3 GPU - GLTF Model Viewer", 1024, 1024)
        .resizable()
        .position_centered()
        .build()
        .unwrap();

    // Create the gpu device
    let gpu_device = Device::new(ShaderFormat::SPIRV, true)
        .unwrap()
        .with_window(&window)
        .unwrap();

    // Create imgui platform and renderer
    let mut imgui = ImGuiSdl3::new(&gpu_device, &window, |ctx| {
        // Disable creation of files on disk (Not sure, this is from the example in the repo)
        ctx.set_ini_filename(None);
        ctx.set_log_filename(None);
    });

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
    let vertex_attribute0 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(0)
        .with_format(sdl3::gpu::VertexElementFormat::Float3)
        .with_offset(0);

    // Vertex attribute for color. a_color (for the vertex input state, which is for the pipeline)
    let vertex_attribute1 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(1)
        .with_format(sdl3::gpu::VertexElementFormat::Float4)
        .with_offset(size_of::<f32>() as u32 * 3); //offset 3 f32's over to pick (xyz)

    // Vertex attribute for texture coordinate. a_tex_coord (for the vertex input state, which is for the pipeline)
    let vertex_attribute2 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(2)
        .with_format(sdl3::gpu::VertexElementFormat::Float2)
        .with_offset(size_of::<f32>() as u32 * 7); // offset 7 f32's over (xyz, rgba)

    // Joints: 4 unsigned 32-bit integers
    let vertex_attribute3 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(3)
        .with_format(sdl3::gpu::VertexElementFormat::Uint4)
        .with_offset(size_of::<f32>() as u32 * 9); // offset 9 f32's over (xyz, rgba, uv)

    // Weights: 4 f32's
    let vertex_attribute4 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(4)
        .with_format(sdl3::gpu::VertexElementFormat::Float4)
        .with_offset(size_of::<f32>() as u32 * 9 + (size_of::<u32>() as u32 * 4)); // offset 9 f32's over (xyz, rgba, uv) and 4 u32's over (j1, j2, j3, j4)

    // Morphs: 3x4 matrix of f32's
    let vertex_attribute5 = VertexAttribute::new()
        .with_buffer_slot(0)
        .with_location(5)
        .with_format(sdl3::gpu::VertexElementFormat::Float3)
        .with_offset(size_of::<f32>() as u32 * 13 + (size_of::<u32>() as u32 * 4)); // offset 9 f32's over (xyz, rgba, uv) and 4 u32's over (j1, j2, j3, j4)

    // Vertex input state (for the pipeline)
    let vertex_input_state = sdl3::gpu::VertexInputState::new()
        .with_vertex_buffer_descriptions(&[vertex_buffer_description])
        .with_vertex_attributes(&[
            vertex_attribute0,
            vertex_attribute1,
            vertex_attribute2,
            vertex_attribute3,
            vertex_attribute4,
            vertex_attribute5,
        ]);

    // Describe the color target (for the target info, which is for the pipeline)
    let color_target_description =
        ColorTargetDescription::new().with_format(gpu_device.get_swapchain_texture_format(&window));
    // Skipping blend state for now, can add that later

    let depth_stencil_texture_create_info = TextureCreateInfo::new()
        .with_type(TextureType::_2D)
        .with_format(TextureFormat::D16Unorm)
        .with_usage(TextureUsage::SAMPLER | TextureUsage::DEPTH_STENCIL_TARGET)
        .with_width(1024)
        .with_height(1024)
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
        .with_compare_op(CompareOp::Less)
        .with_enable_depth_test(true)
        .with_enable_depth_write(true);

    // Make the target info
    let target_info = GraphicsPipelineTargetInfo::new()
        .with_color_target_descriptions(&[color_target_description])
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

    // Load a model...

    loaded_models.push(load_model_and_copy_to_gpu(
        "models/Low-Poly-Base_copy.glb",
        &gpu_device,
    ));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/Avocado.glb",
    //     &gpu_device,
    // ));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/MinimalTriangle.gltf",
    //     &gpu_device,
    // ));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/AnimatedTriangle.gltf",
    //     &gpu_device,
    // ));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/cube_anim_test2.glb",
    //     &gpu_device,
    // ));

    // loaded_models.push(load_model_and_copy_to_gpu("models/ABeautifulGame.glb", &gpu_device));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/GlassHurricaneCandleHolder.glb",
    //     &gpu_device,
    // ));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/axes_test.glb",
    //     &gpu_device,
    // ));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/translate_test.glb",
    //     &gpu_device,
    // ));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/MorphTargetExample.gltf",
    //     &gpu_device,
    // ));

    // loaded_models.push(load_model_and_copy_to_gpu(
    //     "models/VertexSkinExample.gltf",
    //     &gpu_device,
    // ));

    // Put a dummy image into the gpu
    let dummy_image = create_dummy_image_and_copy_to_gpu(&gpu_device);

    // Put a dummy morph buffer into the gpu
    // let dummy_morph = create_dummy_morph_buffer_and_upload_to_gpu(&gpu_device);

    // Done with gpu init

    // Initialize game variables
    let mut player_x = 0.0;
    let mut player_y = 0.0;
    let mut game_ticks = 0u32;
    let mut player_z = 0.0;

    // Keyboard state variables
    let mut key_up = false;
    let mut key_down = false;
    let mut key_left = false;
    let mut key_right = false;
    let mut key_q = false;
    let mut key_e = false;
    let mut key_r = false;

    // Mouse state variables
    let mut mouse_down = false;
    let mut mouse_drag_x = 0.0;
    let mut mouse_drag_y = 0.0;

    let mut current_animation = 0;
    let mut max_animation: usize = 0;

    // Event handling
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        game_ticks += 1;
        let game_seconds = (game_ticks as f32) / 60.0;
        let game_seconds_modulo = game_seconds % 1.0;
        for event in event_pump.poll_iter() {
            imgui.handle_event(&event);
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
                Event::KeyDown {
                    keycode: Some(Keycode::Q),
                    ..
                } => {
                    key_q = true;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::E),
                    ..
                } => {
                    key_e = true;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::R),
                    ..
                } => {
                    key_r = true;
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
                Event::KeyUp {
                    keycode: Some(Keycode::Q),
                    ..
                } => {
                    key_q = false;
                }
                Event::KeyUp {
                    keycode: Some(Keycode::E),
                    ..
                } => {
                    key_e = false;
                }
                Event::KeyUp {
                    keycode: Some(Keycode::R),
                    ..
                } => {
                    key_r = false;
                }
                // Mouse events
                Event::MouseMotion { xrel, yrel, .. } => {
                    if mouse_down {
                        mouse_drag_x += xrel;
                        mouse_drag_y += yrel;
                    }
                }
                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Left,
                    ..
                } => {
                    mouse_down = true;
                }
                Event::MouseButtonUp {
                    mouse_btn: MouseButton::Left,
                    ..
                } => {
                    mouse_down = false;
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
        if key_q {
            player_z += 0.02;
        }
        if key_e {
            player_z -= 0.02;
        }
        if key_r {
            player_x = 0.0;
            player_y = 0.0;
            player_z = 0.0;
            mouse_drag_x = 0.0;
            mouse_drag_y = 0.0;
        }

        // End of game code

        // Acquire the command buffer
        let mut command_buffer = gpu_device.acquire_command_buffer().unwrap();

        // Update the position value in the position uniform
        // position_uniform.x_pos = player_x;
        // position_uniform.y_pos = player_y;

        // let player_transform = multiply_matrices(
        //     multiply_matrices(
        //         matrix_rotate_around_x(player_y),
        //         matrix_rotate_around_y(player_x),
        //     ),
        //     matrix_rotate_around_z(player_z),
        // );
        // let player_transform = multiply_matrices(
        //     multiply_matrices(
        //         matrix_scale_multi(0.3, 0.3, 0.3),
        //         matrix_rotate_multi_axis(0.0, player_x, player_y),
        //     ),
        //     matrix_translate_multi(player_z, player_z, player_z),
        // );
        let player_transform = multiply_matrices_chain(vec![
            matrix_translate_multi(player_x, player_y, player_z),
            // Scale X by -1 to convert from gltf right handedness to sdl-gpu left handedness
            matrix_scale_multi(-1.0, 1.0, 1.0),
            matrix_rotate_multi_axis(0.0, (mouse_drag_x) / 64.0, (mouse_drag_y) / 64.0),
        ]);

        // Push the position uniform data
        // command_buffer.push_vertex_uniform_data(0, &position_uniform);

        // Get the swapchain texture
        let swapchain_texture = command_buffer
            .wait_and_acquire_swapchain_texture(&window)
            .unwrap();
        // Should check if swapchain buffer is actually available, and submit command buffer early if not available?

        // Create the color target
        // let color_target_info =
        let color_targets = [ColorTargetInfo::default()
            .with_clear_color(Color {
                r: 200u8,
                g: 250u8,
                b: 255u8,
                a: 255u8,
            })
            .with_load_op(SDL_GPU_LOADOP_CLEAR)
            .with_store_op(SDL_GPU_STOREOP_STORE)
            .with_texture(&swapchain_texture)];

        let color_targets_ui = [ColorTargetInfo::default()
            // .with_clear_color(Color {
            //     r: 240u8,
            //     g: 250u8,
            //     b: 255u8,
            //     a: 255u8,
            // })
            // .with_load_op(SDL_GPU_LOADOP_CLEAR)
            .with_store_op(SDL_GPU_STOREOP_STORE)
            .with_texture(&swapchain_texture)];

        // Begin a render pass
        let render_pass = gpu_device
            .begin_render_pass(
                &command_buffer,
                // &[color_target_info],
                &color_targets,
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
                // Start with identity matrix. The pair is: (current node, current transformation)
                remaining_node_transform_pairs.push((node, IDENTITY_MATRIX));

                // Start with current player rotation
                // remaining_node_transform_pairs.push((node, player_transform));
            }

            max_animation = model.animations.len().clone();

            while !remaining_node_transform_pairs.is_empty() {
                // Take a node off the list
                let (node, inherited_transform_matrix) =
                    remaining_node_transform_pairs.pop().unwrap();

                // Calculate transform for this node based on it's inherited transform and it's local transform
                // Wrong order...? (Not sure)

                // let multiplied_transform_matrix =
                //     multiply_matrices(inherited_transform_matrix, node.transform().matrix());

                // let multiplied_transform_matrix =
                //     multiply_matrices(node.transform().matrix(), inherited_transform_matrix);

                // Create my own matrix instead of using the one from the library?
                // (Gltf library says the matrix is translation * rotation * scale)
                // let (d_translate, d_rotation, d_scale) = node.transform().decomposed();
                // let alternate_matrix = multiply_matrices_chain(vec![
                //     matrix_translate_multi(d_translate[0], d_translate[1], d_translate[2]),
                //     matrix_rotate_from_quaternion(
                //         d_rotation[0],
                //         d_rotation[1],
                //         d_rotation[2],
                //         d_rotation[3],
                //     ),
                //     matrix_scale_multi(d_scale[0], d_scale[1], d_scale[2]),
                // ]);
                let flipped_matrix = flip_matrix_diagonally(node.transform().matrix());
                // They are in fact different, it seems. Not sure exactly why but probably something to do
                // with "column or row major order"

                let multiplied_transform_matrix_pre_animation =
                    // multiply_matrices(inherited_transform_matrix, alternate_matrix);
                    multiply_matrices(inherited_transform_matrix, flipped_matrix);

                let (animation_transform_matrix, morph_weights) = {
                    let mut animated_morph_weights = [0.0f32, 0.0f32, 0.0f32, 0.0f32];
                    // TODO could probably do some fancy vector reduce trick but for now it just has a mutable variable that all relevant animations apply to
                    // TODO Not sure if this is the correct, approach, might need to get the translate, rotate, scale components out and multiply them in a specific order?
                    let mut animation_full_matrix = IDENTITY_MATRIX;
                    // TODO Should somehow cache which animations -> channels go with which nodes instead of searching here, but for now it's fine

                    for animation in &model.animations {
                        // Look up original animation for reference
                        // let animation_ref = model
                        //     .document
                        //     .animations()
                        //     .skip(animation.animation_index)
                        //     .next()
                        //     .unwrap();

                        // TODO - should check if the animation is "active", otherwise all animations will be playing constantly
                        // For now just doing animation 0
                        if animation.animation_index == 0 {
                            // TODO - This is resource intensive and should be optimized
                            // Maybe split up the for loops so it's not looping through all the animations inside all the nodes?
                            // Maybe cache the results of the animations? (calculated transforms and weights)

                            // https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_004_ScenesNodes.html#global-transforms-of-nodes
                            // The gltf tutorial suggests:
                            // Cache the calculated global transforms of each node
                            // Detect changes in the local transforms of ancestor nodes, then
                            // Update global transforms only when necessary

                            // Also my 3d model is probably incredibly poorly optimized

                            for channel in &animation.channels {
                                if channel.target_node_index == node.index() {
                                    // Look up original channel for reference
                                    // let channel_ref = animation_ref
                                    //     .channels()
                                    //     .skip(channel.channel_index)
                                    //     .next()
                                    //     .unwrap();
                                    // This is a channel that matches the current node in the model, so apply the animation's transforms to the animation_matrix

                                    // Based on the current time and the input (time) data , figure out the output value
                                    // channel.inputs

                                    let current_time = game_seconds
                                        % channel
                                            .inputs
                                            .iter()
                                            .max_by(|a, b| a.total_cmp(b))
                                            .unwrap();

                                    // TODO so far this is only linear interpolation, also needs to support "step" and "cubic spline" interpolations

                                    // https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_007_Animations.html#animation-samplers
                                    // Previous time: Largest time value from inputs that is smaller than current_time
                                    let (previous_index, previous_time) = channel
                                        .inputs
                                        .iter()
                                        .enumerate()
                                        .filter(|(_, time)| time <= &&current_time)
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .unwrap_or_else(|| {
                                            // println!("current_time is: {}", current_time);
                                            (0, &0.0f32)
                                        });

                                    // Next time: Smallest time value from inputs that is greater than current_time
                                    let (next_index, next_time) = channel
                                        .inputs
                                        .iter()
                                        .enumerate()
                                        .filter(|(_, time)| time > &&current_time)
                                        .min_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .unwrap_or_else(|| {
                                            // println!("hi 2");
                                            (0, &0.0f32)
                                        });

                                    let previous_transform =
                                        channel.outputs.get(previous_index).unwrap();
                                    let next_transform = channel.outputs.get(next_index).unwrap();
                                    let interpolation_value = (current_time - previous_time)
                                        / (next_time - previous_time);
                                    // let interpolated_transform = previous_transform + interpolation_value * (next_transform - previous_transform);
                                    let interpolated_transform = match (
                                        previous_transform,
                                        next_transform,
                                    ) {
                                        (
                                            &AnimationOutput::Translation(previous_translation),
                                            &AnimationOutput::Translation(next_translation),
                                        ) => Some(matrix_translate_multi(
                                            previous_translation[0]
                                                + interpolation_value
                                                    * (next_translation[0]
                                                        - previous_translation[0]),
                                            previous_translation[1]
                                                + interpolation_value
                                                    * (next_translation[1]
                                                        - previous_translation[1]),
                                            previous_translation[2]
                                                + interpolation_value
                                                    * (next_translation[2]
                                                        - previous_translation[2]),
                                        )),
                                        (
                                            &AnimationOutput::Rotation(previous_rotation),
                                            &AnimationOutput::Rotation(
                                                next_rotation_before_negative_check,
                                            ),
                                        ) => {
                                            // https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_007_Animations.html#linear
                                            let dot_product_before_negative_check =
                                                quaternion_dot_product(
                                                    previous_rotation,
                                                    next_rotation_before_negative_check,
                                                );
                                            // If dot product is negative, "take the shortest path" / "go the other way around the sphere" by multiplying the dot product and the next rotation by -1
                                            let (next_rotation, dot_product) =
                                                if dot_product_before_negative_check < 0.0 {
                                                    (
                                                        [
                                                            next_rotation_before_negative_check[0]
                                                                * -1.0,
                                                            next_rotation_before_negative_check[1]
                                                                * -1.0,
                                                            next_rotation_before_negative_check[2]
                                                                * -1.0,
                                                            next_rotation_before_negative_check[3]
                                                                * -1.0,
                                                        ],
                                                        dot_product_before_negative_check * -1.0,
                                                    )
                                                } else {
                                                    (
                                                        next_rotation_before_negative_check,
                                                        dot_product_before_negative_check,
                                                    )
                                                };
                                            // If the previous and next quaternions are very close, just linear interpolate between them
                                            Some(if dot_product > 0.9995 {
                                                matrix_rotate_from_quaternion(
                                                    previous_rotation[0]
                                                        + interpolation_value
                                                            * (next_rotation[0]
                                                                - previous_rotation[0]),
                                                    previous_rotation[1]
                                                        + interpolation_value
                                                            * (next_rotation[1]
                                                                - previous_rotation[1]),
                                                    previous_rotation[2]
                                                        + interpolation_value
                                                            * (next_rotation[2]
                                                                - previous_rotation[2]),
                                                    previous_rotation[3]
                                                        + interpolation_value
                                                            * (next_rotation[3]
                                                                - previous_rotation[3]),
                                                )
                                            } else {
                                                // Otherwise, calculate the spherical linear interpolation
                                                let theta_0 = dot_product.acos();
                                                let theta = interpolation_value * theta_0;
                                                let sin_theta = theta.sin();
                                                let sin_theta_0 = theta_0.sin();
                                                let scale_previous_quaternion = theta.cos()
                                                    - dot_product * sin_theta / sin_theta_0;
                                                let scale_next_quaternion = sin_theta / sin_theta_0;
                                                matrix_rotate_from_quaternion(
                                                    scale_previous_quaternion
                                                        * previous_rotation[0]
                                                        + scale_next_quaternion * next_rotation[0],
                                                    scale_previous_quaternion
                                                        * previous_rotation[1]
                                                        + scale_next_quaternion * next_rotation[1],
                                                    scale_previous_quaternion
                                                        * previous_rotation[2]
                                                        + scale_next_quaternion * next_rotation[2],
                                                    scale_previous_quaternion
                                                        * previous_rotation[3]
                                                        + scale_next_quaternion * next_rotation[3],
                                                )
                                            })
                                        }
                                        (
                                            &AnimationOutput::Scale(previous_scale),
                                            &AnimationOutput::Scale(next_scale),
                                        ) => Some(matrix_scale_multi(
                                            previous_scale[0]
                                                + interpolation_value
                                                    * (next_scale[0] - previous_scale[0]),
                                            previous_scale[1]
                                                + interpolation_value
                                                    * (next_scale[1] - previous_scale[1]),
                                            previous_scale[2]
                                                + interpolation_value
                                                    * (next_scale[2] - previous_scale[2]),
                                        )),
                                        (
                                            AnimationOutput::MorphTargetWeight(previous_weight),
                                            AnimationOutput::MorphTargetWeight(next_weight),
                                        ) => {
                                            // Calculate weights
                                            // For 0
                                            animated_morph_weights[0] = previous_weight[0]
                                                + interpolation_value
                                                    * (next_weight[0] - previous_weight[0]);
                                            // For 1
                                            if previous_weight.len() > 1 {
                                                animated_morph_weights[1] = previous_weight[1]
                                                    + interpolation_value
                                                        * (next_weight[1] - previous_weight[1]);
                                            }
                                            // For 2
                                            if previous_weight.len() > 2 {
                                                animated_morph_weights[2] = previous_weight[2]
                                                    + interpolation_value
                                                        * (next_weight[2] - previous_weight[2]);
                                            }
                                            // For 3
                                            if previous_weight.len() > 3 {
                                                animated_morph_weights[3] = previous_weight[3]
                                                    + interpolation_value
                                                        * (next_weight[3] - previous_weight[3]);
                                            }
                                            // Return identity matrix for the transform matrix part
                                            // IDENTITY_MATRIX

                                            // Instead, return None and skip it
                                            None
                                        }
                                        _ => panic!(
                                            "should always have the same animation type for previous and next"
                                        ),
                                    };

                                    // let output_index: usize =
                                    //     (game_ticks as usize) % channel.outputs.len();
                                    // let output_value = channel.outputs.get(output_index).unwrap();
                                    // let animation_piece_matrix = match output_value {
                                    //     AnimationOutput::Translation(translation) => {
                                    //         matrix_translate_multi(
                                    //             translation[0],
                                    //             translation[1],
                                    //             translation[2],
                                    //         )
                                    //     }
                                    //     &AnimationOutput::Rotation(rotation) => {
                                    //         matrix_rotate_from_quaternion(
                                    //             rotation[0],
                                    //             rotation[1],
                                    //             rotation[2],
                                    //             rotation[3],
                                    //         )
                                    //     }
                                    //     &AnimationOutput::Scale(scale) => {
                                    //         matrix_scale_multi(scale[0], scale[1], scale[2])
                                    //     }
                                    //     &AnimationOutput::MorphTargetWeight(_weight) => IDENTITY_MATRIX,
                                    // };
                                    if interpolated_transform.is_some() {
                                        // TODO not sure if correct multiplication order
                                        animation_full_matrix = multiply_matrices(
                                            animation_full_matrix,
                                            interpolated_transform.unwrap(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    (animation_full_matrix, animated_morph_weights)
                };

                // TODO - check if this is the right order, I don't have a good intuition about which matrix should come first
                let multiplied_transform_matrix = multiply_matrices(
                    multiplied_transform_matrix_pre_animation,
                    animation_transform_matrix,
                );

                if let Some(mesh) = node.mesh() {
                    // Render mesh with transform

                    // Player's Transform Matrix  *  Model's Transform Matrix is correct
                    let multiplied_and_player_transform_matrix =
                        multiply_matrices(player_transform, multiplied_transform_matrix);
                    // multiply_matrices(multiplied_transform_matrix, player_transform);

                    // Send transform to shader as a uniform
                    // command_buffer.push_vertex_uniform_data(0, &multiplied_and_player_transform_matrix);
                    // command_buffer.push_vertex_uniform_data(0, &multiplied_transform_matrix);
                    // command_buffer.push_vertex_uniform_data(0, &IDENTITY_MATRIX);
                    // command_buffer.push_vertex_uniform_data(0, &player_transform);

                    let mesh_data = model.meshes.get(mesh.index()).unwrap();
                    for primitive in mesh.primitives() {
                        // Look up the primitive data
                        let primitive_data = mesh_data.primitives.get(primitive.index()).unwrap();

                        command_buffer.push_vertex_uniform_data(
                            0,
                            &VertexUniformBuffer {
                                transform_matrix: multiplied_and_player_transform_matrix,
                                morph_weights: morph_weights,
                                // morph_target_count: primitive_data.morph_target_count,
                                // vertex_count: primitive_data.vertex_count,
                                joint_matrices: Vec::new(),
                            },
                        );

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
                        let base_color_texture_option = primitive
                            .material()
                            .pbr_metallic_roughness()
                            .base_color_texture();

                        if let Some(base_color_texture) = base_color_texture_option {
                            // Texture has image
                            let image_index = base_color_texture.texture().source().index();
                            let image_data = model.images.get(image_index).unwrap();

                            // Bind sampler
                            let texture_sampler_binding = TextureSamplerBinding::new()
                                .with_texture(&image_data.texture)
                                .with_sampler(&image_data.sampler);
                            render_pass.bind_fragment_samplers(0, &[texture_sampler_binding]);
                        } else {
                            // No image - use dummy image

                            // Bind sampler
                            let texture_sampler_binding = TextureSamplerBinding::new()
                                .with_texture(&dummy_image.texture)
                                .with_sampler(&dummy_image.sampler);
                            render_pass.bind_fragment_samplers(0, &[texture_sampler_binding]);
                        }

                        // Is there a way to not clone the buffer? Or does it not matter?
                        // if primitive_data.morph_target_buffer.is_some() {
                        //     render_pass.bind_vertex_storage_buffers(
                        //         0,
                        //         &[primitive_data.morph_target_buffer.clone().unwrap()],
                        //     );
                        // } else {
                        //     render_pass.bind_vertex_storage_buffers(0, &[dummy_morph.clone()]);
                        // }

                        // if let Some(morph_target_buffer) = primitive_data.morph_target_buffer {
                        //     // let buffer_bindings_morph = BufferBinding::new()
                        //     //     .with_buffer(&morph_target_buffer)
                        //     //     .with_offset(0);
                        //     // render_pass.bind_vertex_storage_buffers(0, &[buffer_bindings_morph]);
                        //     render_pass.bind_vertex_storage_buffers(0, &[morph_target_buffer]);
                        // }

                        // Issue the draw call using the indexes and other bound data
                        render_pass.draw_indexed_primitives(primitive_data.index_count, 1, 0, 0, 0);
                    }

                    // End of rendering a mesh
                }
                for child_node in node.children() {
                    // Add child nodes to the list
                    remaining_node_transform_pairs.push((child_node, multiplied_transform_matrix));
                }
            }
        }

        // End the render pass
        gpu_device.end_render_pass(render_pass);

        // Start another render pass for the gui

        // Display gui with imgui
        imgui.render(
            &mut sdl_context,
            &gpu_device,
            &window,
            &event_pump,
            &mut command_buffer,
            // &[color_target_info],
            &color_targets_ui,
            |ui| {
                ui.text(format!("Player x: {}", player_x));
                ui.text(format!("Player y: {}", player_y));
                ui.text(format!("Player z: {}", player_z));
                ui.text(format!("Rotate x: {}", mouse_drag_x));
                ui.text(format!("Rotate y: {}", mouse_drag_y));
                ui.text(format!("Frames: {}", game_ticks));
                // ui.text(format!("Seconds modulo: {}", game_seconds_modulo));
                let reset_button = ui.button("Reset camera");
                if reset_button {
                    player_x = 0.0;
                    player_y = 0.0;
                    player_z = 0.0;
                    mouse_drag_x = 0.0;
                    mouse_drag_y = 0.0;
                }
                ui.text(format!("Max animations: {}", max_animation));
                ui.text(format!("Current animation {}", current_animation));
                let animation_button = ui.button("Next animation");
                if animation_button {
                    current_animation += 1;
                    if current_animation > max_animation {
                        current_animation = 0;
                    }
                }
            },
        );

        // Submit the command buffer
        command_buffer.submit().unwrap();

        // Wait for next frame
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
    println!("Exited loop");

    // Release buffers and pipeline, destroy gpu device and window here (?)
    // drop(pipeline);
    // drop(window);
    // drop(gpu_device);
}

const IDENTITY_MATRIX: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// fn matrix_rotate_around_x(a: f32) -> [[f32; 4]; 4] {
//     // roll
//     [
//         [1.0, 0.0, 0.0, 0.0],
//         [0.0, a.cos(), -a.sin(), 0.0],
//         [0.0, a.sin(), a.cos(), 0.0],
//         [0.0, 0.0, 0.0, 1.0],
//     ]
// }
//
// fn matrix_rotate_around_y(a: f32) -> [[f32; 4]; 4] {
//     // pitch
//     [
//         [a.cos(), 0.0, a.sin(), 0.0],
//         [0.0, 1.0, 0.0, 0.0],
//         [-a.sin(), 0.0, a.cos(), 0.0],
//         [0.0, 0.0, 0.0, 1.0],
//     ]
// }
//
// fn matrix_rotate_around_z(a: f32) -> [[f32; 4]; 4] {
//     // yaw
//     [
//         [a.cos(), -a.sin(), 0.0, 0.0],
//         [a.sin(), a.cos(), 0.0, 0.0],
//         [0.0, 0.0, 1.0, 0.0],
//         [0.0, 0.0, 0.0, 1.0],
//     ]
// }

fn matrix_rotate_from_quaternion(x: f32, y: f32, z: f32, s: f32) -> [[f32; 4]; 4] {
    [
        [
            1.0 - (2.0 * y * y) - (2.0 * z * z),
            (2.0 * x * y) - (2.0 * s * z),
            (2.0 * x * z) + (2.0 * s * y),
            0.0,
        ],
        [
            (2.0 * x * y) + (2.0 * s * z),
            1.0 - (2.0 * x * x) - (2.0 * z * z),
            (2.0 * y * z) - (2.0 * s * x),
            0.0,
        ],
        [
            (2.0 * x * z) - (2.0 * s * y),
            (2.0 * y * z) + (2.0 * s * x),
            1.0 - (2.0 * x * x) - (2.0 * y * y),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn matrix_rotate_multi_axis(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    // x, y, z : yaw, pitch, roll
    // https://en.wikipedia.org/wiki/Rotation_matrix
    [
        [
            x.cos() * y.cos(),
            x.cos() * y.sin() * z.sin() - x.sin() * z.cos(),
            x.cos() * y.sin() * z.cos() + x.sin() * z.sin(),
            0.0,
        ],
        [
            x.sin() * y.cos(),
            x.sin() * y.sin() * z.sin() + x.cos() * z.cos(),
            x.sin() * y.sin() * z.cos() - x.cos() * z.sin(),
            0.0,
        ],
        [-y.sin(), y.cos() * z.sin(), y.cos() * z.cos(), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn multiply_matrices_chain(matrices: Vec<[[f32; 4]; 4]>) -> [[f32; 4]; 4] {
    matrices
        .iter()
        .fold(IDENTITY_MATRIX, |a, b| multiply_matrices(a, *b))
}

fn multiply_matrices(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // https://en.wikipedia.org/wiki/Matrix_multiplication
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2],
            a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2],
            a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2],
            a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3],
        ],
        [
            a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] + a[3][3] * b[3][0],
            a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] + a[3][3] * b[3][1],
            a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] + a[3][3] * b[3][2],
            a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] + a[3][3] * b[3][3],
        ],
    ]
}

fn matrix_scale_multi(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn matrix_translate_multi(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        // [1.0, 0.0, 0.0, 0.0],
        // [0.0, 1.0, 0.0, 0.0],
        // [0.0, 0.0, 1.0, 0.0],
        // [x, y, z, 1.0],
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn flip_matrix_diagonally(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

fn quaternion_dot_product(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

#[cfg(test)]
mod tests {
    use crate::{IDENTITY_MATRIX, multiply_matrices};

    #[test]
    fn test_multiply_matrices_1() {
        assert_eq!(
            multiply_matrices(IDENTITY_MATRIX.clone(), IDENTITY_MATRIX.clone()),
            IDENTITY_MATRIX
        );
    }
}
