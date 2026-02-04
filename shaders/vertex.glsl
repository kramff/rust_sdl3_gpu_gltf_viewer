#version 460

// To compile: 
// glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec4 a_color;
layout (location = 2) in vec2 a_tex_coord;

layout (location = 0) out vec4 v_color;
layout (location = 1) out vec2 v_tex_coord;

layout(std140, set = 1, binding = 0) uniform UniformBlock {
    float x_pos;
    float y_pos;
};

void main()
{
    // Rotate around Y axis (XZ plane rotated)
    vec4 a_position_rotated_1 = vec4(
        a_position[0] * cos(x_pos) - a_position[2] * sin(x_pos),
        a_position[1],
        a_position[2] * cos(x_pos) + a_position[0] * sin(x_pos),
        1.0f
    );

    // Rotate around X axis (YZ plane rotated)
    vec4 a_position_rotated_2 = vec4(
        a_position_rotated_1[0],
        a_position_rotated_1[1] * cos(y_pos) - a_position_rotated_1[2] * sin(y_pos),
        a_position_rotated_1[2] * cos(y_pos) + a_position_rotated_1[1] * sin(y_pos),
        1.0f
    );
    // gl_Position = a_position_rotated_2;

    // Make vertices appear close / far depending on the Z position (projected)
    // vec4 a_position_z_projected = vec4(
    //     a_position_rotated_2[0] / (a_position_rotated_2[2] + 2) * 2,
    //     a_position_rotated_2[1] / (a_position_rotated_2[2] + 2) * 2,
    //     a_position_rotated_2[2],
    //     1.0f
    // );
    // gl_Position = a_position_z_projected;

    // Transform to try and match what the depth buffer is expecting (??)
    // vec4 a_position_transformed_for_depth_buffer = vec4(
    //     a_position_rotated_2.x,
    //     a_position_rotated_2.y,
    //     // Just kinda guessing at these numbers, trying to fit the z values into 0.0 to 1.0 so the depth values are within those bounds as well
    //     (1.5f - a_position_rotated_2.z) * 0.3,
    //     1.0f
    // );
    // Newer version that doesn't flip Z backwards

    // https://learnopengl.com/Advanced-OpenGL/Depth-testing
    // Z from 3d coordinates, plus a bit to push away from camera
    float one_over_z = 1.0f / (a_position_rotated_2.z + 1.1);
    // Near set to 0.1;
    float one_over_near = 1.0f / 0.1f;
    // Far set to 100;
    float one_over_far = 1.0f / 10.0f;
    // Depth = (1 / z - 1 / near) / (1 / far - 1 / near)
    float depth = (one_over_z - one_over_near) / (one_over_far - one_over_near);

    vec4 a_position_transformed_for_depth_buffer = vec4(
        a_position_rotated_2.x,
        a_position_rotated_2.y,
        // Just kinda guessing at these numbers, trying to fit the z values into 0.0 to 1.0 so the depth values are within those bounds as well
        // (a_position_rotated_2.z + 0.5) * 0.3,
        depth,
        1.0f
    );
    gl_Position = a_position_transformed_for_depth_buffer;

    // Pass vertex color to fragment shader
    v_color = a_color;

    // Pass texture coordinate to fragment shader
    v_tex_coord = a_tex_coord;
}

