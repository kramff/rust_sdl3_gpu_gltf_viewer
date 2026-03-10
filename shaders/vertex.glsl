#version 460

// To compile:
// glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_tex_coord;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_tex_coord;

layout(std140, set = 1, binding = 0) uniform UniformBlock {
    // float x_pos;
    // float y_pos;
    mat4 transform_matrix;
};

void main()
{
    // Apply transformation from transformation matrix
    vec4 a_position_transformed = vec4(a_position, 1.0) * transform_matrix;
    // vec4 a_position_transformed = vec4(a_position, 1.0);

    // change the x and y values with the z value to create the illusion of distance/depth
    vec4 a_position_projected = vec4(
            4 * a_position_transformed.x / (4 + a_position_transformed.z),
            4 * a_position_transformed.y / (4 + a_position_transformed.z),
            a_position_transformed.z,
            a_position_transformed.w
        );

    // vec4 a_position_projected = a_position_transformed;

    // https://learnopengl.com/Advanced-OpenGL/Depth-testing
    // Z from 3d coordinates, plus a bit to push away from camera
    // float one_over_z = 1.0f / (a_position_transformed.z + 1.1);
    // Near set to 0.1;
    // float one_over_near = 1.0f / 0.1f;
    // Far set to 100;
    // float one_over_far = 1.0f / 10.0f;
    // Depth = (1 / z - 1 / near) / (1 / far - 1 / near)
    // float depth = (one_over_z - one_over_near) / (one_over_far - one_over_near);
    float depth = (1.0f / (a_position_projected.z + 1.1f) - 10.0f) / -10.0f;

    gl_Position = vec4(
            a_position_projected.x,
            a_position_projected.y,
            depth,
            1.0f
        );

    // Pass vertex color to fragment shader
    v_color = a_color;

    // Pass texture coordinate to fragment shader
    v_tex_coord = a_tex_coord;
}
