#version 460

// To compile:
// glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_tex_coord;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_tex_coord;

// set = 1: for vertex shaders, set 1 is used for uniforms
layout(std140, set = 1, binding = 0) uniform UniformBlock {
    // float x_pos;
    // float y_pos;

    // TODO - Figure out more about "Uniform Buffer Layout"
    // https://registry.khronos.org/OpenGL/specs/gl/glspec45.core.pdf#page=159
    // If these uniform values are in a different order, they get messed up
    // Because there's padding around them to make them, and my rendering code isn't accounting for that
    // (Or something like that, I'm not entirely sure)
    mat4 transform_matrix;
    vec4 morph_weights;
    uint morph_target_count;
    uint vertex_count;
};


layout(std430, binding = 0) readonly buffer BufferBlock {
    vec4 morph_target[];
};

void main()
{
    // Apply the morph target
    
    vec4 morph_1;
    if (morph_target_count >= 1) {
        morph_1 = morph_target[gl_VertexIndex];
    }
    else {
        morph_1 = vec4(0.0, 0.0, 0.0, 0.0);
    }
    vec4 morph_2;
    if (morph_target_count >= 2) {
        morph_2 = morph_target[vertex_count + gl_VertexIndex];
    }
    else {
        morph_2 = vec4(0.0, 0.0, 0.0, 0.0);
    }
    vec4 morph_3;
    if (morph_target_count >= 3) {
        morph_3 = morph_target[(2 * vertex_count) + gl_VertexIndex];
    }
    else {
        morph_3 = vec4(0.0, 0.0, 0.0, 0.0);
    }
    vec4 morph_4;
    if (morph_target_count >= 4) {
        morph_4 = morph_target[(3 * vertex_count) + gl_VertexIndex];
    }
    else {
        morph_4 = vec4(0.0, 0.0, 0.0, 0.0);
    }
    vec4 a_position_morphed = vec4(
            a_position.x + morph_1.x * morph_weights[0] + morph_2.x * morph_weights[1] + morph_3.x * morph_weights[2] + morph_4.x * morph_weights[3],
            a_position.y + morph_1.y * morph_weights[0] + morph_2.y * morph_weights[1] + morph_3.y * morph_weights[2] + morph_4.y * morph_weights[3],
            a_position.z + morph_1.z * morph_weights[0] + morph_2.z * morph_weights[1] + morph_3.z * morph_weights[2] + morph_4.z * morph_weights[3],
            1.0
        );
    
    // vec4 a_position_morphed = vec4(
    //         a_position.x + morph_weights.x,
    //         a_position.y + morph_weights.y,
    //         a_position.z + morph_weights.z,
    //         1.0
    //     );
    // vec4 a_position_morphed = vec4(
    //         a_position.x,
    //         a_position.y,
    //         a_position.z,
    //         1.0
    //     );

    // Apply transformation from transformation matrix
    vec4 a_position_transformed = vec4(a_position_morphed) * transform_matrix;
    // vec4 a_position_transformed = vec4(a_position, 1.0) * transform_matrix;
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
