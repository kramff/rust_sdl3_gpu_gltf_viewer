#version 460

// To compile:
// glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv

// VERTEX ATTRIBUTES

// x, y, z position
layout(location = 0) in vec3 position;

// r, g, b, a color
layout(location = 1) in vec4 color;

// u, v texture coordinate
layout(location = 2) in vec2 tex_coord;

// 4 unsigned 32-bit integers for joints, which represent the index of the joint to use
layout(location = 3) in uvec4 joints;

// 4 f32's for weights, to be used with joints for vertex skinning animations
layout(location = 4) in vec4 weights;

// 3x4 matrix of morphs: each morph is an x, y, z coordinate
layout(location = 5) in mat3x4 morphs;

// VARIABLES TO PASS TO FRAGMENT SHADER

// r, g, b, a, color to pass to fragment shader
layout(location = 0) out vec4 v_color;

// u, v texture coordinates to pass to fragment shader
layout(location = 1) out vec2 v_tex_coord;

// UNIFORM

// for vertex shaders, set 1 is used for uniforms
layout(std140, set = 1, binding = 0) uniform UniformBlock {
    // float x_pos;
    // float y_pos;

    // TODO - Figure out more about "Uniform Buffer Layout"
    // https://registry.khronos.org/OpenGL/specs/gl/glspec45.core.pdf#page=159
    // If these uniform values are in a different order, they get messed up
    // Because there's padding around them to make them, and my rendering code isn't accounting for that
    // (Or something like that, I'm not entirely sure)

    // the computed transform matrix
    mat4 transform_matrix;

    // how much to apply the (up to 4) morphs
    vec4 morph_weights;

    // Array of matrix transforms, one for each joint
    mat4 joint_matrices[600];
};

// CODE

void main()
{
    // Apply morph targets
    vec3 position3 = position + morph_weights * morphs;

    // Make position a vec4 instead of a vec3
    vec4 position4 = vec4(position3.x, position3.y, position3.z, 1.0);

    // Prep for vertex skinning
    mat4 skin_matrix =
        joint_matrices[joints.x] * weights.x +
        joint_matrices[joints.y] * weights.y +
        joint_matrices[joints.z] * weights.z +
        joint_matrices[joints.w] * weights.w;

    // Apply vertex skinning
    position4 = position4 * skin_matrix;

    // Apply transform matrix
    position4 = position4 * transform_matrix;

    // Create illusion of depth by dividing x and y by z
    position4 = vec4(
        4 * position4.x / (4 + position4.z),
        4 * position4.y / (4 + position4.z),
        position4.z,
        position4.w
    );

    // Calculate depth value for depth testing
    float depth = (1.0f / (position4.z + 1.1f) - 10.0f) / -10.0f;

    // Pass (x, y) position and depth to fragment shader
    gl_Position = vec4(
        position4.x,
        position4.y,
        depth,
        1.0f
    );

    // Pass vertex color to fragment shader
    v_color = color;

    // Pass texture coordinate to fragment shader
    v_tex_coord = tex_coord;
    
    // Apply the morph target

    // TODO - This code could probably be way more efficient.
    // Check how many morph target counts there are FIRST and then go down different branches.
    // Try to minimize how many if statements there are and minimize unnecessary calculations.
    // Is it better to have more code that isn't ran in all cases than to do extra work every time?
    // Or, is it better to have no conditionals? Not sure

    /*
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
    // This is doing so much extra work especially if there are no morph weights being used.
    vec4 a_position_morphed = vec4(
            a_position.x + morph_1.x * morph_weights[0] + morph_2.x * morph_weights[1] + morph_3.x * morph_weights[2] + morph_4.x * morph_weights[3],
            a_position.y + morph_1.y * morph_weights[0] + morph_2.y * morph_weights[1] + morph_3.y * morph_weights[2] + morph_4.y * morph_weights[3],
            a_position.z + morph_1.z * morph_weights[0] + morph_2.z * morph_weights[1] + morph_3.z * morph_weights[2] + morph_4.z * morph_weights[3],
            1.0
        );
    */

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
    /* vec4 a_position_transformed = vec4(a_position_morphed) * transform_matrix; */
    // vec4 a_position_transformed = vec4(a_position, 1.0) * transform_matrix;
    // vec4 a_position_transformed = vec4(a_position, 1.0);

    // change the x and y values with the z value to create the illusion of distance/depth
    /* vec4 a_position_projected = vec4(
            4 * a_position_transformed.x / (4 + a_position_transformed.z),
            4 * a_position_transformed.y / (4 + a_position_transformed.z),
            a_position_transformed.z,
            a_position_transformed.w
        ); */

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
    /* float depth = (1.0f / (a_position_projected.z + 1.1f) - 10.0f) / -10.0f; */

    /* gl_Position = vec4(
            a_position_projected.x,
            a_position_projected.y,
            depth,
            1.0f
        ); */

    // Pass vertex color to fragment shader
    /* v_color = color; */

    // Pass texture coordinate to fragment shader
    /* v_tex_coord = tex_coord; */
}
