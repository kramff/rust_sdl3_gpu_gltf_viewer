#version 460

// 
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
    // Position based on x_pos and y_pos uniforms in a faked 3d scuffed version for now
    gl_Position = vec4(a_position[0] * cos(x_pos) + a_position[2] * sin(x_pos), a_position[1] * cos(y_pos) + a_position[2] * sin(y_pos), a_position[2], 1.0f);

    // Position based on x_pos and y_pos just translating horizontal and vertical
    // gl_Position = vec4(a_position[0] + x_pos, a_position[1] + y_pos, a_position[2], 1.0f);

    // Pass vertex color to fragment shader
    v_color = a_color;

    // Pass texture coordinate to fragment shader
    v_tex_coord = a_tex_coord;
}

