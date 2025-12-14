#version 460

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec4 a_color;
layout (location = 0) out vec4 v_color;

layout(std140, set = 1, binding = 0) uniform UniformBlock {
    float x_pos;
    float y_pos;
};

void main()
{
    gl_Position = vec4(a_position[0] * cos(x_pos) + a_position[2] * sin(x_pos), a_position[1] * cos(y_pos) + a_position[2] * sin(y_pos), a_position[2], 1.0f);
    // gl_Position = vec4(a_position[0] + x_pos, a_position[1] + y_pos, a_position[2], 1.0f);
    v_color = a_color;
}

