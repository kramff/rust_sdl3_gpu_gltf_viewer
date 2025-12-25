#version 460

// To compile:
// glslc -fshader-stage=fragment shaders/fragment.glsl -o shaders/fragment.spv

layout (location = 0) in vec4 v_color;
layout (location = 1) in vec2 v_tex_coord;
layout (location = 0) out vec4 FragColor;

layout(std140, set = 3, binding = 0) uniform UniformBlock {
    float time;
};

// layout(std140, set = 3, binding = 1) uniform UniformBlock2 {
//     // gsampler2d texture_sampler;
//     sampler texture_sampler;
// }; 
layout(binding = 1) uniform sampler2D texture_sampler;

void main()
{
    // TODO - figure out texture
    // Color based on texture
    vec4 texture_color = texture(texture_sampler, v_tex_coord);
    FragColor = texture_color;

    // Color with flashing based on time uniform
    // float pulse = sin(time * 2.0) * 0.5 + 0.5; // range [0, 1]
    // FragColor = vec4(v_color.rgb * (0.8 + pulse * 0.5), v_color.a);

    // Color based on vertex color
    // FragColor = vec4(v_color.rgba);
}
